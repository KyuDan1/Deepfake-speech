import pandas as pd
import os
import time
import subprocess
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import numpy as np
import librosa
import umap
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from transformers import WhisperProcessor, WhisperModel, WavLMModel, Wav2Vec2FeatureExtractor
import torchaudio
import matplotlib.cm as cm

# ==========================================
# 1. 설정 및 데이터 준비
# ==========================================

# 경로 설정
LIBRISPEECH_ROOT = "./my_librispeech/LibriSpeech"
LIBRISPEECH_SUBSET = "test-clean"
SPEAKERS_FILE = Path(LIBRISPEECH_ROOT) / "SPEAKERS.TXT"
CUDA_DEVICE = "0"

# ★★★ 샘플 수 설정 (여기서 조절) ★★★
N_SAMPLES = 1000

OUTPUT_DIR = Path(f"generated_results_f5_tts_{N_SAMPLES}").resolve()
PROMPT_WAV_DIR = Path("prompt_wav_files").resolve()
FIGURE_DIR = Path(f"analysis_figures_f5_tts_{N_SAMPLES}").resolve()

# 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_WAV_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# SPEAKERS.TXT 파싱하여 speaker_id -> gender 매핑 생성
def parse_speakers_file(filepath):
    speaker_gender = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                try:
                    spk_id = int(parts[0])
                    gender = parts[1].strip()
                    speaker_gender[spk_id] = gender
                except ValueError:
                    continue
    return speaker_gender

SPEAKER_GENDER_MAP = parse_speakers_file(SPEAKERS_FILE)
print(f"✅ SPEAKERS.TXT 로드 완료: {len(SPEAKER_GENDER_MAP)}명의 화자 정보")

# 라이브러리 로드
from libri_dataframe import build_librispeech_dataframe

print("=== 데이터프레임 구축 및 샘플링 ===")
dataframe = build_librispeech_dataframe(
    librispeech_root=LIBRISPEECH_ROOT,
    subset=LIBRISPEECH_SUBSET,
)

# [핵심] 다양한 화자가 섞이도록 랜덤 샘플링
# dataframe_sampled = dataframe.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
dataframe_sampled = dataframe[:N_SAMPLES]
print(f"✅ 샘플링 완료: {N_SAMPLES}개 파일 (화자 수: {len(dataframe_sampled['speaker_id'].unique())}명)")


# ==========================================
# 2. 오디오 생성 및 데이터셋 구성 (F5-TTS)
# ==========================================

def resolve_path(path_str): return str(Path(path_str).resolve())
def normalize_text(text):
    text = text.lower().strip()
    if not text.endswith(('.', '!', '?', ',')): text += '.'
    return text
def convert_to_wav(input_path, output_path):
    try:
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)
    except: pass

input_data = []

# 생성 목록 준비
for idx, row in enumerate(dataframe_sampled.to_dict('records')):
    spk_id = row['speaker_id']
    output_path = OUTPUT_DIR / f"gen_random_{idx:03d}_{spk_id}.wav"
    
    # 프롬프트용 WAV 변환
    prompt_wav = PROMPT_WAV_DIR / f"prompt_{spk_id}.wav"
    if not prompt_wav.exists():
        convert_to_wav(resolve_path(row['audio_path']), str(prompt_wav))

    input_data.append({
        "idx": idx,
        "speaker_id": spk_id,  # Speaker ID 추가
        "original_path": resolve_path(row['audio_path']),
        "output_path": str(output_path),
        "text": row['transcript'],
        "prompt_wav": str(prompt_wav)
    })

print("=== 오디오 생성 확인 (F5-TTS) ===")
# 이미 생성된 파일은 건너뛰고 없는 것만 생성
generation_queue = [item for item in input_data if not os.path.exists(item['output_path'])]

if generation_queue:
    print(f"새로 생성할 파일: {len(generation_queue)}개")
    for item in generation_queue:
        print(f"Generating {Path(item['output_path']).name}...")
        cmd = [
            "f5-tts_infer-cli", "-m", "F5TTS_v1_Base",
            "-r", item['prompt_wav'], "-s", normalize_text(item['text']),
            "-t", normalize_text(item['text']), "-w", item['output_path']
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
        subprocess.run(cmd, capture_output=True, env=env)
else:
    print("모든 오디오 파일이 이미 존재합니다. 생성을 건너뜁니다.")


# ==========================================
# 3. 공통 시각화 함수 (화살표 그리기 포함)
# ==========================================

def plot_and_save(X_embedded, y, filenames, title, filename_suffix):
    plt.figure(figsize=(14, 11))
    
    target_groups = ["Original (Real)", "Gen (F5-TTS)"]
    colors = ['blue', 'green']
    markers = ['o', '^']

    # 1. 점 찍기
    for i, group_name in enumerate(target_groups):
        mask = (y == i)
        plt.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            c=colors[i], label=group_name, marker=markers[i],
            s=100, alpha=0.7, edgecolors='white', zorder=2
        )

    # 2. 화살표 그리기
    unique_names = sorted(list(set(filenames)))
    for name in unique_names:
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        
        if indices_real and indices_gen:
            start = X_embedded[indices_real[0]]
            end = X_embedded[indices_gen[0]]
            plt.annotate("", xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=1),
                         zorder=1)

    plt.title(f"{title} (n={N_SAMPLES})", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = FIGURE_DIR / f"{filename_suffix}_{N_SAMPLES}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.close()


def plot_by_speaker(X_embedded, y, filenames, speaker_ids, title, filename_suffix):
    """Speaker ID별로 색상 구분, Real은 Blues 계열, Gen은 Reds 계열로 분리"""
    plt.figure(figsize=(16, 12))
    
    unique_speakers = sorted(list(set(speaker_ids)))
    n_speakers = len(unique_speakers)
    
    # Real: Blues 계열, Gen: Oranges/Reds 계열 - 각각 speaker별로 다른 색상
    cmap_real = cm.get_cmap('Blues', n_speakers + 3)  # +3으로 너무 연한 색 피하기
    cmap_gen = cm.get_cmap('Oranges', n_speakers + 3)
    
    speaker_color_real = {spk: cmap_real(i + 3) for i, spk in enumerate(unique_speakers)}
    speaker_color_gen = {spk: cmap_gen(i + 3) for i, spk in enumerate(unique_speakers)}
    
    # Real (원) 그리기 - Blues 계열
    for spk in unique_speakers:
        mask = np.array([(s == spk and l == 0) for s, l in zip(speaker_ids, y)])
        if mask.any():
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                       c=[speaker_color_real[spk]], marker='o', s=100, alpha=0.8,
                       edgecolors='black', linewidths=0.5, zorder=2)
    
    # Gen (삼각형) 그리기 - Oranges 계열
    for spk in unique_speakers:
        mask = np.array([(s == spk and l == 1) for s, l in zip(speaker_ids, y)])
        if mask.any():
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                       c=[speaker_color_gen[spk]], marker='^', s=100, alpha=0.8,
                       edgecolors='black', linewidths=0.5, zorder=2)
    
    # 화살표 그리기
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            start = X_embedded[indices_real[0]]
            end = X_embedded[indices_gen[0]]
            plt.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=0.8),
                        zorder=1)
    
    # 범례 (색상 팔레트 설명)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=12, 
               markeredgecolor='black', label='Real (Blues)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange', markersize=12, 
               markeredgecolor='black', label='Gen (Oranges)')
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.title(f"{title} (n={N_SAMPLES})\n(Real=Blues, Gen=Oranges, {n_speakers} Speakers)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = FIGURE_DIR / f"{filename_suffix}_by_speaker_{N_SAMPLES}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Speaker별 그래프 저장: {save_path}")
    plt.close()


def plot_by_gender(X_embedded, y, filenames, speaker_ids, title, filename_suffix):
    """Gender별로 색상 구분, Real/Gen도 다른 색상으로 구분"""
    plt.figure(figsize=(14, 11))
    
    # Real/Gen × Male/Female = 4가지 조합에 각각 다른 색상
    color_map = {
        ('M', 0): 'dodgerblue',      # Male Real - 밝은 파랑
        ('M', 1): 'navy',            # Male Gen - 진한 파랑
        ('F', 0): 'lightcoral',      # Female Real - 연한 빨강
        ('F', 1): 'darkred',         # Female Gen - 진한 빨강
    }
    label_map = {
        ('M', 0): 'Male (Real)',
        ('M', 1): 'Male (Gen)',
        ('F', 0): 'Female (Real)',
        ('F', 1): 'Female (Gen)',
    }
    marker_map = {0: 'o', 1: '^'}  # Real=원, Gen=삼각형
    
    # Gender 정보 매핑
    genders = [SPEAKER_GENDER_MAP.get(spk, 'Unknown') for spk in speaker_ids]
    
    # 4가지 조합 각각 그리기
    for gender in ['M', 'F']:
        for label_val in [0, 1]:  # 0=Real, 1=Gen
            mask = np.array([(g == gender and l == label_val) for g, l in zip(genders, y)])
            if mask.any():
                plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                           c=color_map[(gender, label_val)], 
                           marker=marker_map[label_val], 
                           s=100, alpha=0.7,
                           edgecolors='white', linewidths=0.5, 
                           label=label_map[(gender, label_val)], zorder=2)
    
    # 화살표 그리기
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            start = X_embedded[indices_real[0]]
            end = X_embedded[indices_gen[0]]
            plt.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.2, linewidth=0.8),
                        zorder=1)
    
    # 성별 통계
    n_male = sum(1 for g in genders if g == 'M') // 2
    n_female = sum(1 for g in genders if g == 'F') // 2
    
    plt.title(f"{title} (n={N_SAMPLES})\n(Male: {n_male}, Female: {n_female})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = FIGURE_DIR / f"{filename_suffix}_by_gender_{N_SAMPLES}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Gender별 그래프 저장: {save_path}")
    plt.close()


# ==========================================
# 4. 분석 1: Gradient Field (t-SNE & UMAP)
# ==========================================

print("\n[Analysis 1] Gradient Field Feature Extraction...")

def get_gradient_features(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80, normalized=True
        )(waveform)
        mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10., amin=1e-10, db_multiplier=1.)
        
        target_width = 256
        n_mels, width = mel_db.squeeze(0).shape
        if width > target_width: mel_db = mel_db[:, :, :target_width]
        else:
            pad = target_width - width
            mel_db = torch.nn.functional.pad(mel_db, (0, pad))
            
        mel_np = mel_db.squeeze(0).numpy()
        grad_y, grad_x = np.gradient(mel_np)
        return np.sqrt(grad_x**2 + grad_y**2).flatten()
    except: return None

features, labels, filenames, speaker_ids = [], [], [], []

for item in input_data:
    # Real
    f = get_gradient_features(item['original_path'])
    if f is not None:
        features.append(f)
        labels.append(0)
        filenames.append(f"s_{item['idx']}")
        speaker_ids.append(item['speaker_id'])
    # Gen
    if os.path.exists(item['output_path']):
        f = get_gradient_features(item['output_path'])
        if f is not None:
            features.append(f)
            labels.append(1)
            filenames.append(f"s_{item['idx']}")
            speaker_ids.append(item['speaker_id'])

X_grad = np.array(features)
y_grad = np.array(labels)
scaler = StandardScaler()
X_grad_scaled = scaler.fit_transform(X_grad)

print(f"Gradient Data Shape: {X_grad.shape}")

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_grad_scaled)
plot_and_save(X_tsne, y_grad, filenames, 
              "t-SNE (Gradient Field): Real vs Gen Flow", "Gradient_Field_tSNE")
plot_by_speaker(X_tsne, y_grad, filenames, speaker_ids,
                "t-SNE (Gradient Field): By Speaker", "Gradient_Field_tSNE")
plot_by_gender(X_tsne, y_grad, filenames, speaker_ids,
               "t-SNE (Gradient Field): By Gender", "Gradient_Field_tSNE")

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_grad_scaled)
plot_and_save(X_umap, y_grad, filenames, 
              "UMAP (Gradient Field): Real vs Gen Flow", "Gradient_Field_UMAP")
plot_by_speaker(X_umap, y_grad, filenames, speaker_ids,
                "UMAP (Gradient Field): By Speaker", "Gradient_Field_UMAP")
plot_by_gender(X_umap, y_grad, filenames, speaker_ids,
               "UMAP (Gradient Field): By Gender", "Gradient_Field_UMAP")


# ==========================================
# 5. 분석 2: Whisper Encoder (UMAP)
# ==========================================

print("\n[Analysis 2] Whisper Feature Extraction...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
whisper_model.eval()

def get_whisper_features(audio_path):
    try:
        audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        if len(audio) > 30*16000: audio = audio[:30*16000]
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            outputs = whisper_model.encoder(inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    except: return None

features_w, labels_w, filenames_w, speaker_ids_w = [], [], [], []

for item in input_data:
    f = get_whisper_features(item['original_path'])
    if f is not None:
        features_w.append(f); labels_w.append(0); filenames_w.append(f"s_{item['idx']}")
        speaker_ids_w.append(item['speaker_id'])
    if os.path.exists(item['output_path']):
        f = get_whisper_features(item['output_path'])
        if f is not None:
            features_w.append(f); labels_w.append(1); filenames_w.append(f"s_{item['idx']}")
            speaker_ids_w.append(item['speaker_id'])

X_whisper = np.array(features_w)
y_whisper = np.array(labels_w)
X_whisper_scaled = scaler.fit_transform(X_whisper)

print(f"Whisper Data Shape: {X_whisper.shape}")
reducer_w = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_whisper_umap = reducer_w.fit_transform(X_whisper_scaled)
plot_and_save(X_whisper_umap, y_whisper, filenames_w, 
              "UMAP (Whisper Features): Real vs Gen Flow", "Whisper_Encoder_UMAP")
plot_by_speaker(X_whisper_umap, y_whisper, filenames_w, speaker_ids_w,
                "UMAP (Whisper Features): By Speaker", "Whisper_Encoder_UMAP")
plot_by_gender(X_whisper_umap, y_whisper, filenames_w, speaker_ids_w,
               "UMAP (Whisper Features): By Gender", "Whisper_Encoder_UMAP")

# 메모리 정리 (WavLM 로드를 위해 Whisper 해제)
del whisper_model, whisper_processor
torch.cuda.empty_cache()
gc.collect()


# ==========================================
# 6. 분석 3: WavLM-Large (UMAP) - [NEW]
# ==========================================

print("\n[Analysis 3] WavLM-Large Feature Extraction...")

# 모델 로드
wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
wavlm_model.eval()

def get_wavlm_features(audio_path):
    try:
        audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        if len(audio) > 30*16000: audio = audio[:30*16000]
        inputs = wavlm_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            outputs = wavlm_model(inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    except Exception as e:
        print(e)
        return None

features_v, labels_v, filenames_v, speaker_ids_v = [], [], [], []

print("Extracting WavLM features...")
for i, item in enumerate(input_data):
    if (i+1) % 20 == 0: print(f"Processing {i+1}/{len(input_data)}...")
    
    # Real
    f = get_wavlm_features(item['original_path'])
    if f is not None:
        features_v.append(f); labels_v.append(0); filenames_v.append(f"s_{item['idx']}")
        speaker_ids_v.append(item['speaker_id'])
    # Gen
    if os.path.exists(item['output_path']):
        f = get_wavlm_features(item['output_path'])
        if f is not None:
            features_v.append(f); labels_v.append(1); filenames_v.append(f"s_{item['idx']}")
            speaker_ids_v.append(item['speaker_id'])

X_wavlm = np.array(features_v)
y_wavlm = np.array(labels_v)
X_wavlm_scaled = scaler.fit_transform(X_wavlm)

print(f"WavLM Data Shape: {X_wavlm.shape}")
reducer_v = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_wavlm_umap = reducer_v.fit_transform(X_wavlm_scaled)
plot_and_save(X_wavlm_umap, y_wavlm, filenames_v, 
              "UMAP (WavLM-Large): Real vs Gen Flow", "WavLM_Large_UMAP")
plot_by_speaker(X_wavlm_umap, y_wavlm, filenames_v, speaker_ids_v,
                "UMAP (WavLM-Large): By Speaker", "WavLM_Large_UMAP")
plot_by_gender(X_wavlm_umap, y_wavlm, filenames_v, speaker_ids_v,
               "UMAP (WavLM-Large): By Gender", "WavLM_Large_UMAP")

# 메모리 정리
del wavlm_model, wavlm_extractor
torch.cuda.empty_cache()
gc.collect()

print("\n=== 모든 분석 완료! 'analysis_figures' 폴더를 확인하세요. ===")