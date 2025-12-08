import pandas as pd
import os
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
from TTS.api import TTS
import matplotlib.cm as cm
import functools

# PyTorch 2.6+ weights_only 이슈 해결
import torch.serialization
torch.serialization.add_safe_globals([dict])
_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)


# ==========================================
# 1. 설정 및 데이터 준비
# ==========================================

LIBRISPEECH_ROOT = "./my_librispeech/LibriSpeech"
LIBRISPEECH_SUBSET = "test-clean"
SPEAKERS_FILE = Path(LIBRISPEECH_ROOT) / "SPEAKERS.TXT"
CUDA_DEVICE = "0"

# ★★★ 핵심 설정 ★★★
N_SPEAKERS = 2          # 샘플링할 화자 수 (1 또는 2 권장)
SAMPLES_PER_SPEAKER = 50  # 화자당 샘플 수

OUTPUT_DIR = Path("generated_results_XTTS_FewSpeakers").resolve()
PROMPT_WAV_DIR = Path("prompt_wav_files").resolve()
FIGURE_DIR = Path("analysis_figures_xtts_v2_few_speakers").resolve()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_WAV_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# SPEAKERS.TXT 파싱
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
print(f"✅ SPEAKERS.TXT 로드 완료: {len(SPEAKER_GENDER_MAP)}명")

from libri_dataframe import build_librispeech_dataframe

print("=== 데이터프레임 구축 및 샘플링 ===")
dataframe = build_librispeech_dataframe(
    librispeech_root=LIBRISPEECH_ROOT,
    subset=LIBRISPEECH_SUBSET,
)

# ★★★ 특정 화자만 샘플링 ★★★
speaker_counts = dataframe['speaker_id'].value_counts()
top_speakers = speaker_counts.head(N_SPEAKERS).index.tolist()

print(f"선택된 화자: {top_speakers}")
for spk in top_speakers:
    gender = SPEAKER_GENDER_MAP.get(spk, 'Unknown')
    print(f"  - Speaker {spk}: {gender} ({speaker_counts[spk]}개 샘플)")

dataframe_filtered = dataframe[dataframe['speaker_id'].isin(top_speakers)]
dataframe_sampled = dataframe_filtered.groupby('speaker_id').apply(
    lambda x: x.sample(n=min(SAMPLES_PER_SPEAKER, len(x)), random_state=42)
).reset_index(drop=True)

N_TOTAL = len(dataframe_sampled)
print(f"✅ 샘플링 완료: 총 {N_TOTAL}개 파일 ({N_SPEAKERS}명 화자)")


# ==========================================
# 2. 오디오 생성 (XTTS-v2)
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
for idx, row in enumerate(dataframe_sampled.to_dict('records')):
    spk_id = row['speaker_id']
    output_path = OUTPUT_DIR / f"gen_spk{spk_id}_{idx:03d}.wav"
    
    prompt_wav = PROMPT_WAV_DIR / f"prompt_{spk_id}.wav"
    if not prompt_wav.exists():
        convert_to_wav(resolve_path(row['audio_path']), str(prompt_wav))

    input_data.append({
        "idx": idx,
        "speaker_id": spk_id,
        "original_path": resolve_path(row['audio_path']),
        "output_path": str(output_path),
        "text": row['transcript'],
        "prompt_wav": str(prompt_wav)
    })

to_generate = [item for item in input_data if not os.path.exists(item['output_path'])]

if to_generate:
    print(f"=== XTTS-v2 오디오 생성 ({len(to_generate)}개) ===")
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    for item in to_generate:
        print(f"Generating {Path(item['output_path']).name}...")
        try:
            tts.tts_to_file(
                text=normalize_text(item['text']),
                file_path=item['output_path'],
                speaker_wav=item['prompt_wav'],
                language="en"
            )
        except Exception as e:
            print(f"Error: {e}")
    
    del tts
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("=== 모든 오디오가 이미 존재합니다. ===")


# ==========================================
# 3. 시각화 함수 (2-Subplot)
# ==========================================

def plot_dual_subplot(X_embedded, y, filenames, speaker_ids, title, filename_suffix):
    """2개의 subplot: Real vs Gen + Speaker별 구분"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    genders = [SPEAKER_GENDER_MAP.get(spk, 'Unknown') for spk in speaker_ids]
    unique_speakers = sorted(list(set(speaker_ids)))
    
    # ========== 왼쪽: Real vs Gen ==========
    ax1 = axes[0]
    
    mask_real = (y == 0)
    ax1.scatter(X_embedded[mask_real, 0], X_embedded[mask_real, 1],
               c='blue', label='Original (Real)', marker='o',
               s=80, alpha=0.7, edgecolors='white', zorder=2)
    
    mask_gen = (y == 1)
    ax1.scatter(X_embedded[mask_gen, 0], X_embedded[mask_gen, 1],
               c='green', label='Gen (XTTS-v2)', marker='^',
               s=80, alpha=0.7, edgecolors='white', zorder=2)
    
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            start = X_embedded[indices_real[0]]
            end = X_embedded[indices_gen[0]]
            ax1.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=0.8),
                        zorder=1)
    
    ax1.set_title("Real vs Generated", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========== 오른쪽: Speaker별 ==========
    ax2 = axes[1]
    
    speaker_colors = {
        (unique_speakers[0], 0): 'dodgerblue',
        (unique_speakers[0], 1): 'navy',
    }
    speaker_labels = {
        (unique_speakers[0], 0): f'Spk {unique_speakers[0]} ({SPEAKER_GENDER_MAP.get(unique_speakers[0], "?")})-Real',
        (unique_speakers[0], 1): f'Spk {unique_speakers[0]} ({SPEAKER_GENDER_MAP.get(unique_speakers[0], "?")})-Gen',
    }
    
    if len(unique_speakers) >= 2:
        speaker_colors[(unique_speakers[1], 0)] = 'lightcoral'
        speaker_colors[(unique_speakers[1], 1)] = 'darkred'
        speaker_labels[(unique_speakers[1], 0)] = f'Spk {unique_speakers[1]} ({SPEAKER_GENDER_MAP.get(unique_speakers[1], "?")})-Real'
        speaker_labels[(unique_speakers[1], 1)] = f'Spk {unique_speakers[1]} ({SPEAKER_GENDER_MAP.get(unique_speakers[1], "?")})-Gen'
    
    marker_map = {0: 'o', 1: '^'}
    
    for spk in unique_speakers:
        for label_val in [0, 1]:
            mask = np.array([(s == spk and l == label_val) for s, l in zip(speaker_ids, y)])
            if mask.any() and (spk, label_val) in speaker_colors:
                ax2.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                           c=speaker_colors[(spk, label_val)],
                           marker=marker_map[label_val],
                           s=80, alpha=0.7, edgecolors='white', linewidths=0.5,
                           label=speaker_labels[(spk, label_val)], zorder=2)
    
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            start = X_embedded[indices_real[0]]
            end = X_embedded[indices_gen[0]]
            ax2.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=0.8),
                        zorder=1)
    
    ax2.set_title("By Speaker (Real vs Gen)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f"{title}\n({N_SPEAKERS} Speakers, {N_TOTAL} samples)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = FIGURE_DIR / f"{filename_suffix}_{N_SPEAKERS}spk_{N_TOTAL}samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장: {save_path}")
    plt.close()


def plot_dual_by_gender(X_embedded, y, filenames, speaker_ids, title, filename_suffix):
    """2개의 subplot: Real vs Gen + Gender별 구분"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    genders = [SPEAKER_GENDER_MAP.get(spk, 'Unknown') for spk in speaker_ids]
    
    ax1 = axes[0]
    mask_real = (y == 0)
    ax1.scatter(X_embedded[mask_real, 0], X_embedded[mask_real, 1],
               c='blue', label='Original (Real)', marker='o',
               s=80, alpha=0.7, edgecolors='white', zorder=2)
    mask_gen = (y == 1)
    ax1.scatter(X_embedded[mask_gen, 0], X_embedded[mask_gen, 1],
               c='green', label='Gen (XTTS-v2)', marker='^',
               s=80, alpha=0.7, edgecolors='white', zorder=2)
    
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            ax1.annotate("", xy=X_embedded[indices_gen[0]], xytext=X_embedded[indices_real[0]],
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=0.8), zorder=1)
    
    ax1.set_title("Real vs Generated", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = axes[1]
    color_map = {
        ('M', 0): 'dodgerblue', ('M', 1): 'navy',
        ('F', 0): 'lightcoral', ('F', 1): 'darkred',
    }
    label_map = {
        ('M', 0): 'Male (Real)', ('M', 1): 'Male (Gen)',
        ('F', 0): 'Female (Real)', ('F', 1): 'Female (Gen)',
    }
    marker_map = {0: 'o', 1: '^'}
    
    for gender in ['M', 'F']:
        for label_val in [0, 1]:
            mask = np.array([(g == gender and l == label_val) for g, l in zip(genders, y)])
            if mask.any():
                ax2.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                           c=color_map[(gender, label_val)], marker=marker_map[label_val],
                           s=80, alpha=0.7, edgecolors='white', linewidths=0.5,
                           label=label_map[(gender, label_val)], zorder=2)
    
    for name in sorted(list(set(filenames))):
        indices_real = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 0]
        indices_gen = [i for i, (f, l) in enumerate(zip(filenames, y)) if f == name and l == 1]
        if indices_real and indices_gen:
            ax2.annotate("", xy=X_embedded[indices_gen[0]], xytext=X_embedded[indices_real[0]],
                        arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.3, linewidth=0.8), zorder=1)
    
    n_male = sum(1 for g in genders if g == 'M') // 2
    n_female = sum(1 for g in genders if g == 'F') // 2
    ax2.set_title(f"By Gender (M:{n_male}, F:{n_female})", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f"{title}\n({N_SPEAKERS} Speakers, {N_TOTAL} samples)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = FIGURE_DIR / f"{filename_suffix}_gender_{N_SPEAKERS}spk_{N_TOTAL}samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gender 그래프 저장: {save_path}")
    plt.close()


# ==========================================
# 4. 분석 1: Gradient Field
# ==========================================

print("\n[Analysis 1] Gradient Field...")

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
        else: mel_db = torch.nn.functional.pad(mel_db, (0, target_width - width))
        mel_np = mel_db.squeeze(0).numpy()
        grad_y, grad_x = np.gradient(mel_np)
        return np.sqrt(grad_x**2 + grad_y**2).flatten()
    except: return None

features, labels, filenames, speaker_ids = [], [], [], []
for item in input_data:
    f = get_gradient_features(item['original_path'])
    if f is not None:
        features.append(f); labels.append(0)
        filenames.append(f"s_{item['idx']}"); speaker_ids.append(item['speaker_id'])
    if os.path.exists(item['output_path']):
        f = get_gradient_features(item['output_path'])
        if f is not None:
            features.append(f); labels.append(1)
            filenames.append(f"s_{item['idx']}"); speaker_ids.append(item['speaker_id'])

X_grad = np.array(features); y_grad = np.array(labels)
scaler = StandardScaler()
X_grad_scaled = scaler.fit_transform(X_grad)
print(f"Gradient Shape: {X_grad.shape}")

tsne = TSNE(n_components=2, perplexity=min(30, len(X_grad)-1), random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_grad_scaled)
plot_dual_subplot(X_tsne, y_grad, filenames, speaker_ids, "t-SNE (Gradient)", "Gradient_tSNE")
plot_dual_by_gender(X_tsne, y_grad, filenames, speaker_ids, "t-SNE (Gradient)", "Gradient_tSNE")

reducer = umap.UMAP(n_neighbors=min(15, len(X_grad)-1), min_dist=0.3, random_state=42)
X_umap = reducer.fit_transform(X_grad_scaled)
plot_dual_subplot(X_umap, y_grad, filenames, speaker_ids, "UMAP (Gradient)", "Gradient_UMAP")
plot_dual_by_gender(X_umap, y_grad, filenames, speaker_ids, "UMAP (Gradient)", "Gradient_UMAP")


# ==========================================
# 5. 분석 2: Whisper
# ==========================================

print("\n[Analysis 2] Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
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
        features_w.append(f); labels_w.append(0)
        filenames_w.append(f"s_{item['idx']}"); speaker_ids_w.append(item['speaker_id'])
    if os.path.exists(item['output_path']):
        f = get_whisper_features(item['output_path'])
        if f is not None:
            features_w.append(f); labels_w.append(1)
            filenames_w.append(f"s_{item['idx']}"); speaker_ids_w.append(item['speaker_id'])

X_whisper = np.array(features_w); y_whisper = np.array(labels_w)
X_whisper_scaled = scaler.fit_transform(X_whisper)
print(f"Whisper Shape: {X_whisper.shape}")

reducer_w = umap.UMAP(n_neighbors=min(15, len(X_whisper)-1), min_dist=0.3, random_state=42)
X_whisper_umap = reducer_w.fit_transform(X_whisper_scaled)
plot_dual_subplot(X_whisper_umap, y_whisper, filenames_w, speaker_ids_w, "UMAP (Whisper)", "Whisper_UMAP")
plot_dual_by_gender(X_whisper_umap, y_whisper, filenames_w, speaker_ids_w, "UMAP (Whisper)", "Whisper_UMAP")

del whisper_model, whisper_processor
torch.cuda.empty_cache(); gc.collect()


# ==========================================
# 6. 분석 3: WavLM
# ==========================================

print("\n[Analysis 3] WavLM...")
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
    except: return None

features_v, labels_v, filenames_v, speaker_ids_v = [], [], [], []
for i, item in enumerate(input_data):
    if (i+1) % 20 == 0: print(f"Processing {i+1}/{len(input_data)}...")
    f = get_wavlm_features(item['original_path'])
    if f is not None:
        features_v.append(f); labels_v.append(0)
        filenames_v.append(f"s_{item['idx']}"); speaker_ids_v.append(item['speaker_id'])
    if os.path.exists(item['output_path']):
        f = get_wavlm_features(item['output_path'])
        if f is not None:
            features_v.append(f); labels_v.append(1)
            filenames_v.append(f"s_{item['idx']}"); speaker_ids_v.append(item['speaker_id'])

X_wavlm = np.array(features_v); y_wavlm = np.array(labels_v)
X_wavlm_scaled = scaler.fit_transform(X_wavlm)
print(f"WavLM Shape: {X_wavlm.shape}")

reducer_v = umap.UMAP(n_neighbors=min(15, len(X_wavlm)-1), min_dist=0.3, random_state=42)
X_wavlm_umap = reducer_v.fit_transform(X_wavlm_scaled)
plot_dual_subplot(X_wavlm_umap, y_wavlm, filenames_v, speaker_ids_v, "UMAP (WavLM)", "WavLM_UMAP")
plot_dual_by_gender(X_wavlm_umap, y_wavlm, filenames_v, speaker_ids_v, "UMAP (WavLM)", "WavLM_UMAP")

del wavlm_model, wavlm_extractor
torch.cuda.empty_cache(); gc.collect()

print(f"\n=== 완료! '{FIGURE_DIR}' 확인 ===")


