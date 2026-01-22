#!/usr/bin/env python3
"""
Speaker-Invariant Detector: Layer-wise Evaluation (Parallel Version)

이 스크립트는 WavLM-Large의 모든 레이어(25개: embedding + 24 transformer layers)에서
feature를 추출하여, 각 레이어별로 Speaker-Invariant Detector를 학습하고 평가합니다.

** 병렬화 최적화 **
- 모든 레이어 features를 한 번의 forward pass로 동시 추출
- joblib을 사용한 25개 레이어 detector 병렬 학습
- 모든 (레이어, 데이터셋) 조합 병렬 평가

Training Data: ASVspoof 2019 LA train
Test Data: ASVspoof 2019 LA eval, ASVspoof 2021 DF eval, WaveFake, In-The-Wild

Output: results/layerwise_speaker_invariant_results.csv
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import librosa
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# ML Libraries
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# Deep Learning
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# Progress bar
from tqdm.auto import tqdm

# Warnings
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # ASVspoof2019 LA
    LA_ROOT = Path("/mnt/ddn/kyudan/Deepfake-speech/LA")
    LA_TRAIN_AUDIO = LA_ROOT / "ASVspoof2019_LA_train" / "flac"
    LA_DEV_AUDIO = LA_ROOT / "ASVspoof2019_LA_dev" / "flac"
    LA_EVAL_AUDIO = LA_ROOT / "ASVspoof2019_LA_eval" / "flac"
    LA_PROTOCOLS = LA_ROOT / "ASVspoof2019_LA_cm_protocols"

    # ASVspoof2021 DF
    DF_ROOT = Path("/mnt/tmp/Deepfake-speech/data/ASVspoof2021")

    # WaveFake
    WAVEFAKE_ROOT = Path("/mnt/tmp/Deepfake-speech/data/WaveFake")

    # In-The-Wild
    INTHEWILD_ROOT = Path("/mnt/tmp/Deepfake-speech/data/InTheWild_hf")

    # Models
    MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
    TRAINED_DIR = MODELS_DIR / "trained"

    # Results
    RESULTS_DIR = Path("/mnt/tmp/Deepfake-speech/results")

    # Feature cache
    CACHE_DIR = Path("/mnt/tmp/Deepfake-speech/cache")


# Create directories
Config.TRAINED_DIR.mkdir(parents=True, exist_ok=True)
Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Loaders (from cross_dataset_eval.py)
# =============================================================================

def load_asvspoof_protocol(protocol_path: str) -> pd.DataFrame:
    """
    ASVspoof2019 protocol 파일을 로드합니다.
    Protocol format: SPEAKER_ID FILE_ID - ATTACK_TYPE LABEL
    """
    data = []

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id = parts[0]
                file_id = parts[1]
                attack_type = parts[3]
                label = parts[4]

                data.append({
                    'speaker_id': speaker_id,
                    'file_id': file_id,
                    'attack_type': attack_type,
                    'label': label
                })

    df = pd.DataFrame(data)
    return df


def build_asvspoof_dataframe(
    protocol_path: str,
    audio_dir: str,
    audio_extension: str = '.flac'
) -> pd.DataFrame:
    """ASVspoof2019 데이터셋 DataFrame을 생성합니다."""
    df = load_asvspoof_protocol(protocol_path)
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'bonafide' else 1)

    audio_dir = Path(audio_dir)
    df['audio_path'] = df['file_id'].apply(lambda x: str(audio_dir / f"{x}{audio_extension}"))

    df['exists'] = df['audio_path'].apply(lambda x: Path(x).exists())
    missing_count = (~df['exists']).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} audio files not found")

    df = df[df['exists']].copy()
    df = df.drop(columns=['exists'])

    return df


def get_asvspoof19_datasets() -> Dict[str, pd.DataFrame]:
    """ASVspoof2019 LA train/dev/eval 데이터셋 로드"""
    datasets = {}

    print("Loading ASVspoof2019 LA train...")
    datasets['train'] = build_asvspoof_dataframe(
        protocol_path=str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt"),
        audio_dir=str(Config.LA_TRAIN_AUDIO)
    )

    print("Loading ASVspoof2019 LA dev...")
    datasets['dev'] = build_asvspoof_dataframe(
        protocol_path=str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.dev.trl.txt"),
        audio_dir=str(Config.LA_DEV_AUDIO)
    )

    print("Loading ASVspoof2019 LA eval...")
    datasets['eval'] = build_asvspoof_dataframe(
        protocol_path=str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.eval.trl.txt"),
        audio_dir=str(Config.LA_EVAL_AUDIO)
    )

    return datasets


def get_asvspoof21_df_data() -> Optional[pd.DataFrame]:
    """ASVspoof2021 DF 데이터셋 로드"""
    if not Config.DF_ROOT.exists():
        print(f"Warning: ASVspoof2021 DF not found at {Config.DF_ROOT}")
        return None

    protocol_path = Config.DF_ROOT / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"

    if not protocol_path.exists():
        candidates = list(Config.DF_ROOT.glob("**/trial_metadata.txt"))
        if candidates:
            protocol_path = candidates[0]
        else:
            print("Warning: Protocol file (trial_metadata.txt) not found.")
            return None

    print(f"Loading protocol from: {protocol_path}")

    print("Scanning audio files in part folders...")
    audio_path_map = {}

    part_folders = [
        "ASVspoof2021_DF_eval_part00",
        "ASVspoof2021_DF_eval_part01",
        "ASVspoof2021_DF_eval_part02",
        "ASVspoof2021_DF_eval_part03"
    ]

    for part in part_folders:
        part_dir = Config.DF_ROOT / part
        if part_dir.exists():
            for file_path in tqdm(part_dir.rglob("*.flac"), desc=f"Scanning {part}", leave=False):
                audio_path_map[file_path.stem] = str(file_path)

    print(f"Found {len(audio_path_map)} audio files.")

    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_id = parts[1]
                label = parts[5]

                if file_id in audio_path_map:
                    data.append({
                        'file_id': file_id,
                        'label': label,
                        'binary_label': 0 if label == 'bonafide' else 1,
                        'audio_path': audio_path_map[file_id]
                    })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples (Bonafide: {(df['binary_label']==0).sum()}, Spoof: {(df['binary_label']==1).sum()})")
    return df


def get_wavefake_data(max_samples: int = None) -> Optional[pd.DataFrame]:
    """WaveFake 데이터셋 로드"""
    if not Config.WAVEFAKE_ROOT.exists():
        print(f"Warning: WaveFake not found at {Config.WAVEFAKE_ROOT}")
        return None

    data = []
    print(f"Loading WaveFake from {Config.WAVEFAKE_ROOT}...")

    # LJSpeech (Real)
    ljs_dir = Config.WAVEFAKE_ROOT / "LJSpeech-1.1" / "wavs"
    if ljs_dir.exists():
        files = list(ljs_dir.glob("*.wav"))
        print(f"  [Real] LJSpeech: found {len(files)} files")
        for f in files:
            data.append({
                'file_id': f.stem,
                'source': 'LJSpeech',
                'label': 'bonafide',
                'binary_label': 0,
                'audio_path': str(f)
            })

    # JSUT basic5000 (Real)
    jsut_target_dir = Config.WAVEFAKE_ROOT / "jsut_ver1.1" / "basic5000"
    if jsut_target_dir.exists():
        files = list(jsut_target_dir.rglob("*.wav"))
        print(f"  [Real] JSUT (basic5000): found {len(files)} files")
        for f in files:
            data.append({
                'file_id': f.stem,
                'source': 'JSUT_basic5000',
                'label': 'bonafide',
                'binary_label': 0,
                'audio_path': str(f)
            })

    # Generated (Fake)
    gen_dir = Config.WAVEFAKE_ROOT / "generated_audio"
    if not gen_dir.exists():
        gen_dir = Config.WAVEFAKE_ROOT

    fake_files = []
    for algo_path in gen_dir.iterdir():
        if algo_path.is_dir() and ("ljspeech" in algo_path.name or "jsut" in algo_path.name):
            curr_files = list(algo_path.glob("*.wav"))
            algo_name = algo_path.name
            fake_files.extend([(f, algo_name) for f in curr_files])

    print(f"  [Fake] Generated: found {len(fake_files)} files")

    for f, algo_name in fake_files:
        data.append({
            'file_id': f.stem,
            'source': algo_name,
            'label': 'spoof',
            'binary_label': 1,
            'audio_path': str(f)
        })

    if len(data) == 0:
        print("Warning: No audio files found in WaveFake directory")
        return None

    df = pd.DataFrame(data)

    if max_samples and len(df) > max_samples:
        df_real = df[df['binary_label'] == 0]
        df_fake = df[df['binary_label'] == 1]

        n_real = min(len(df_real), max_samples // 2)
        n_fake = min(len(df_fake), max_samples // 2)

        df_real = df_real.sample(n=n_real, random_state=42)
        df_fake = df_fake.sample(n=n_fake, random_state=42)

        df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  -> Subsampled to {len(df)} files")

    print(f"Total WaveFake: {len(df)} (Real: {(df['binary_label']==0).sum()}, Fake: {(df['binary_label']==1).sum()})")
    return df


def get_inthewild_data() -> Optional[pd.DataFrame]:
    """In-The-Wild 데이터셋 로드"""
    if not Config.INTHEWILD_ROOT.exists():
        print(f"Warning: In-The-Wild not found at {Config.INTHEWILD_ROOT}")
        return None

    release_dir = Config.INTHEWILD_ROOT / "release_in_the_wild"
    if not release_dir.exists():
        release_dir = Config.INTHEWILD_ROOT

    meta_path = release_dir / "meta.csv"
    if not meta_path.exists():
        print(f"Warning: meta.csv not found at {meta_path}")
        return None

    print(f"Loading In-The-Wild from {release_dir}...")

    meta_df = pd.read_csv(meta_path)

    data = []
    for _, row in meta_df.iterrows():
        file_name = row['file']
        label = row['label']

        audio_path = release_dir / file_name
        if not audio_path.exists():
            continue

        binary_label = 0 if label == 'bona-fide' else 1

        data.append({
            'file_id': Path(file_name).stem,
            'label': label,
            'binary_label': binary_label,
            'audio_path': str(audio_path)
        })

    if len(data) == 0:
        print("Warning: No audio files found in In-The-Wild directory")
        return None

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples (Real: {(df['binary_label']==0).sum()}, Fake: {(df['binary_label']==1).sum()})")
    return df


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Equal Error Rate (EER) 계산
    Returns EER as percentage (0-100)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100


def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict:
    """종합 평가 메트릭 계산"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'eer': compute_eer(y_true, y_scores)
    }
    return metrics


# =============================================================================
# Multi-GPU Feature Extraction Helper
# =============================================================================

def _extract_on_single_gpu(
    gpu_id: int,
    audio_paths: List[str],
    model_name: str = "microsoft/wavlm-large"
) -> np.ndarray:
    """
    단일 GPU에서 feature 추출 (멀티프로세스용)
    """
    device = f"cuda:{gpu_id}"

    # Load model on this GPU
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_layers = 25
    hidden_size = 1024

    all_features = []
    for path in tqdm(audio_paths, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values, output_hidden_states=True, return_dict=True)

            all_hidden = torch.stack(outputs.hidden_states, dim=0)
            pooled = all_hidden.mean(dim=2).squeeze(1)
            all_features.append(pooled.cpu().numpy())
        except Exception as e:
            print(f"GPU {gpu_id} error for {path}: {e}")
            all_features.append(np.zeros((num_layers, hidden_size)))

    return np.array(all_features)


# =============================================================================
# WavLM Layer-wise Feature Extractor (Optimized for Multi-GPU Extraction)
# =============================================================================

class WavLMLayerwiseExtractor:
    """
    WavLM-Large 모델의 모든 레이어에서 feature를 추출합니다.

    ** 최적화 포인트 **
    - 한 번의 forward pass로 모든 25개 레이어 features를 동시 추출
    - 멀티 GPU 병렬 추출 지원 (데이터 분할 방식)
    - 레이어별 캐싱으로 I/O 최소화

    WavLM-Large 구조:
    - 1 CNN feature extractor embedding (layer 0)
    - 24 transformer layers (layers 1-24)
    - Total: 25 hidden states when output_hidden_states=True
    - hidden_size: 1024
    """

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"Loading WavLM model: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # WavLM-Large: 24 transformer layers + 1 embedding = 25 hidden states
        self.num_layers = 25  # indices 0-24
        self.hidden_size = 1024

        print(f"WavLM loaded on {self.device}")
        print(f"  Total hidden states: {self.num_layers} (embedding + 24 transformer layers)")
        print(f"  Available GPUs for parallel extraction: {self.num_gpus}")

    def extract_all_layers(self, audio_path: str) -> Optional[np.ndarray]:
        """
        오디오 파일에서 모든 레이어의 feature를 추출합니다.

        Returns:
            np.ndarray of shape (25, 1024) - all layers at once
            None if extraction fails
        """
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)

            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_values,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Stack all hidden states and mean pool at once
            # outputs.hidden_states: tuple of 25 tensors, each (batch, time, 1024)
            all_hidden = torch.stack(outputs.hidden_states, dim=0)  # (25, batch, time, 1024)
            pooled = all_hidden.mean(dim=2).squeeze(1)  # (25, 1024)

            return pooled.cpu().numpy()

        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def extract_batch_all_layers(
        self,
        audio_paths: List[str],
        cache_prefix: Optional[str] = None,
        desc: str = "Extracting all layers",
        use_multi_gpu: bool = True
    ) -> np.ndarray:
        """
        모든 오디오 파일에서 모든 레이어의 feature를 한 번에 추출합니다.

        캐싱: 레이어별로 개별 파일에 저장 (예: cache_prefix_layer_0.pkl, ...)
        - 파일당 ~100MB로 관리 용이
        - 특정 레이어만 선택적 로드 가능
        - 추출 중 오류 시 부분 복구 가능

        ** 멀티 GPU 지원 **
        - use_multi_gpu=True이고 GPU가 2개 이상이면 데이터를 분할하여 병렬 추출
        - 각 GPU에서 독립적으로 WavLM 모델 로드 후 처리

        Args:
            audio_paths: 오디오 파일 경로 리스트
            cache_prefix: 캐시 파일 prefix (예: "cache/asvspoof19_train" -> "cache/asvspoof19_train_layer_0.pkl")
            desc: tqdm description
            use_multi_gpu: 멀티 GPU 사용 여부 (기본: True)

        Returns:
            np.ndarray of shape (N_samples, 25, 1024)
        """
        # Check if all layer caches exist
        all_cached = True
        if cache_prefix:
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                if not Path(cache_path).exists():
                    all_cached = False
                    break
        else:
            all_cached = False

        # If all cached, load from cache
        if all_cached:
            print(f"Loading cached features (all {self.num_layers} layers) from {cache_prefix}_layer_*.pkl")
            all_features = []
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                with open(cache_path, 'rb') as f:
                    layer_features = pickle.load(f)
                all_features.append(layer_features)
            # Stack: list of (N, 1024) -> (25, N, 1024) -> transpose to (N, 25, 1024)
            all_features = np.stack(all_features, axis=1)  # (N, 25, 1024)
            print(f"  Loaded shape: {all_features.shape}")
            return all_features

        # Extract features
        print(f"Extracting features for {len(audio_paths)} files...")

        # Multi-GPU extraction
        if use_multi_gpu and self.num_gpus >= 2:
            print(f"  Using {self.num_gpus} GPUs for parallel extraction...")
            all_features = self._extract_multi_gpu(audio_paths)
        else:
            # Single GPU extraction
            all_features = []
            for path in tqdm(audio_paths, desc=desc):
                layer_features = self.extract_all_layers(path)
                if layer_features is not None:
                    all_features.append(layer_features)
                else:
                    all_features.append(np.zeros((self.num_layers, self.hidden_size)))
            all_features = np.array(all_features)  # (N, 25, 1024)

        # Save cache per layer
        if cache_prefix:
            Path(cache_prefix).parent.mkdir(parents=True, exist_ok=True)
            total_size = 0
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                layer_data = all_features[:, layer_idx, :]  # (N, 1024)
                with open(cache_path, 'wb') as f:
                    pickle.dump(layer_data, f)
                total_size += layer_data.nbytes
            print(f"Features cached to {cache_prefix}_layer_*.pkl")
            print(f"  Shape: {all_features.shape}, Total size: {total_size / 1e9:.2f} GB ({self.num_layers} files)")

        return all_features

    def _extract_multi_gpu(self, audio_paths: List[str]) -> np.ndarray:
        """
        멀티 GPU를 사용하여 병렬로 feature 추출

        데이터를 GPU 수만큼 분할하여 각 GPU에서 독립적으로 처리
        """
        from torch.multiprocessing import spawn, set_start_method
        import torch.multiprocessing as tmp

        # Split data across GPUs
        n_samples = len(audio_paths)
        chunk_size = (n_samples + self.num_gpus - 1) // self.num_gpus

        chunks = []
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_samples)
            if start_idx < n_samples:
                chunks.append(audio_paths[start_idx:end_idx])

        print(f"  Data split: {[len(c) for c in chunks]} samples per GPU")

        # Use ProcessPoolExecutor for multi-GPU extraction
        # Each process loads its own model on assigned GPU
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            for gpu_id, chunk in enumerate(chunks):
                future = executor.submit(
                    _extract_on_single_gpu,
                    gpu_id,
                    chunk,
                    self.model_name
                )
                futures.append(future)

            # Collect results
            results = [f.result() for f in futures]

        # Concatenate results in order
        all_features = np.concatenate(results, axis=0)
        print(f"  Multi-GPU extraction complete: {all_features.shape}")

        return all_features


# =============================================================================
# Layer-wise Speaker-Invariant Detector
# =============================================================================

class LayerwiseSpeakerInvariantDetector:
    """
    특정 WavLM 레이어의 feature를 사용하는 Speaker-Invariant Detector

    Pipeline:
    1. StandardScaler normalization
    2. Speaker centroid computation
    3. PCA on centroids -> speaker subspace
    4. Orthogonal projection: P_perp = I - U @ U.T
    5. LogisticRegression classification
    """

    def __init__(self, layer_idx: int, n_speaker_components: int = 10):
        self.layer_idx = layer_idx
        self.n_speaker_components = n_speaker_components

        self.scaler = StandardScaler()
        self.pca = None
        self.projection_matrix = None
        self.classifier = LogisticRegression(
            random_state=42,
            solver='liblinear',
            max_iter=1000
        )
        self.is_fitted = False

    def fit(
        self,
        X_features: np.ndarray,
        labels: List[int],
        speaker_ids: List[str]
    ) -> float:
        """
        Pre-extracted features로 학습합니다.

        Args:
            X_features: (N, 1024) pre-extracted features
            labels: Binary labels (0: bonafide, 1: spoof)
            speaker_ids: Speaker IDs for centroid computation

        Returns:
            Training accuracy
        """
        y = np.array(labels)

        # 1. Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # 2. Compute speaker centroids
        spk_map = {}
        for idx, spk in enumerate(speaker_ids):
            if spk not in spk_map:
                spk_map[spk] = []
            spk_map[spk].append(idx)

        speaker_centroids = []
        for spk, indices in spk_map.items():
            centroid = np.mean(X_scaled[indices], axis=0)
            speaker_centroids.append(centroid)
        speaker_centroids = np.array(speaker_centroids)

        # 3. PCA on centroids
        n_components = min(self.n_speaker_components, len(speaker_centroids) - 1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(speaker_centroids)

        # 4. Orthogonal projection matrix: P_perp = I - U @ U.T
        U = self.pca.components_.T  # (1024, n_components)
        I = np.eye(U.shape[0])
        self.projection_matrix = I - (U @ U.T)

        # 5. Project features and train classifier
        X_projected = X_scaled @ self.projection_matrix
        self.classifier.fit(X_projected, y)
        self.is_fitted = True

        return self.classifier.score(X_projected, y)

    def predict_batch(self, X_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-extracted features로 예측합니다.

        Returns:
            (predictions, scores)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X_features)
        X_projected = X_scaled @ self.projection_matrix

        predictions = self.classifier.predict(X_projected)
        probabilities = self.classifier.predict_proba(X_projected)
        scores = probabilities[:, 1]  # Spoof probability

        return predictions, scores

    def save(self, path: str):
        """모델 저장"""
        save_data = {
            'layer_idx': self.layer_idx,
            'n_speaker_components': self.n_speaker_components,
            'scaler': self.scaler,
            'pca': self.pca,
            'projection_matrix': self.projection_matrix,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model (layer {self.layer_idx}) saved to {path}")

    def load(self, path: str):
        """모델 로드"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.layer_idx = save_data['layer_idx']
        self.n_speaker_components = save_data['n_speaker_components']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.projection_matrix = save_data['projection_matrix']
        self.classifier = save_data['classifier']
        self.is_fitted = save_data['is_fitted']
        print(f"Model (layer {self.layer_idx}) loaded from {path}")


# =============================================================================
# Parallel Training & Evaluation Functions
# =============================================================================

def train_single_layer_detector(
    layer_idx: int,
    train_features: np.ndarray,  # (N, 1024) for this layer
    train_labels: List[int],
    train_speaker_ids: List[str],
    n_speaker_components: int = 10,
    save_path: Optional[str] = None
) -> Tuple[int, float, 'LayerwiseSpeakerInvariantDetector']:
    """
    단일 레이어 detector 학습 (병렬 실행용)
    """
    detector = LayerwiseSpeakerInvariantDetector(
        layer_idx=layer_idx,
        n_speaker_components=n_speaker_components
    )
    train_acc = detector.fit(train_features, train_labels, train_speaker_ids)

    if save_path:
        detector.save(save_path)

    return layer_idx, train_acc, detector


def evaluate_single_layer(
    layer_idx: int,
    dataset_name: str,
    detector: 'LayerwiseSpeakerInvariantDetector',
    eval_features: np.ndarray,  # (N, 1024) for this layer
    labels: np.ndarray
) -> Dict:
    """
    단일 (레이어, 데이터셋) 조합 평가 (병렬 실행용)
    """
    preds, scores = detector.predict_batch(eval_features)
    metrics = evaluate_detector(labels, preds, scores)
    metrics['layer'] = layer_idx
    metrics['dataset'] = dataset_name
    return metrics


# =============================================================================
# Main Execution (Parallel Version)
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Configuration
    N_SPEAKER_COMPONENTS = 10  # Fixed based on prior experiments
    NUM_LAYERS = 25  # 0=embedding, 1-24=transformer layers
    N_JOBS = min(25, mp.cpu_count())  # 병렬 작업 수

    print(f"\nParallel configuration:")
    print(f"  CPU cores available: {mp.cpu_count()}")
    print(f"  Parallel jobs for training/eval: {N_JOBS}")

    # =========================================================================
    # 1. Load Training Data (ASVspoof 2019 LA train)
    # =========================================================================
    print("\n" + "="*80)
    print("[1] Loading ASVspoof 2019 LA training data...")
    print("="*80)

    asvspoof19 = get_asvspoof19_datasets()
    train_df = asvspoof19['train']

    train_audio_paths = train_df['audio_path'].tolist()
    train_labels = train_df['binary_label'].tolist()
    train_speaker_ids = train_df['speaker_id'].tolist()

    print(f"  Training samples: {len(train_audio_paths)}")
    print(f"  Bonafide: {sum(1 for l in train_labels if l == 0)}")
    print(f"  Spoof: {sum(1 for l in train_labels if l == 1)}")
    print(f"  Speakers: {len(set(train_speaker_ids))}")

    # =========================================================================
    # 2. Load Evaluation Datasets
    # =========================================================================
    print("\n" + "="*80)
    print("[2] Loading evaluation datasets...")
    print("="*80)

    eval_datasets = {}

    # ASVspoof 2019 LA eval
    eval_datasets['ASVspoof19_LA'] = asvspoof19['eval']
    print(f"  ASVspoof19_LA: {len(eval_datasets['ASVspoof19_LA'])} samples")

    # ASVspoof 2021 DF
    # df_data = get_asvspoof21_df_data()
    # if df_data is not None:
    #     eval_datasets['ASVspoof21_DF'] = df_data
    #     print(f"  ASVspoof21_DF: {len(df_data)} samples")

    # WaveFake (subsample for efficiency)
    wf_data = get_wavefake_data()
    if wf_data is not None:
        eval_datasets['WaveFake'] = wf_data
        print(f"  WaveFake: {len(wf_data)} samples")

    # In-The-Wild
    # itw_data = get_inthewild_data()
    # if itw_data is not None:
    #     eval_datasets['In-The-Wild'] = itw_data
    #     print(f"  In-The-Wild: {len(itw_data)} samples")

    print(f"\n  Total datasets to evaluate: {list(eval_datasets.keys())}")

    # =========================================================================
    # 3. Initialize Layer-wise Feature Extractor
    # =========================================================================
    print("\n" + "="*80)
    print("[3] Initializing WavLM Layer-wise Extractor...")
    print("="*80)

    extractor = WavLMLayerwiseExtractor(device=device)

    # =========================================================================
    # 4. Extract Training Features for ALL Layers at Once
    # =========================================================================
    print("\n" + "="*80)
    print("[4] Extracting training features for ALL layers (single pass)...")
    print("="*80)

    train_cache_prefix = str(Config.CACHE_DIR / "asvspoof19_train")
    train_features_all = extractor.extract_batch_all_layers(
        train_audio_paths,
        cache_prefix=train_cache_prefix,
        desc="Extracting train features (all 25 layers)"
    )
    # train_features_all shape: (N_train, 25, 1024)
    print(f"  Train features shape: {train_features_all.shape}")

    # =========================================================================
    # 5. Train ALL Layer Detectors in PARALLEL
    # =========================================================================
    print("\n" + "="*80)
    print(f"[5] Training {NUM_LAYERS} detectors in PARALLEL (n_jobs={N_JOBS})...")
    print("="*80)

    # Prepare per-layer data
    train_tasks = []
    for layer_idx in range(NUM_LAYERS):
        save_path = str(Config.TRAINED_DIR / f"si_detector_layer_{layer_idx}.pkl")
        train_tasks.append((
            layer_idx,
            train_features_all[:, layer_idx, :],  # (N, 1024)
            train_labels,
            train_speaker_ids,
            N_SPEAKER_COMPONENTS,
            save_path
        ))

    # Parallel training using joblib
    print("  Starting parallel training...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(train_single_layer_detector)(
            layer_idx, features, labels, speaker_ids, n_comp, save_path
        )
        for layer_idx, features, labels, speaker_ids, n_comp, save_path in train_tasks
    )

    # Collect trained detectors
    detectors = {}
    for layer_idx, train_acc, detector in results:
        detectors[layer_idx] = detector
        print(f"  Layer {layer_idx:2d}: Train Acc = {train_acc:.4f}")

    # =========================================================================
    # 6. Extract Eval Features & Evaluate ALL in PARALLEL
    # =========================================================================
    print("\n" + "="*80)
    print("[6] Extracting eval features and evaluating in PARALLEL...")
    print("="*80)

    all_results = []

    for dataset_name, dataset_df in eval_datasets.items():
        print(f"\n  === {dataset_name} ({len(dataset_df)} samples) ===")

        audio_paths = dataset_df['audio_path'].tolist()
        labels = np.array(dataset_df['binary_label'].tolist())

        # Extract all layers at once
        safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        eval_cache_prefix = str(Config.CACHE_DIR / safe_name)

        eval_features_all = extractor.extract_batch_all_layers(
            audio_paths,
            cache_prefix=eval_cache_prefix,
            desc=f"Extracting {dataset_name} features (all layers)"
        )
        # eval_features_all shape: (N_eval, 25, 1024)
        print(f"    Eval features shape: {eval_features_all.shape}")

        # Parallel evaluation for all layers
        print(f"    Evaluating all {NUM_LAYERS} layers in parallel...")
        eval_tasks = [
            (layer_idx, dataset_name, detectors[layer_idx], eval_features_all[:, layer_idx, :], labels)
            for layer_idx in range(NUM_LAYERS)
        ]

        layer_results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(evaluate_single_layer)(
                layer_idx, ds_name, detector, features, lbls
            )
            for layer_idx, ds_name, detector, features, lbls in eval_tasks
        )

        # Collect and print results
        for metrics in sorted(layer_results, key=lambda x: x['layer']):
            all_results.append(metrics)
            print(f"    Layer {metrics['layer']:2d}: EER={metrics['eer']:6.2f}%, F1={metrics['f1_score']:.4f}")

    # =========================================================================
    # 7. Save Results to CSV
    # =========================================================================
    print("\n" + "="*80)
    print("[7] Saving results...")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    cols = ['layer', 'dataset', 'precision', 'recall', 'f1_score', 'eer']
    results_df = results_df[cols]

    output_path = Config.RESULTS_DIR / "layerwise_speaker_invariant_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    # =========================================================================
    # 8. Print Best Layer Summary
    # =========================================================================
    print("\n" + "="*80)
    print("BEST LAYER PER DATASET (by EER)")
    print("="*80)

    best_summary = []

    for dataset_name in eval_datasets.keys():
        subset = results_df[results_df['dataset'] == dataset_name]
        best_idx = subset['eer'].idxmin()
        best_row = subset.loc[best_idx]

        print(f"\n  {dataset_name}:")
        print(f"    Best Layer: {int(best_row['layer'])}")
        print(f"    EER:        {best_row['eer']:.2f}%")
        print(f"    F1-Score:   {best_row['f1_score']:.4f}")
        print(f"    Precision:  {best_row['precision']:.4f}")
        print(f"    Recall:     {best_row['recall']:.4f}")

        best_summary.append({
            'dataset': dataset_name,
            'best_layer': int(best_row['layer']),
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1_score': best_row['f1_score'],
            'eer': best_row['eer']
        })

    # Save best layer summary
    best_df = pd.DataFrame(best_summary)
    best_output_path = Config.RESULTS_DIR / "layerwise_best_layer_summary.csv"
    best_df.to_csv(best_output_path, index=False)
    print(f"\n  Best layer summary saved to {best_output_path}")

    print("\n" + "="*80)
    print("[Done]")
    print("="*80)


if __name__ == "__main__":
    main()
