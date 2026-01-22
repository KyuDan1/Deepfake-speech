#!/usr/bin/env python3
"""
Attack Type-wise EER Evaluation for ASVspoof2019 LA Evaluation Dataset

이 스크립트는 ASVspoof2019 LA evaluation dataset에서 attack type별로 EER을 측정합니다.

평가 모델:
1. AASIST (Pre-trained)
2. RawNet2 (Pre-trained)
3. WavLM Frozen Baseline
4. WavLM Speaker Invariant (n_component=10, layer-wise evaluation)

Attack Types in ASVspoof2019 LA:
- A01-A06: Training set에 포함된 공격 (seen attacks)
- A07-A19: Evaluation set에만 있는 공격 (unseen attacks)
- bonafide: 실제 음성

Output: results/attack_type_eer_results.csv
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# ML Libraries
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# Deep Learning
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch.nn.functional as F

# Config parsing
import yaml
import json

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

    # Models
    MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
    RAWNET2_DIR = MODELS_DIR / "asvspoof2021_baseline" / "2021" / "DF" / "Baseline-RawNet2"
    AASIST_DIR = MODELS_DIR / "aasist"
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
# Dataset Loaders
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


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Equal Error Rate (EER) 계산
    Returns EER as percentage (0-100)
    """
    # Check if we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100


# =============================================================================
# WavLM Feature Extractor
# =============================================================================

class WavLMFeatureExtractor:
    """WavLM-Large 모델을 사용한 Feature Extractor"""

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading WavLM model: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print(f"WavLM loaded on {self.device}")

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        """오디오 파일에서 WavLM feature 추출 (last hidden state mean pooled)"""
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)

            features = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return features

        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def extract_batch(
        self,
        audio_paths: List[str],
        cache_path: Optional[str] = None,
        desc: str = "Extracting features"
    ) -> np.ndarray:
        """배치 feature 추출 (캐싱 지원)"""
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        features = []
        for path in tqdm(audio_paths, desc=desc):
            feat = self.extract(path)
            if feat is not None:
                features.append(feat)
            else:
                features.append(np.zeros(1024))

        features = np.array(features)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Features cached to {cache_path}")

        return features


# =============================================================================
# WavLM Layer-wise Feature Extractor (for Speaker Invariant)
# =============================================================================

class WavLMLayerwiseExtractor:
    """WavLM-Large 모델의 모든 레이어에서 feature를 추출합니다."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading WavLM model: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_layers = 25  # 0=embedding, 1-24=transformer layers
        self.hidden_size = 1024

        print(f"WavLM loaded on {self.device}")
        print(f"  Total hidden states: {self.num_layers}")

    def extract_all_layers(self, audio_path: str) -> Optional[np.ndarray]:
        """오디오 파일에서 모든 레이어의 feature를 추출합니다."""
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values, output_hidden_states=True, return_dict=True)

            all_hidden = torch.stack(outputs.hidden_states, dim=0)
            pooled = all_hidden.mean(dim=2).squeeze(1)

            return pooled.cpu().numpy()

        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def extract_batch_all_layers(
        self,
        audio_paths: List[str],
        cache_prefix: Optional[str] = None,
        desc: str = "Extracting all layers"
    ) -> np.ndarray:
        """모든 오디오 파일에서 모든 레이어의 feature를 한 번에 추출합니다."""
        # Check cache
        all_cached = True
        if cache_prefix:
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                if not Path(cache_path).exists():
                    all_cached = False
                    break
        else:
            all_cached = False

        if all_cached:
            print(f"Loading cached features from {cache_prefix}_layer_*.pkl")
            all_features = []
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                with open(cache_path, 'rb') as f:
                    layer_features = pickle.load(f)
                all_features.append(layer_features)
            all_features = np.stack(all_features, axis=1)
            print(f"  Loaded shape: {all_features.shape}")
            return all_features

        print(f"Extracting features for {len(audio_paths)} files...")
        all_features = []
        for path in tqdm(audio_paths, desc=desc):
            layer_features = self.extract_all_layers(path)
            if layer_features is not None:
                all_features.append(layer_features)
            else:
                all_features.append(np.zeros((self.num_layers, self.hidden_size)))
        all_features = np.array(all_features)

        if cache_prefix:
            Path(cache_prefix).parent.mkdir(parents=True, exist_ok=True)
            for layer_idx in range(self.num_layers):
                cache_path = f"{cache_prefix}_layer_{layer_idx}.pkl"
                layer_data = all_features[:, layer_idx, :]
                with open(cache_path, 'wb') as f:
                    pickle.dump(layer_data, f)
            print(f"Features cached to {cache_prefix}_layer_*.pkl")

        return all_features


# =============================================================================
# Model Classes
# =============================================================================

class WavLMFrozenBaseline:
    """WavLM (Frozen) + Logistic Regression Baseline"""

    def __init__(self, wavlm_extractor: WavLMFeatureExtractor = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if wavlm_extractor is None:
            self.wavlm = WavLMFeatureExtractor(device=self.device)
        else:
            self.wavlm = wavlm_extractor

        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_fitted = False

    def fit(self, audio_paths: List[str], labels: List[int], cache_path: Optional[str] = None) -> float:
        print("Extracting features for training...")
        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        y = np.array(labels)

        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        print("Training Logistic Regression...")
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        train_acc = self.classifier.score(X_scaled, y)
        print(f"Training complete. Train accuracy: {train_acc:.4f}")
        return train_acc

    def predict_batch(self, audio_paths: List[str], cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.transform(X)

        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        scores = probabilities[:, 1]

        return predictions, scores

    def save(self, path: str):
        save_data = {'scaler': self.scaler, 'classifier': self.classifier, 'is_fitted': self.is_fitted}
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.scaler = save_data['scaler']
        self.classifier = save_data['classifier']
        self.is_fitted = save_data['is_fitted']
        print(f"Model loaded from {path}")


class SpeakerInvariantDetector:
    """Speaker-Invariant Deepfake Detector"""

    def __init__(self, n_speaker_components: int = 10, wavlm_extractor: WavLMFeatureExtractor = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_speaker_components = n_speaker_components

        if wavlm_extractor is None:
            self.wavlm = WavLMFeatureExtractor(device=self.device)
        else:
            self.wavlm = wavlm_extractor

        self.pca = None
        self.projection_matrix = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def fit(self, audio_paths: List[str], labels: List[int], speaker_ids: List[str], cache_path: Optional[str] = None) -> float:
        print(f"Training Speaker-Invariant Detector (n_components={self.n_speaker_components})...")
        X_raw = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X_raw)

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

        n_components = min(self.n_speaker_components, len(speaker_centroids) - 1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(speaker_centroids)

        U = self.pca.components_.T
        I = np.eye(U.shape[0])
        self.projection_matrix = I - (U @ U.T)

        X_projected = X_scaled @ self.projection_matrix
        self.classifier.fit(X_projected, y)
        self.is_fitted = True

        train_acc = self.classifier.score(X_projected, y)
        print(f"Training complete. Train accuracy: {train_acc:.4f}")
        return train_acc

    def predict_batch(self, audio_paths: List[str], cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.transform(X)
        X_projected = X_scaled @ self.projection_matrix

        predictions = self.classifier.predict(X_projected)
        probabilities = self.classifier.predict_proba(X_projected)
        scores = probabilities[:, 1]

        return predictions, scores

    def save(self, path: str):
        save_data = {
            'n_speaker_components': self.n_speaker_components,
            'scaler': self.scaler, 'pca': self.pca,
            'projection_matrix': self.projection_matrix,
            'classifier': self.classifier, 'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.n_speaker_components = save_data['n_speaker_components']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.projection_matrix = save_data['projection_matrix']
        self.classifier = save_data['classifier']
        self.is_fitted = save_data['is_fitted']
        print(f"Model loaded from {path}")


class LayerwiseSpeakerInvariantDetector:
    """특정 WavLM 레이어의 feature를 사용하는 Speaker-Invariant Detector"""

    def __init__(self, layer_idx: int, n_speaker_components: int = 10):
        self.layer_idx = layer_idx
        self.n_speaker_components = n_speaker_components

        self.scaler = StandardScaler()
        self.pca = None
        self.projection_matrix = None
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def fit(self, X_features: np.ndarray, labels: List[int], speaker_ids: List[str]) -> float:
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X_features)

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

        n_components = min(self.n_speaker_components, len(speaker_centroids) - 1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(speaker_centroids)

        U = self.pca.components_.T
        I = np.eye(U.shape[0])
        self.projection_matrix = I - (U @ U.T)

        X_projected = X_scaled @ self.projection_matrix
        self.classifier.fit(X_projected, y)
        self.is_fitted = True

        return self.classifier.score(X_projected, y)

    def predict_batch(self, X_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X_features)
        X_projected = X_scaled @ self.projection_matrix

        predictions = self.classifier.predict(X_projected)
        probabilities = self.classifier.predict_proba(X_projected)
        scores = probabilities[:, 1]

        return predictions, scores

    def save(self, path: str):
        save_data = {
            'layer_idx': self.layer_idx,
            'n_speaker_components': self.n_speaker_components,
            'scaler': self.scaler, 'pca': self.pca,
            'projection_matrix': self.projection_matrix,
            'classifier': self.classifier, 'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.layer_idx = save_data['layer_idx']
        self.n_speaker_components = save_data['n_speaker_components']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.projection_matrix = save_data['projection_matrix']
        self.classifier = save_data['classifier']
        self.is_fitted = save_data['is_fitted']


class RawNet2FrozenBaseline:
    """RawNet2 (Frozen) Baseline"""

    def __init__(self, model_dir: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = Path(model_dir) if model_dir else Config.MODELS_DIR / "asvspoof2021_baseline"
        self.repo_dir = self.root_dir / "2021"
        self.code_dir = self.repo_dir / "DF" / "Baseline-RawNet2"
        self.nb_samp = 64000
        self.model = None
        self.is_loaded = False
        self.weights_path = self.code_dir / "models" / "model_DF_CCE_100_32_0.0001" / "epoch_70.pth"

        self._load_model()

    def _load_model(self):
        try:
            if str(self.code_dir) not in sys.path:
                sys.path.insert(0, str(self.code_dir))

            from model import RawNet

            config_path = self.code_dir / "model_config_RawNet2.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_config = config['model']
            else:
                model_config = {
                    'nb_samp': 64000, 'first_conv': 128, 'in_channels': 1,
                    'filts': [128, [128, 128], [128, 512], [512, 512]],
                    'blocks': [2, 4], 'nb_fc_node': 1024, 'gru_node': 1024,
                    'nb_gru_layer': 3, 'nb_classes': 2
                }

            self.model = RawNet(model_config, self.device).to(self.device)

            if not self.weights_path.exists():
                print(f"[Error] Weights not found: {self.weights_path}")
                return

            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                print(f"[RawNet2] Model loaded successfully!")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    adjusted_config = model_config.copy()
                    adjusted_config['first_conv'] = 20
                    adjusted_config['filts'] = [20, [20, 20], [20, 128], [128, 128]]
                    self.model = RawNet(adjusted_config, self.device).to(self.device)
                    self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                    print(f"[RawNet2] Model loaded with adjusted config!")
                else:
                    raise e

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.is_loaded = True

        except Exception as e:
            print(f"[Error] Loading RawNet2: {e}")

    def predict(self, audio_path: str):
        if not self.is_loaded:
            return None
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            if len(audio) < self.nb_samp:
                num_repeats = int(self.nb_samp / len(audio)) + 1
                audio = np.tile(audio, num_repeats)[:self.nb_samp]
            else:
                audio = audio[:self.nb_samp]

            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(audio_tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            spoof_prob = probs[0]
            return {'label': 1 if spoof_prob > 0.5 else 0, 'score': float(spoof_prob)}
        except Exception as e:
            print(f"Error: {e}")
            return None

    def predict_batch(self, audio_paths: list):
        preds, scores = [], []
        for path in tqdm(audio_paths, desc="RawNet2 Inference"):
            res = self.predict(path)
            if res:
                preds.append(res['label'])
                scores.append(res['score'])
            else:
                preds.append(0)
                scores.append(0.0)
        return np.array(preds), np.array(scores)


class AASISTFrozenBaseline:
    """AASIST (Frozen) Baseline"""

    def __init__(self, model_dir: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir) if model_dir else Config.AASIST_DIR
        self.model = None
        self.config = None
        self.is_loaded = False
        self.nb_samp = 64600

        if self.model_dir.exists():
            self._load_model()
        else:
            print(f"Warning: AASIST not found at {self.model_dir}")

    def _load_config(self):
        config_path = self.model_dir / "config" / "AASIST.conf"
        if not config_path.exists():
            return {
                'nb_samp': 64600, 'first_conv': 128,
                'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                'gat_dims': [64, 32], 'pool_ratios': [0.5, 0.7, 0.5, 0.5],
                'temperatures': [2.0, 2.0, 100.0, 100.0]
            }
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('model_config', config)

    def _load_model(self):
        try:
            models_dir_str = str(self.model_dir / "models")
            if models_dir_str not in sys.path:
                sys.path.insert(0, models_dir_str)

            from AASIST import Model as AASIST

            self.config = self._load_config()
            self.nb_samp = self.config.get('nb_samp', 64600)

            weights_path = self.model_dir / "models" / "weights" / "AASIST.pth"
            if not weights_path.exists():
                print(f"Error: AASIST weights not found at {weights_path}")
                return

            self.model = AASIST(self.config).to(self.device)
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            self.is_loaded = True
            print(f"AASIST loaded from {weights_path}")

        except Exception as e:
            print(f"Error loading AASIST: {e}")

    def predict(self, audio_path: str) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            if len(audio) < self.nb_samp:
                num_repeats = int(self.nb_samp / len(audio)) + 1
                audio = np.tile(audio, num_repeats)[:self.nb_samp]
            else:
                audio = audio[:self.nb_samp]

            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, output = self.model(audio_tensor)
                probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

            spoof_prob = probs[0]
            return {'label': 1 if spoof_prob > 0.5 else 0, 'score': float(spoof_prob)}

        except Exception as e:
            print(f"Error: {e}")
            return None

    def predict_batch(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            return np.zeros(len(audio_paths)), np.zeros(len(audio_paths))

        preds, scores = [], []
        for path in tqdm(audio_paths, desc="AASIST Inference"):
            res = self.predict(path)
            if res:
                preds.append(res['label'])
                scores.append(res['score'])
            else:
                preds.append(0)
                scores.append(0.0)
        return np.array(preds), np.array(scores)


# =============================================================================
# Attack Type-wise Evaluation Functions
# =============================================================================

def evaluate_attack_type_eer(
    model_name: str,
    attack_type: str,
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict:
    """특정 attack type에 대한 EER 계산"""
    eer = compute_eer(y_true, y_scores)
    return {
        'model': model_name,
        'attack_type': attack_type,
        'n_samples': len(y_true),
        'n_bonafide': int((y_true == 0).sum()),
        'n_spoof': int((y_true == 1).sum()),
        'eer': eer
    }


def get_bonafide_indices(eval_df: pd.DataFrame) -> np.ndarray:
    """bonafide 샘플의 인덱스 반환"""
    return np.where(eval_df['attack_type'] == '-')[0]


def get_attack_indices(eval_df: pd.DataFrame, attack_type: str) -> np.ndarray:
    """특정 attack type 샘플의 인덱스 반환"""
    return np.where(eval_df['attack_type'] == attack_type)[0]


# =============================================================================
# Main Execution
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Configuration
    N_SPEAKER_COMPONENTS = 10
    NUM_LAYERS = 25
    N_JOBS = min(25, mp.cpu_count())

    print(f"\nConfiguration:")
    print(f"  N_SPEAKER_COMPONENTS: {N_SPEAKER_COMPONENTS}")
    print(f"  NUM_LAYERS for layer-wise: {NUM_LAYERS}")
    print(f"  Parallel jobs: {N_JOBS}")

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("[1] Loading ASVspoof 2019 LA datasets...")
    print("="*80)

    asvspoof19 = get_asvspoof19_datasets()
    train_df = asvspoof19['train']
    eval_df = asvspoof19['eval']

    train_audio_paths = train_df['audio_path'].tolist()
    train_labels = train_df['binary_label'].tolist()
    train_speaker_ids = train_df['speaker_id'].tolist()

    eval_audio_paths = eval_df['audio_path'].tolist()
    eval_labels = np.array(eval_df['binary_label'].tolist())

    # Attack types in evaluation set
    attack_types = sorted([at for at in eval_df['attack_type'].unique() if at != '-'])
    print(f"\n  Attack types in eval set: {attack_types}")
    print(f"  Total eval samples: {len(eval_df)}")
    print(f"  Bonafide samples: {(eval_df['attack_type'] == '-').sum()}")

    # =========================================================================
    # 2. Initialize Models
    # =========================================================================
    print("\n" + "="*80)
    print("[2] Initializing models...")
    print("="*80)

    # 2.1 WavLM Feature Extractor (shared)
    print("\n  Initializing WavLM Feature Extractor...")
    wavlm_extractor = WavLMFeatureExtractor(device=device)

    # 2.2 WavLM Frozen Baseline
    print("\n  Loading WavLM Frozen Baseline...")
    wavlm_baseline = WavLMFrozenBaseline(wavlm_extractor=wavlm_extractor, device=device)
    wavlm_model_path = Config.TRAINED_DIR / "wavlm_frozen_baseline.pkl"
    train_cache_path = str(Config.CACHE_DIR / "asvspoof19_train_wavlm_features.pkl")

    if wavlm_model_path.exists():
        wavlm_baseline.load(str(wavlm_model_path))
    else:
        print("  Training WavLM Frozen Baseline...")
        wavlm_baseline.fit(train_audio_paths, train_labels, cache_path=train_cache_path)
        wavlm_baseline.save(str(wavlm_model_path))

    # 2.3 Speaker-Invariant Detector (n=10)
    print("\n  Loading Speaker-Invariant Detector (n=10)...")
    si_detector = SpeakerInvariantDetector(n_speaker_components=N_SPEAKER_COMPONENTS, wavlm_extractor=wavlm_extractor, device=device)
    si_model_path = Config.TRAINED_DIR / f"si_detector_n{N_SPEAKER_COMPONENTS}.pkl"

    if si_model_path.exists():
        si_detector.load(str(si_model_path))
    else:
        print("  Training Speaker-Invariant Detector...")
        si_detector.fit(train_audio_paths, train_labels, train_speaker_ids, cache_path=train_cache_path)
        si_detector.save(str(si_model_path))

    # 2.4 RawNet2
    print("\n  Initializing RawNet2...")
    rawnet2_baseline = RawNet2FrozenBaseline(device=device)

    # 2.5 AASIST
    print("\n  Initializing AASIST...")
    aasist_baseline = AASISTFrozenBaseline(device=device)

    # 2.6 WavLM Layer-wise Extractor (for layer-wise SI evaluation)
    print("\n  Initializing WavLM Layer-wise Extractor...")
    layerwise_extractor = WavLMLayerwiseExtractor(device=device)

    # =========================================================================
    # 3. Extract Evaluation Features
    # =========================================================================
    print("\n" + "="*80)
    print("[3] Extracting evaluation features...")
    print("="*80)

    # WavLM features (for WavLM Frozen and SI)
    eval_cache_path = str(Config.CACHE_DIR / "asvspoof19_eval_wavlm_features.pkl")
    eval_features = wavlm_extractor.extract_batch(eval_audio_paths, cache_path=eval_cache_path, desc="Extracting eval features")

    # Layer-wise features (for SI layer-wise)
    eval_layerwise_cache_prefix = str(Config.CACHE_DIR / "asvspoof19_eval")
    eval_features_all_layers = layerwise_extractor.extract_batch_all_layers(
        eval_audio_paths,
        cache_prefix=eval_layerwise_cache_prefix,
        desc="Extracting eval features (all layers)"
    )
    print(f"  Eval features shape (all layers): {eval_features_all_layers.shape}")

    # =========================================================================
    # 4. Train Layer-wise Speaker Invariant Detectors
    # =========================================================================
    print("\n" + "="*80)
    print("[4] Training layer-wise Speaker-Invariant Detectors...")
    print("="*80)

    # Extract training features for all layers
    train_layerwise_cache_prefix = str(Config.CACHE_DIR / "asvspoof19_train")
    train_features_all_layers = layerwise_extractor.extract_batch_all_layers(
        train_audio_paths,
        cache_prefix=train_layerwise_cache_prefix,
        desc="Extracting train features (all layers)"
    )
    print(f"  Train features shape (all layers): {train_features_all_layers.shape}")

    # Train detectors for each layer
    layerwise_detectors = {}
    for layer_idx in tqdm(range(NUM_LAYERS), desc="Training layer-wise detectors"):
        detector_path = Config.TRAINED_DIR / f"si_detector_layer_{layer_idx}_n{N_SPEAKER_COMPONENTS}.pkl"

        detector = LayerwiseSpeakerInvariantDetector(layer_idx=layer_idx, n_speaker_components=N_SPEAKER_COMPONENTS)

        if detector_path.exists():
            detector.load(str(detector_path))
        else:
            train_acc = detector.fit(
                train_features_all_layers[:, layer_idx, :],
                train_labels,
                train_speaker_ids
            )
            detector.save(str(detector_path))

        layerwise_detectors[layer_idx] = detector

    print(f"  Trained {len(layerwise_detectors)} layer-wise detectors")

    # =========================================================================
    # 5. Evaluate All Models on Each Attack Type
    # =========================================================================
    print("\n" + "="*80)
    print("[5] Evaluating models on each attack type...")
    print("="*80)

    all_results = []

    # Get bonafide indices (used for all attack type evaluations)
    bonafide_mask = eval_df['attack_type'] == '-'
    bonafide_indices = np.where(bonafide_mask)[0]

    # Get all predictions and scores first
    print("\n  Getting predictions from all models...")

    # WavLM Frozen
    print("    WavLM Frozen...")
    wavlm_X_scaled = wavlm_baseline.scaler.transform(eval_features)
    wavlm_scores = wavlm_baseline.classifier.predict_proba(wavlm_X_scaled)[:, 1]

    # Speaker Invariant
    print("    Speaker Invariant (n=10)...")
    si_X_scaled = si_detector.scaler.transform(eval_features)
    si_X_projected = si_X_scaled @ si_detector.projection_matrix
    si_scores = si_detector.classifier.predict_proba(si_X_projected)[:, 1]

    # RawNet2
    print("    RawNet2...")
    if rawnet2_baseline.is_loaded:
        _, rawnet2_scores = rawnet2_baseline.predict_batch(eval_audio_paths)
    else:
        rawnet2_scores = np.zeros(len(eval_audio_paths))

    # AASIST
    print("    AASIST...")
    if aasist_baseline.is_loaded:
        _, aasist_scores = aasist_baseline.predict_batch(eval_audio_paths)
    else:
        aasist_scores = np.zeros(len(eval_audio_paths))

    # Layer-wise SI predictions
    print("    Layer-wise Speaker Invariant...")
    layerwise_scores = {}
    for layer_idx in range(NUM_LAYERS):
        _, scores = layerwise_detectors[layer_idx].predict_batch(eval_features_all_layers[:, layer_idx, :])
        layerwise_scores[layer_idx] = scores

    # Evaluate for each attack type
    print("\n  Computing EER for each attack type...")

    for attack_type in tqdm(attack_types, desc="Attack types"):
        # Get indices for this attack type
        attack_mask = eval_df['attack_type'] == attack_type
        attack_indices = np.where(attack_mask)[0]

        # Combine bonafide and this attack type
        combined_indices = np.concatenate([bonafide_indices, attack_indices])
        combined_labels = eval_labels[combined_indices]

        # WavLM Frozen
        result = evaluate_attack_type_eer(
            "WavLM Frozen",
            attack_type,
            combined_labels,
            wavlm_scores[combined_indices]
        )
        all_results.append(result)

        # Speaker Invariant
        result = evaluate_attack_type_eer(
            "Speaker Invariant (n=10)",
            attack_type,
            combined_labels,
            si_scores[combined_indices]
        )
        all_results.append(result)

        # RawNet2
        if rawnet2_baseline.is_loaded:
            result = evaluate_attack_type_eer(
                "RawNet2",
                attack_type,
                combined_labels,
                rawnet2_scores[combined_indices]
            )
            all_results.append(result)

        # AASIST
        if aasist_baseline.is_loaded:
            result = evaluate_attack_type_eer(
                "AASIST",
                attack_type,
                combined_labels,
                aasist_scores[combined_indices]
            )
            all_results.append(result)

        # Layer-wise SI
        for layer_idx in range(NUM_LAYERS):
            result = evaluate_attack_type_eer(
                f"SI Layer {layer_idx}",
                attack_type,
                combined_labels,
                layerwise_scores[layer_idx][combined_indices]
            )
            all_results.append(result)

    # =========================================================================
    # 6. Overall EER (all spoof types combined)
    # =========================================================================
    print("\n  Computing overall EER (all attacks combined)...")

    # WavLM Frozen
    result = evaluate_attack_type_eer("WavLM Frozen", "Overall", eval_labels, wavlm_scores)
    all_results.append(result)

    # Speaker Invariant
    result = evaluate_attack_type_eer("Speaker Invariant (n=10)", "Overall", eval_labels, si_scores)
    all_results.append(result)

    # RawNet2
    if rawnet2_baseline.is_loaded:
        result = evaluate_attack_type_eer("RawNet2", "Overall", eval_labels, rawnet2_scores)
        all_results.append(result)

    # AASIST
    if aasist_baseline.is_loaded:
        result = evaluate_attack_type_eer("AASIST", "Overall", eval_labels, aasist_scores)
        all_results.append(result)

    # Layer-wise SI
    for layer_idx in range(NUM_LAYERS):
        result = evaluate_attack_type_eer(f"SI Layer {layer_idx}", "Overall", eval_labels, layerwise_scores[layer_idx])
        all_results.append(result)

    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("[6] Saving results...")
    print("="*80)

    results_df = pd.DataFrame(all_results)

    # Save full results
    output_path = Config.RESULTS_DIR / "attack_type_eer_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"  Full results saved to {output_path}")

    # Create pivot table for easy viewing
    pivot_df = results_df.pivot(index='attack_type', columns='model', values='eer')
    pivot_path = Config.RESULTS_DIR / "attack_type_eer_pivot.csv"
    pivot_df.to_csv(pivot_path)
    print(f"  Pivot table saved to {pivot_path}")

    # =========================================================================
    # 8. Print Summary
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Main models summary
    main_models = ["AASIST", "RawNet2", "WavLM Frozen", "Speaker Invariant (n=10)"]
    print("\n[Main Models - Overall EER]")
    for model in main_models:
        overall = results_df[(results_df['model'] == model) & (results_df['attack_type'] == 'Overall')]
        if len(overall) > 0:
            print(f"  {model}: {overall['eer'].values[0]:.2f}%")

    # Best layer for SI
    print("\n[Layer-wise SI - Best Layer]")
    si_overall = results_df[(results_df['model'].str.startswith('SI Layer')) & (results_df['attack_type'] == 'Overall')]
    if len(si_overall) > 0:
        best_idx = si_overall['eer'].idxmin()
        best_row = si_overall.loc[best_idx]
        print(f"  Best Layer: {best_row['model']} with EER={best_row['eer']:.2f}%")

    # Per attack type summary (main models only)
    print("\n[Per Attack Type EER (Main Models)]")
    main_results = results_df[results_df['model'].isin(main_models)]
    main_pivot = main_results.pivot(index='attack_type', columns='model', values='eer')
    print(main_pivot.to_string())

    print("\n" + "="*80)
    print("[Done]")
    print("="*80)


if __name__ == "__main__":
    main()
