#!/usr/bin/env python3
"""
Cross-Dataset Deepfake Speech Detection (Argparse Version)

Usage:
    python cross_dataset_eval_argparse.py --train-data asvspoof2019
    python cross_dataset_eval_argparse.py --train-data asvspoof5
    python cross_dataset_eval_argparse.py --train-data both
    python cross_dataset_eval_argparse.py --train-data both --layerwise --attack-eval

    # Model selection examples:
    python cross_dataset_eval_argparse.py --models wavlm si              # Only WavLM and SI detectors
    python cross_dataset_eval_argparse.py --models rawnet2 aasist        # Only RawNet2 and AASIST
    python cross_dataset_eval_argparse.py --models si -n 10 16           # SI detectors with n=10,16

    # Custom weights:
    python cross_dataset_eval_argparse.py --models rawnet2 --rawnet2-weights /path/to/weights.pth
    python cross_dataset_eval_argparse.py --models aasist --aasist-weights /path/to/weights.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import librosa
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import torch.nn.functional as F

import yaml
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Deepfake Speech Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train-data', '-t', type=str, choices=['asvspoof2019', 'asvspoof5', 'both'],
                        default='asvspoof2019', help='Training data option')
    parser.add_argument('--n-components', '-n', type=int, nargs='+', default=[1, 5, 10, 16],
                        help='n_speaker_components list for SI detector')
    parser.add_argument('--layerwise', '-l', action='store_true', help='Enable layer-wise evaluation')
    parser.add_argument('--attack-eval', '-a', action='store_true', help='Enable attack type-wise evaluation')

    # Model selection arguments
    parser.add_argument('--models', '-m', type=str, nargs='+',
                        default=['wavlm', 'si', 'rawnet2', 'aasist'],
                        choices=['wavlm', 'si', 'rawnet2', 'aasist'],
                        help='Models to evaluate (wavlm: WavLM Baseline, si: Speaker-Invariant, rawnet2: RawNet2, aasist: AASIST)')
    parser.add_argument('--skip-rawnet2', action='store_true', help='Skip RawNet2 (deprecated, use --models)')
    parser.add_argument('--skip-aasist', action='store_true', help='Skip AASIST (deprecated, use --models)')
    parser.add_argument('--rawnet2-weights', type=str, default=None,
                        help='Custom path to RawNet2 weights (overrides auto-selection)')
    parser.add_argument('--aasist-weights', type=str, default=None,
                        help='Custom path to AASIST weights')

    parser.add_argument('--eval-datasets', type=str, nargs='+',
                        default=['asvspoof19', 'asvspoof21', 'asvspoof5', 'wavefake', 'inthewild'])
    parser.add_argument('--wavefake-samples', type=int, default=5000, help='Max WaveFake samples (0=all)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining')

    return parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

class Config:
    LA_ROOT = Path("/mnt/ddn/kyudan/Deepfake-speech/LA")
    LA_TRAIN_AUDIO = LA_ROOT / "ASVspoof2019_LA_train" / "flac"
    LA_DEV_AUDIO = LA_ROOT / "ASVspoof2019_LA_dev" / "flac"
    LA_EVAL_AUDIO = LA_ROOT / "ASVspoof2019_LA_eval" / "flac"
    LA_PROTOCOLS = LA_ROOT / "ASVspoof2019_LA_cm_protocols"

    ASVSPOOF5_ROOT = Path("/mnt/tmp/Deepfake-speech/data/ASVspoof5")
    ASVSPOOF5_TRAIN_AUDIO = ASVSPOOF5_ROOT / "flac_T"
    ASVSPOOF5_DEV_AUDIO = ASVSPOOF5_ROOT / "flac_D"
    ASVSPOOF5_EVAL_AUDIO = ASVSPOOF5_ROOT / "flac_E_eval"
    ASVSPOOF5_TRAIN_PROTOCOL = ASVSPOOF5_ROOT / "ASVspoof5.train.tsv"
    ASVSPOOF5_DEV_PROTOCOL = ASVSPOOF5_ROOT / "ASVspoof5.dev.track_1.tsv"
    ASVSPOOF5_EVAL_PROTOCOL = ASVSPOOF5_ROOT / "ASVspoof5.eval.track_1.tsv"

    DF_ROOT = Path("/mnt/tmp/Deepfake-speech/data/ASVspoof2021")
    WAVEFAKE_ROOT = Path("/mnt/tmp/Deepfake-speech/data/WaveFake")
    INTHEWILD_ROOT = Path("/mnt/tmp/Deepfake-speech/data/InTheWild_hf")

    MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
    RAWNET2_DIR = MODELS_DIR / "asvspoof2021_baseline" / "2021" / "DF" / "Baseline-RawNet2"
    AASIST_DIR = MODELS_DIR / "aasist"
    TRAINED_DIR = MODELS_DIR / "trained"
    RESULTS_DIR = Path("/mnt/tmp/Deepfake-speech/results")
    CACHE_DIR = Path("/mnt/tmp/Deepfake-speech/cache")


Config.TRAINED_DIR.mkdir(parents=True, exist_ok=True)
Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_asvspoof_protocol(protocol_path: str) -> pd.DataFrame:
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                data.append({'speaker_id': parts[0], 'file_id': parts[1],
                             'attack_type': parts[3], 'label': parts[4]})
    return pd.DataFrame(data)


def build_asvspoof_dataframe(protocol_path: str, audio_dir: str) -> pd.DataFrame:
    df = load_asvspoof_protocol(protocol_path)
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'bonafide' else 1)
    audio_dir = Path(audio_dir)
    df['audio_path'] = df['file_id'].apply(lambda x: str(audio_dir / f"{x}.flac"))
    df['exists'] = df['audio_path'].apply(lambda x: Path(x).exists())
    if (~df['exists']).sum() > 0:
        print(f"Warning: {(~df['exists']).sum()} files not found")
    return df[df['exists']].drop(columns=['exists']).copy()


def get_asvspoof19_datasets() -> Dict[str, pd.DataFrame]:
    datasets = {}
    print("Loading ASVspoof2019 LA...")
    datasets['train'] = build_asvspoof_dataframe(
        str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.train.trn.txt"), str(Config.LA_TRAIN_AUDIO))
    datasets['dev'] = build_asvspoof_dataframe(
        str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.dev.trl.txt"), str(Config.LA_DEV_AUDIO))
    datasets['eval'] = build_asvspoof_dataframe(
        str(Config.LA_PROTOCOLS / "ASVspoof2019.LA.cm.eval.trl.txt"), str(Config.LA_EVAL_AUDIO))
    return datasets


def load_asvspoof5_protocol(protocol_path: str) -> pd.DataFrame:
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                # Format: speaker_id file_id gender ? ? ? codec attack_type label ?
                data.append({'speaker_id': parts[0], 'file_id': parts[1], 'gender': parts[2],
                             'codec': parts[6], 'attack_type': parts[7], 'label': parts[8]})
    return pd.DataFrame(data)


def build_asvspoof5_dataframe(protocol_path: str, audio_dir: str) -> pd.DataFrame:
    df = load_asvspoof5_protocol(protocol_path)
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'bonafide' else 1)
    audio_dir = Path(audio_dir)
    df['audio_path'] = df['file_id'].apply(lambda x: str(audio_dir / f"{x}.flac"))
    df['exists'] = df['audio_path'].apply(lambda x: Path(x).exists())
    if (~df['exists']).sum() > 0:
        print(f"Warning: {(~df['exists']).sum()} files not found")
    return df[df['exists']].drop(columns=['exists']).copy()


def get_asvspoof5_datasets() -> Dict[str, pd.DataFrame]:
    datasets = {}
    if Config.ASVSPOOF5_TRAIN_PROTOCOL.exists() and Config.ASVSPOOF5_TRAIN_AUDIO.exists():
        print("Loading ASVspoof5 train...")
        datasets['train'] = build_asvspoof5_dataframe(
            str(Config.ASVSPOOF5_TRAIN_PROTOCOL), str(Config.ASVSPOOF5_TRAIN_AUDIO))
    if Config.ASVSPOOF5_DEV_PROTOCOL.exists() and Config.ASVSPOOF5_DEV_AUDIO.exists():
        print("Loading ASVspoof5 dev...")
        datasets['dev'] = build_asvspoof5_dataframe(
            str(Config.ASVSPOOF5_DEV_PROTOCOL), str(Config.ASVSPOOF5_DEV_AUDIO))
    if Config.ASVSPOOF5_EVAL_PROTOCOL.exists() and Config.ASVSPOOF5_EVAL_AUDIO.exists():
        print("Loading ASVspoof5 eval...")
        datasets['eval'] = build_asvspoof5_dataframe(
            str(Config.ASVSPOOF5_EVAL_PROTOCOL), str(Config.ASVSPOOF5_EVAL_AUDIO))
    return datasets


def get_asvspoof21_df_data() -> Optional[pd.DataFrame]:
    if not Config.DF_ROOT.exists():
        return None
    protocol_path = Config.DF_ROOT / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    if not protocol_path.exists():
        candidates = list(Config.DF_ROOT.glob("**/trial_metadata.txt"))
        protocol_path = candidates[0] if candidates else None
    if not protocol_path or not protocol_path.exists():
        return None

    print(f"Loading ASVspoof2021 DF...")
    audio_path_map = {}
    for part in ["ASVspoof2021_DF_eval_part00", "ASVspoof2021_DF_eval_part01",
                 "ASVspoof2021_DF_eval_part02", "ASVspoof2021_DF_eval_part03"]:
        part_dir = Config.DF_ROOT / part
        if part_dir.exists():
            for fp in tqdm(part_dir.rglob("*.flac"), desc=f"Scanning {part}", leave=False):
                audio_path_map[fp.stem] = str(fp)

    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[1] in audio_path_map:
                data.append({'file_id': parts[1], 'label': parts[5],
                             'binary_label': 0 if parts[5] == 'bonafide' else 1,
                             'audio_path': audio_path_map[parts[1]]})
    print(f"Loaded {len(data)} samples")
    return pd.DataFrame(data)


def get_wavefake_data(max_samples: int = 5000) -> Optional[pd.DataFrame]:
    if not Config.WAVEFAKE_ROOT.exists():
        return None
    data = []
    print("Loading WaveFake...")
    ljs_dir = Config.WAVEFAKE_ROOT / "LJSpeech-1.1" / "wavs"
    if ljs_dir.exists():
        for f in ljs_dir.glob("*.wav"):
            data.append({'file_id': f.stem, 'source': 'LJSpeech', 'binary_label': 0, 'audio_path': str(f)})
    jsut_dir = Config.WAVEFAKE_ROOT / "jsut_ver1.1" / "basic5000"
    if jsut_dir.exists():
        for f in jsut_dir.rglob("*.wav"):
            data.append({'file_id': f.stem, 'source': 'JSUT', 'binary_label': 0, 'audio_path': str(f)})
    gen_dir = Config.WAVEFAKE_ROOT / "generated_audio"
    if not gen_dir.exists():
        gen_dir = Config.WAVEFAKE_ROOT
    for algo_path in gen_dir.iterdir():
        if algo_path.is_dir() and ("ljspeech" in algo_path.name or "jsut" in algo_path.name):
            for f in algo_path.glob("*.wav"):
                data.append({'file_id': f.stem, 'source': algo_path.name, 'binary_label': 1, 'audio_path': str(f)})
    if not data:
        return None
    df = pd.DataFrame(data)
    if max_samples and max_samples > 0 and len(df) > max_samples:
        df_r = df[df['binary_label'] == 0].sample(n=min(len(df[df['binary_label']==0]), max_samples//2), random_state=42)
        df_f = df[df['binary_label'] == 1].sample(n=min(len(df[df['binary_label']==1]), max_samples//2), random_state=42)
        df = pd.concat([df_r, df_f]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"WaveFake: {len(df)} samples")
    return df


def get_inthewild_data() -> Optional[pd.DataFrame]:
    if not Config.INTHEWILD_ROOT.exists():
        return None
    release_dir = Config.INTHEWILD_ROOT / "release_in_the_wild"
    if not release_dir.exists():
        release_dir = Config.INTHEWILD_ROOT
    meta_path = release_dir / "meta.csv"
    if not meta_path.exists():
        return None
    print("Loading In-The-Wild...")
    meta_df = pd.read_csv(meta_path)
    data = []
    for _, row in meta_df.iterrows():
        audio_path = release_dir / row['file']
        if audio_path.exists():
            data.append({'file_id': Path(row['file']).stem, 'label': row['label'],
                         'binary_label': 0 if row['label'] == 'bona-fide' else 1, 'audio_path': str(audio_path)})
    print(f"In-The-Wild: {len(data)} samples")
    return pd.DataFrame(data) if data else None


# =============================================================================
# Metrics
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
    return eer * 100


def evaluate_detector(y_true, y_pred, y_scores) -> Dict:
    return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0), 'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'eer': compute_eer(y_true, y_scores)}


# =============================================================================
# Feature Extractors
# =============================================================================

class WavLMFeatureExtractor:
    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"WavLM loaded on {self.device}")

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(inputs.input_values.to(self.device))
            return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        except Exception as e:
            return None

    def extract_batch(self, audio_paths: List[str], cache_path: Optional[str] = None, desc: str = "Extracting") -> np.ndarray:
        if cache_path and Path(cache_path).exists():
            print(f"Loading cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        # features = [self.extract(p) or np.zeros(1024) for p in tqdm(audio_paths, desc=desc)]

        # 수정 전 (에러 발생):
        # features = [self.extract(p) or np.zeros(1024) for p in tqdm(audio_paths, desc=desc)]
        # features = np.array(features)
        # 수정 후: 명시적 None 체크
        features = []
        for p in tqdm(audio_paths, desc=desc):
            feat = self.extract(p)
            if feat is None:
                features.append(np.zeros(1024))
            else:
                features.append(feat)
                
        features = np.array(features)
        
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        return features


class WavLMLayerwiseExtractor:
    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM (layer-wise)...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.num_layers = 25
        self.hidden_size = 1024

    def extract_all_layers(self, audio_path: str) -> Optional[np.ndarray]:
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(inputs.input_values.to(self.device), output_hidden_states=True, return_dict=True)
            return torch.stack(outputs.hidden_states, dim=0).mean(dim=2).squeeze(1).cpu().numpy()
        except:
            return None

    def extract_batch_all_layers(self, audio_paths: List[str], cache_prefix: Optional[str] = None, desc: str = "Extracting") -> np.ndarray:
        if cache_prefix and all(Path(f"{cache_prefix}_layer_{i}.pkl").exists() for i in range(self.num_layers)):
            print(f"Loading cache: {cache_prefix}_layer_*.pkl")
            return np.stack([pickle.load(open(f"{cache_prefix}_layer_{i}.pkl", 'rb')) for i in range(self.num_layers)], axis=1)
        features = [self.extract_all_layers(p) or np.zeros((self.num_layers, self.hidden_size)) for p in tqdm(audio_paths, desc=desc)]
        features = np.array(features)
        if cache_prefix:
            Path(cache_prefix).parent.mkdir(parents=True, exist_ok=True)
            for i in range(self.num_layers):
                with open(f"{cache_prefix}_layer_{i}.pkl", 'wb') as f:
                    pickle.dump(features[:, i, :], f)
        return features


# =============================================================================
# Models
# =============================================================================

class WavLMFrozenBaseline:
    def __init__(self, wavlm_extractor=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wavlm = wavlm_extractor or WavLMFeatureExtractor(device=self.device)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_fitted = False

    def fit(self, audio_paths, labels, cache_path=None):
        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, np.array(labels))
        self.is_fitted = True
        return self.classifier.score(X_scaled, np.array(labels))

    def predict_batch(self, audio_paths, cache_path=None):
        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled), self.classifier.predict_proba(X_scaled)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'classifier': self.classifier, 'is_fitted': self.is_fitted}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.scaler, self.classifier, self.is_fitted = d['scaler'], d['classifier'], d['is_fitted']


class SpeakerInvariantDetector:
    def __init__(self, n_speaker_components=10, wavlm_extractor=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_speaker_components = n_speaker_components
        self.wavlm = wavlm_extractor or WavLMFeatureExtractor(device=self.device)
        self.scaler = StandardScaler()
        self.pca = None
        self.projection_matrix = None
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def fit(self, audio_paths, labels, speaker_ids, cache_path=None):
        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.fit_transform(X)
        spk_map = {}
        for idx, spk in enumerate(speaker_ids):
            spk_map.setdefault(spk, []).append(idx)
        centroids = np.array([np.mean(X_scaled[ids], axis=0) for ids in spk_map.values()])
        n_comp = min(self.n_speaker_components, len(centroids) - 1)
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(centroids)
        U = self.pca.components_.T
        self.projection_matrix = np.eye(U.shape[0]) - (U @ U.T)
        X_proj = X_scaled @ self.projection_matrix
        self.classifier.fit(X_proj, np.array(labels))
        self.is_fitted = True
        return self.classifier.score(X_proj, np.array(labels))

    def predict_batch(self, audio_paths, cache_path=None):
        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_proj = self.scaler.transform(X) @ self.projection_matrix
        return self.classifier.predict(X_proj), self.classifier.predict_proba(X_proj)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'n_speaker_components': self.n_speaker_components, 'scaler': self.scaler,
                         'pca': self.pca, 'projection_matrix': self.projection_matrix,
                         'classifier': self.classifier, 'is_fitted': self.is_fitted}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for k, v in d.items():
            setattr(self, k, v)


class LayerwiseSpeakerInvariantDetector:
    def __init__(self, layer_idx, n_speaker_components=10):
        self.layer_idx = layer_idx
        self.n_speaker_components = n_speaker_components
        self.scaler = StandardScaler()
        self.pca = None
        self.projection_matrix = None
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def fit(self, X_features, labels, speaker_ids):
        X_scaled = self.scaler.fit_transform(X_features)
        spk_map = {}
        for idx, spk in enumerate(speaker_ids):
            spk_map.setdefault(spk, []).append(idx)
        centroids = np.array([np.mean(X_scaled[ids], axis=0) for ids in spk_map.values()])
        n_comp = min(self.n_speaker_components, len(centroids) - 1)
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(centroids)
        U = self.pca.components_.T
        self.projection_matrix = np.eye(U.shape[0]) - (U @ U.T)
        X_proj = X_scaled @ self.projection_matrix
        self.classifier.fit(X_proj, np.array(labels))
        self.is_fitted = True
        return self.classifier.score(X_proj, np.array(labels))

    def predict_batch(self, X_features):
        X_proj = self.scaler.transform(X_features) @ self.projection_matrix
        return self.classifier.predict(X_proj), self.classifier.predict_proba(X_proj)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'layer_idx': self.layer_idx, 'n_speaker_components': self.n_speaker_components,
                         'scaler': self.scaler, 'pca': self.pca, 'projection_matrix': self.projection_matrix,
                         'classifier': self.classifier, 'is_fitted': self.is_fitted}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for k, v in d.items():
            setattr(self, k, v)


class RawNet2FrozenBaseline:
    def __init__(self, device=None, weights_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.code_dir = Config.MODELS_DIR / "asvspoof2021_baseline" / "2021" / "DF" / "Baseline-RawNet2"
        if weights_path:
            self.weights_path = Path(weights_path)
        else:
            self.weights_path = self.code_dir / "models" / "model_DF_CCE_100_32_0.0001" / "epoch_70.pth"
        self.nb_samp = 64000
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if str(self.code_dir) not in sys.path:
                sys.path.insert(0, str(self.code_dir))
            from model import RawNet
            cfg_path = self.code_dir / "model_config_RawNet2.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)['model']
            else:
                cfg = {'nb_samp': 64000, 'first_conv': 128, 'in_channels': 1,
                       'filts': [128, [128, 128], [128, 512], [512, 512]],
                       'blocks': [2, 4], 'nb_fc_node': 1024, 'gru_node': 1024, 'nb_gru_layer': 3, 'nb_classes': 2}
            self.model = RawNet(cfg, self.device).to(self.device)
            if not self.weights_path.exists():
                return
            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            except RuntimeError:
                cfg['first_conv'] = 20
                cfg['filts'] = [20, [20, 20], [20, 128], [128, 128]]
                self.model = RawNet(cfg, self.device).to(self.device)
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.is_loaded = True
            print("[RawNet2] Loaded")
        except Exception as e:
            print(f"[RawNet2] Error: {e}")

    def predict_batch(self, audio_paths):
        if not self.is_loaded:
            return np.zeros(len(audio_paths)), np.zeros(len(audio_paths))
        preds, scores = [], []
        for path in tqdm(audio_paths, desc="RawNet2"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) < self.nb_samp:
                    audio = np.tile(audio, int(self.nb_samp / len(audio)) + 1)[:self.nb_samp]
                else:
                    audio = audio[:self.nb_samp]
                with torch.no_grad():
                    out = self.model(torch.FloatTensor(audio).unsqueeze(0).to(self.device))
                    probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
                preds.append(1 if probs[0] > 0.5 else 0)
                scores.append(float(probs[0]))
            except:
                preds.append(0)
                scores.append(0.0)
        return np.array(preds), np.array(scores)


class AASISTFrozenBaseline:
    def __init__(self, device=None, weights_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Config.AASIST_DIR
        self.custom_weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self.is_loaded = False
        self.nb_samp = 64600
        if self.model_dir.exists() or self.custom_weights_path:
            self._load_model()

    def _load_model(self):
        try:
            models_dir = str(self.model_dir / "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            from AASIST import Model as AASIST
            cfg_path = self.model_dir / "config" / "AASIST.conf"
            cfg = json.load(open(cfg_path)).get('model_config', {}) if cfg_path.exists() else {}
            self.nb_samp = cfg.get('nb_samp', 64600)
            # Use custom weights path if provided
            if self.custom_weights_path and self.custom_weights_path.exists():
                weights_path = self.custom_weights_path
            else:
                weights_path = self.model_dir / "models" / "weights" / "AASIST.pth"
            if not weights_path.exists():
                return
            self.model = AASIST(cfg).to(self.device)
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.is_loaded = True
            print(f"[AASIST] Loaded from {weights_path}")
        except Exception as e:
            print(f"[AASIST] Error: {e}")

    def predict_batch(self, audio_paths):
        if not self.is_loaded:
            return np.zeros(len(audio_paths)), np.zeros(len(audio_paths))
        preds, scores = [], []
        for path in tqdm(audio_paths, desc="AASIST"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) < self.nb_samp:
                    audio = np.tile(audio, int(self.nb_samp / len(audio)) + 1)[:self.nb_samp]
                else:
                    audio = audio[:self.nb_samp]
                with torch.no_grad():
                    _, out = self.model(torch.FloatTensor(audio).unsqueeze(0).to(self.device))
                    probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
                preds.append(1 if probs[0] > 0.5 else 0)
                scores.append(float(probs[0]))
            except:
                preds.append(0)
                scores.append(0.0)
        return np.array(preds), np.array(scores)


# =============================================================================
# Attack Type-wise Evaluation
# =============================================================================

def evaluate_attack_type_eer(eval_df, model_predictions, dataset_name):
    results = []
    if 'attack_type' not in eval_df.columns:
        return pd.DataFrame()
    bonafide_mask = (eval_df['attack_type'] == '-') | (eval_df['attack_type'] == 'bonafide')
    bonafide_idx = np.where(bonafide_mask)[0]
    bonafide_labels = eval_df.loc[bonafide_mask, 'binary_label'].values
    attack_types = [at for at in eval_df['attack_type'].unique() if at not in ['-', 'bonafide']]

    for at in sorted(attack_types):
        at_idx = np.where(eval_df['attack_type'] == at)[0]
        at_labels = eval_df.loc[eval_df['attack_type'] == at, 'binary_label'].values
        comb_idx = np.concatenate([bonafide_idx, at_idx])
        comb_labels = np.concatenate([bonafide_labels, at_labels])
        for model_name, (_, scores) in model_predictions.items():
            eer = compute_eer(comb_labels, scores[comb_idx]) if len(np.unique(comb_labels)) >= 2 else float('nan')
            results.append({'dataset': dataset_name, 'model': model_name, 'attack_type': at,
                            'n_bonafide': len(bonafide_idx), 'n_spoof': len(at_idx), 'eer': eer})

    all_labels = eval_df['binary_label'].values
    for model_name, (_, scores) in model_predictions.items():
        results.append({'dataset': dataset_name, 'model': model_name, 'attack_type': 'Overall',
                        'n_bonafide': (all_labels == 0).sum(), 'n_spoof': (all_labels == 1).sum(),
                        'eer': compute_eer(all_labels, scores)})
    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("="*80)
    print("Cross-Dataset Deepfake Speech Detection")
    print("="*80)
    print(f"  --train-data: {args.train_data}")
    print(f"  --models: {args.models}")
    print(f"  --n-components: {args.n_components}")
    print(f"  --layerwise: {args.layerwise}")
    print(f"  --attack-eval: {args.attack_eval}")
    print("="*80)

    # Handle deprecated skip flags
    if args.skip_rawnet2 and 'rawnet2' in args.models:
        args.models.remove('rawnet2')
    if args.skip_aasist and 'aasist' in args.models:
        args.models.remove('aasist')

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 1. Load Training Data
    print("\n" + "="*80)
    print("[1] Loading Training Data")
    print("="*80)

    asvspoof19_data = get_asvspoof19_datasets()

    if args.train_data == "asvspoof2019":
        train_df = asvspoof19_data['train']
    elif args.train_data == "asvspoof5":
        asvspoof5_data = get_asvspoof5_datasets()
        train_df = asvspoof5_data['train']
    else:
        asvspoof5_data = get_asvspoof5_datasets()
        train_df = pd.concat([asvspoof19_data['train'], asvspoof5_data['train']], ignore_index=True)

    train_paths = train_df['audio_path'].tolist()
    train_labels = train_df['binary_label'].tolist()
    train_spk_ids = train_df['speaker_id'].tolist()
    print(f"Train: {len(train_paths)} samples")

    # 2. Train Models
    print("\n" + "="*80)
    print("[2] Training Models")
    print("="*80)

    # Initialize WavLM extractor only if needed
    wavlm_extractor = None
    train_cache = None
    if 'wavlm' in args.models or 'si' in args.models or args.layerwise:
        wavlm_extractor = WavLMFeatureExtractor(device=device)
        train_cache = str(Config.CACHE_DIR / f"train_{args.train_data}_wavlm_features.pkl")

    # Train WavLM Baseline
    wavlm_baseline = None
    if 'wavlm' in args.models:
        wavlm_baseline = WavLMFrozenBaseline(wavlm_extractor=wavlm_extractor, device=device)
        wavlm_path = Config.TRAINED_DIR / f"wavlm_frozen_baseline_{args.train_data}.pkl"
        if wavlm_path.exists() and not args.force_retrain:
            wavlm_baseline.load(str(wavlm_path))
            print(f"Loaded: {wavlm_path}")
        else:
            acc = wavlm_baseline.fit(train_paths, train_labels, cache_path=train_cache)
            wavlm_baseline.save(str(wavlm_path))
            print(f"Trained WavLM Baseline: {acc:.4f}")

    # Train Speaker-Invariant Detectors
    si_detectors = {}
    if 'si' in args.models:
        for n in args.n_components:
            det = SpeakerInvariantDetector(n_speaker_components=n, wavlm_extractor=wavlm_extractor, device=device)
            si_path = Config.TRAINED_DIR / f"si_detector_n{n}_{args.train_data}.pkl"
            if si_path.exists() and not args.force_retrain:
                det.load(str(si_path))
                print(f"Loaded: {si_path}")
            else:
                acc = det.fit(train_paths, train_labels, train_spk_ids, cache_path=train_cache)
                det.save(str(si_path))
                print(f"Trained SI (n={n}): {acc:.4f}")
            si_detectors[n] = det

    # Load RawNet2
    rawnet2 = None
    if 'rawnet2' in args.models:
        # Custom weights path or auto-select based on training data
        if args.rawnet2_weights:
            rawnet2_weights = args.rawnet2_weights
        elif args.train_data in ["asvspoof5", "both"]:
            rawnet2_weights = str(Config.MODELS_DIR / "rawnet2_asvspoof5" / "models" / "rawnet2_asvspoof5_weighted_CCE_100_32_0.0001" / "best_eer_model.pth")
        else:
            rawnet2_weights = None
        rawnet2 = RawNet2FrozenBaseline(device=device, weights_path=rawnet2_weights)

    # Load AASIST
    aasist = None
    if 'aasist' in args.models:
        aasist = AASISTFrozenBaseline(device=device, weights_path=args.aasist_weights)

    # 3. Load Eval Datasets
    print("\n" + "="*80)
    print("[3] Loading Evaluation Datasets")
    print("="*80)

    datasets_to_eval = {}
    if 'asvspoof19' in args.eval_datasets:
        datasets_to_eval['ASVspoof19 LA'] = asvspoof19_data['eval']
    if 'asvspoof21' in args.eval_datasets:
        d = get_asvspoof21_df_data()
        if d is not None:
            datasets_to_eval['ASVspoof21 DF'] = d
    if 'asvspoof5' in args.eval_datasets:
        d = get_asvspoof5_datasets()
        if 'eval' in d:
            datasets_to_eval['ASVspoof5 Eval'] = d['eval']
    if 'wavefake' in args.eval_datasets:
        d = get_wavefake_data(max_samples=args.wavefake_samples)
        if d is not None:
            datasets_to_eval['WaveFake'] = d
    if 'inthewild' in args.eval_datasets:
        d = get_inthewild_data()
        if d is not None:
            datasets_to_eval['In-The-Wild'] = d

    print(f"Eval datasets: {list(datasets_to_eval.keys())}")

    # 4. Cross-Dataset Evaluation
    print("\n" + "="*80)
    print("[4] Cross-Dataset Evaluation")
    print("="*80)

    eval_models = {}
    if wavlm_baseline and wavlm_baseline.is_fitted:
        eval_models['WavLM Baseline'] = wavlm_baseline
    for n, det in si_detectors.items():
        if det and det.is_fitted:
            eval_models[f'SI-Detector (n={n})'] = det
    if rawnet2 and rawnet2.is_loaded:
        eval_models['RawNet2'] = rawnet2
    if aasist and aasist.is_loaded:
        eval_models['AASIST'] = aasist

    print(f"Eval models: {list(eval_models.keys())}")

    cross_results = []
    for ds_name, df in datasets_to_eval.items():
        print(f"\n  === {ds_name} ({len(df)} samples) ===")
        paths = df['audio_path'].tolist()
        labels = np.array(df['binary_label'].tolist())
        safe_name = ds_name.lower().replace(" ", "_").replace("-", "_")
        cache = str(Config.CACHE_DIR / f"{safe_name}_wavlm_features.pkl")

        for model_name, model in eval_models.items():
            print(f"    {model_name}...", end=" ", flush=True)
            try:
                if 'cache_path' in model.predict_batch.__code__.co_varnames:
                    preds, scores = model.predict_batch(paths, cache_path=cache)
                else:
                    print()
                    preds, scores = model.predict_batch(paths)
                m = evaluate_detector(labels, preds, scores)
                m['model'], m['dataset'] = model_name, ds_name
                cross_results.append(m)
                print(f"EER: {m['eer']:.2f}%")
            except Exception as e:
                print(f"Error: {e}")

    cross_df = pd.DataFrame(cross_results)
    cross_path = Config.RESULTS_DIR / f"cross_dataset_results_{args.train_data}.csv"
    cross_df.to_csv(cross_path, index=False)
    print(f"\nSaved: {cross_path}")

    # 5. Attack Type-wise (Optional)
    if args.attack_eval:
        print("\n" + "="*80)
        print("[5] Attack Type-wise Evaluation")
        print("="*80)

        for ds_name, df in datasets_to_eval.items():
            if 'attack_type' not in df.columns:
                continue
            print(f"\n  === {ds_name} ===")
            paths = df['audio_path'].tolist()
            safe_name = ds_name.lower().replace(" ", "_").replace("-", "_")
            cache = str(Config.CACHE_DIR / f"{safe_name}_wavlm_features.pkl")

            predictions = {}
            for model_name, model in eval_models.items():
                try:
                    if 'cache_path' in model.predict_batch.__code__.co_varnames:
                        _, scores = model.predict_batch(paths, cache_path=cache)
                    else:
                        _, scores = model.predict_batch(paths)
                    predictions[model_name] = (None, scores)
                except:
                    pass

            at_df = evaluate_attack_type_eer(df, predictions, ds_name)
            if not at_df.empty:
                pivot = at_df.pivot(index='attack_type', columns='model', values='eer')
                print(pivot.round(2).to_string())
                at_path = Config.RESULTS_DIR / f"attack_type_eer_{safe_name}_{args.train_data}.csv"
                at_df.to_csv(at_path, index=False)
                print(f"  Saved: {at_path}")

    # 6. Layer-wise (Optional)
    if args.layerwise:
        print("\n" + "="*80)
        print("[6] Layer-wise SI Evaluation")
        print("="*80)

        lw_extractor = WavLMLayerwiseExtractor(device=device)
        NUM_LAYERS, N_COMP = 25, 10

        train_lw_cache = str(Config.CACHE_DIR / f"train_{args.train_data}")
        train_feats = lw_extractor.extract_batch_all_layers(train_paths, cache_prefix=train_lw_cache)

        lw_detectors = {}
        for li in tqdm(range(NUM_LAYERS), desc="Training layer detectors"):
            det_path = Config.TRAINED_DIR / f"si_detector_layer_{li}_{args.train_data}.pkl"
            det = LayerwiseSpeakerInvariantDetector(layer_idx=li, n_speaker_components=N_COMP)
            if det_path.exists() and not args.force_retrain:
                det.load(str(det_path))
            else:
                det.fit(train_feats[:, li, :], train_labels, train_spk_ids)
                det.save(str(det_path))
            lw_detectors[li] = det

        lw_results = []
        for ds_name, df in datasets_to_eval.items():
            print(f"\n  === {ds_name} ===")
            paths = df['audio_path'].tolist()
            labels = np.array(df['binary_label'].tolist())
            safe_name = ds_name.lower().replace(" ", "_").replace("-", "_")
            eval_cache = str(Config.CACHE_DIR / safe_name)
            eval_feats = lw_extractor.extract_batch_all_layers(paths, cache_prefix=eval_cache)

            for li in range(NUM_LAYERS):
                _, scores = lw_detectors[li].predict_batch(eval_feats[:, li, :])
                lw_results.append({'dataset': ds_name, 'layer': li, 'eer': compute_eer(labels, scores)})

            ds_res = [r for r in lw_results if r['dataset'] == ds_name]
            best = min(ds_res, key=lambda x: x['eer'])
            print(f"    Best: Layer {best['layer']} (EER: {best['eer']:.2f}%)")

        lw_df = pd.DataFrame(lw_results)
        lw_path = Config.RESULTS_DIR / f"layerwise_si_results_{args.train_data}.csv"
        lw_df.to_csv(lw_path, index=False)
        print(f"\nSaved: {lw_path}")

        fig, ax = plt.subplots(figsize=(14, 6))
        for ds_name in datasets_to_eval.keys():
            d = lw_df[lw_df['dataset'] == ds_name]
            ax.plot(d['layer'], d['eer'], marker='o', label=ds_name)
        ax.set_xlabel('WavLM Layer')
        ax.set_ylabel('EER (%)')
        ax.set_title(f'Layer-wise SI (Trained on {args.train_data})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(range(NUM_LAYERS))
        plt.tight_layout()
        plot_path = Config.RESULTS_DIR / f"layerwise_si_eer_plot_{args.train_data}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved: {plot_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if not cross_df.empty:
        pivot = cross_df.pivot(index='model', columns='dataset', values='eer')
        print(pivot.round(2).to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
