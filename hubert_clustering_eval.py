"""
HuBERT-based Deepfake Detection with Clustering Metrics

This script evaluates HuBERT features for deepfake detection using:
1. Last layer embeddings (without speaker invariant)
2. Last layer embeddings with speaker invariant processing

Metrics computed:
- EER (Equal Error Rate) on ASV19 LA eval
- Silhouette Score (clustering quality - higher is better)
- Davies-Bouldin Index (clustering quality - lower is better)

Usage:
    python hubert_clustering_eval.py \
        --train_root /path/to/LA \
        --eval_root /path/to/LA \
        --mode both \
        --n_speaker_components 10 \
        --output_dir results/hubert_clustering
"""

import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, roc_curve,
    silhouette_score, silhouette_samples, davies_bouldin_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import (
    HubertModel, Wav2Vec2Model, WavLMModel,
    Wav2Vec2FeatureExtractor, AutoFeatureExtractor
)
import pickle
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib


# =============================================================================
# Audio Loading Worker (for multiprocessing)
# =============================================================================
def _load_audio_worker(audio_path, sr=16000):
    """Load a single audio file (multiprocessing worker)"""
    try:
        audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        return audio_path, audio, None
    except Exception as e:
        return audio_path, None, str(e)


# =============================================================================
# Supported SSL Models
# =============================================================================
SSL_MODELS = {
    # HuBERT models
    'hubert-large': 'facebook/hubert-large-ls960-ft',
    'hubert-base': 'facebook/hubert-base-ls960',
    # Wav2Vec2 models
    'wav2vec2-large': 'facebook/wav2vec2-large-960h',
    'wav2vec2-base': 'facebook/wav2vec2-base-960h',
    # WavLM models
    'wavlm-large': 'microsoft/wavlm-large',
    'wavlm-base': 'microsoft/wavlm-base',
    'wavlm-base-plus': 'microsoft/wavlm-base-plus',
    # XLS-R models
    'xls-r-300m': 'facebook/wav2vec2-xls-r-300m',
    'xls-r-1b': 'facebook/wav2vec2-xls-r-1b',
    'xls-r-2b': 'facebook/wav2vec2-xls-r-2b',
    # XLSR-Wav2Vec2
    'xlsr-53': 'facebook/wav2vec2-large-xlsr-53',
}


def get_model_class(model_name):
    """Return appropriate model class based on model name"""
    model_name_lower = model_name.lower()
    if 'hubert' in model_name_lower:
        return HubertModel
    elif 'wavlm' in model_name_lower:
        return WavLMModel
    else:  # wav2vec2, xls-r, xlsr
        return Wav2Vec2Model


# =============================================================================
# SSL Feature Extractor (supports multiple models)
# =============================================================================
class SSLFeatureExtractor:
    """SSL Model Feature Extractor with caching support (HuBERT, Wav2Vec2, WavLM, XLS-R)"""

    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device=None,
                 cache_dir=None, batch_size=32, num_workers=8, layers=None):
        """
        Args:
            model_name: HuggingFace model name
            device: cuda or cpu
            cache_dir: Directory for caching features
            batch_size: Batch size for GPU inference
            num_workers: Number of workers for audio loading
            layers: List of layer indices to extract (None = last layer only)
                    e.g., [8, 22] for WavLM speaker layers
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.layers = layers  # None means use last_hidden_state

        # Create a safe cache name from model name
        cache_name = model_name.replace('/', '_').replace('-', '_')
        if layers:
            layer_str = "_".join(map(str, layers))
            cache_name = f"{cache_name}_layers_{layer_str}"
        else:
            cache_name = f"{cache_name}_last_hidden_state"

        # Setup cache directory
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_subdir = self.cache_dir / cache_name
            self.cache_subdir.mkdir(parents=True, exist_ok=True)
            print(f"Feature caching enabled: {self.cache_subdir}")
        else:
            self.cache_subdir = None

        # Load SSL model
        model_class = get_model_class(model_name)
        print(f"Loading SSL model: {model_name} ({model_class.__name__}) on {self.device}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        # Feature dim depends on whether we concat multiple layers
        if layers:
            self.feature_dim = self.hidden_size * len(layers)
            print(f"Using layers {layers}, Feature dimension: {self.feature_dim} ({self.hidden_size} x {len(layers)})")
        else:
            self.feature_dim = self.hidden_size
            print(f"Using last layer, Feature dimension: {self.feature_dim}")

        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_path(self, audio_path):
        """Get cache file path using MD5 hash"""
        if not self.cache_subdir:
            return None
        abs_path = str(Path(audio_path).resolve())
        path_hash = hashlib.md5(abs_path.encode()).hexdigest()
        subdir1 = path_hash[:2]
        subdir2 = path_hash[2:4]
        return self.cache_subdir / subdir1 / subdir2 / f"{path_hash}.npy"

    def _load_from_cache(self, audio_path):
        """Load feature from cache if exists"""
        cache_path = self._get_cache_path(audio_path)
        if cache_path and cache_path.exists():
            try:
                feature = np.load(cache_path)
                self.cache_hits += 1
                return feature
            except Exception:
                return None
        return None

    def _save_to_cache(self, audio_path, feature):
        """Save feature to cache"""
        cache_path = self._get_cache_path(audio_path)
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, feature)
            except Exception:
                pass

    def extract(self, audio_path, use_cache=True):
        """Extract SSL feature (mean over time)"""
        # Check cache first
        if use_cache and self.cache_subdir:
            cached = self._load_from_cache(audio_path)
            if cached is not None:
                return cached
            self.cache_misses += 1

        try:
            audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                if self.layers:
                    # Extract specific layers
                    outputs = self.model(input_values, output_hidden_states=True)
                    # hidden_states: tuple of (batch, seq_len, hidden_size) for each layer
                    # Layer 0 is embedding, layers 1-N are transformer layers
                    layer_features = []
                    for layer_idx in self.layers:
                        # +1 because index 0 is the embedding layer
                        layer_out = outputs.hidden_states[layer_idx + 1]
                        layer_features.append(layer_out.mean(dim=1))  # Mean over time
                    features = torch.cat(layer_features, dim=-1).squeeze(0).cpu().numpy()
                else:
                    # Use last hidden state
                    outputs = self.model(input_values)
                    features = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

            if use_cache and self.cache_subdir:
                self._save_to_cache(audio_path, features)

            return features
        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def extract_batch(self, audio_paths, use_cache=True, desc="Extracting"):
        """Batch extraction with multiprocessing and GPU batching"""
        all_features = {}
        paths_to_load = []

        # Phase 1: Check cache
        print(f"  [1/3] Checking cache...")
        for path in tqdm(audio_paths, desc="  Cache check", leave=False):
            if use_cache and self.cache_subdir:
                cached = self._load_from_cache(path)
                if cached is not None:
                    all_features[path] = cached
                    continue
            paths_to_load.append(path)
            self.cache_misses += 1

        if self.cache_subdir:
            print(f"  Cache: {len(all_features)} hits, {len(paths_to_load)} to extract")

        if not paths_to_load:
            return np.array([all_features[p] for p in audio_paths]), list(range(len(audio_paths)))

        # Phase 2: Load audio with multiprocessing
        print(f"  [2/3] Loading audio files...")
        audio_data = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_load_audio_worker, p): p for p in paths_to_load}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {desc} (loading)", leave=False):
                path, audio, error = future.result()
                if audio is not None:
                    audio_data[path] = audio
                else:
                    print(f"  Failed to load {path}: {error}")

        # Phase 3: GPU batch processing
        print(f"  [3/3] GPU batch inference...")
        paths_with_audio = list(audio_data.keys())
        for batch_start in tqdm(range(0, len(paths_with_audio), self.batch_size),
                                desc=f"  {desc} (GPU batch)", leave=False):
            batch_paths = paths_with_audio[batch_start:batch_start + self.batch_size]
            batch_audios = [audio_data[p] for p in batch_paths]

            # Create batch input
            inputs = self.feature_extractor(
                batch_audios, sampling_rate=16000, return_tensors="pt",
                padding=True, truncation=True, max_length=16000*30
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                if self.layers:
                    # Extract specific layers
                    outputs = self.model(input_values, output_hidden_states=True)
                    layer_features = []
                    for layer_idx in self.layers:
                        layer_out = outputs.hidden_states[layer_idx + 1]
                        layer_features.append(layer_out.mean(dim=1))
                    batch_features = torch.cat(layer_features, dim=-1).cpu().numpy()
                else:
                    outputs = self.model(input_values)
                    batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            for i, path in enumerate(batch_paths):
                feat = batch_features[i]
                all_features[path] = feat
                if use_cache:
                    self._save_to_cache(path, feat)

        # Return in original order
        result = []
        valid_indices = []
        for idx, path in enumerate(audio_paths):
            if path in all_features:
                result.append(all_features[path])
                valid_indices.append(idx)
            else:
                result.append(None)

        # Filter out None values
        result = [r for r in result if r is not None]
        return np.array(result), valid_indices


# =============================================================================
# SSL Baseline Detector (Mode 1: No Speaker Invariant)
# =============================================================================
class SSLBaselineDetector:
    """SSL Model Last Layer + Logistic Regression (No Speaker Invariant)"""

    def __init__(self, ssl_extractor=None, model_name="facebook/hubert-large-ls960-ft",
                 device=None, cache_dir=None, batch_size=32, num_workers=8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if ssl_extractor:
            self.ssl = ssl_extractor
        else:
            self.ssl = SSLFeatureExtractor(
                model_name=model_name,
                device=self.device,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers
            )

        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False
        self._train_features = None

    def fit(self, audio_paths, labels, speaker_ids=None):
        """Train on audio files"""
        print("Extracting features for training...")
        self.ssl.cache_hits = 0
        self.ssl.cache_misses = 0

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Training")
        y = np.array(labels)[valid_indices]

        print(f"  Successfully extracted: {len(X)} / {len(audio_paths)} samples")

        # Scale and train
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        # Store training features for clustering analysis
        self._train_features = X_scaled
        self._train_labels = y
        if speaker_ids is not None:
            self._train_speakers = np.array(speaker_ids)[valid_indices]
        else:
            self._train_speakers = None

        acc = self.classifier.score(X_scaled, y)
        print(f"Training complete. Accuracy: {acc:.4f}")

        return acc

    def get_features(self, audio_paths):
        """Get scaled features for clustering analysis"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Features")
        X_scaled = self.scaler.transform(X)
        return X_scaled, valid_indices

    def predict_batch(self, audio_paths):
        """Batch prediction returning predictions and probabilities"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        print("Running batch prediction...")
        self.ssl.cache_hits = 0
        self.ssl.cache_misses = 0

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Predicting")
        X_scaled = self.scaler.transform(X)

        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities, valid_indices


# =============================================================================
# SSL Speaker Invariant Detector (Mode 2)
# =============================================================================
# WavLM speaker-related layers (from WavLM paper)
WAVLM_SPEAKER_LAYERS = [8, 22]


class SSLSpeakerInvariantDetector:
    """SSL Model Last Layer + Speaker Invariant Processing + Logistic Regression

    For WavLM: Uses layers 8 and 22 which are known to contain speaker information.
    For other models: Uses last hidden state.
    """

    def __init__(self, n_speaker_components=10, ssl_extractor=None,
                 model_name="facebook/hubert-large-ls960-ft",
                 device=None, cache_dir=None, batch_size=32, num_workers=8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_speaker_components = n_speaker_components

        # Get model name from extractor if provided
        if ssl_extractor:
            model_name = ssl_extractor.model_name

        # Check if this is a WavLM model
        self.is_wavlm = 'wavlm' in model_name.lower()

        if self.is_wavlm:
            # For WavLM: create new extractor with speaker-related layers [8, 22]
            print(f"WavLM detected: Using speaker-related layers {WAVLM_SPEAKER_LAYERS}")
            self.ssl = SSLFeatureExtractor(
                model_name=model_name,
                device=self.device,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                layers=WAVLM_SPEAKER_LAYERS
            )
        elif ssl_extractor:
            self.ssl = ssl_extractor
        else:
            self.ssl = SSLFeatureExtractor(
                model_name=model_name,
                device=self.device,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers
            )

        self.scaler = StandardScaler()
        self.pca = None
        self.projection_matrix = None
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def fit(self, audio_paths, labels, speaker_ids):
        """
        Train with speaker invariant processing:
        1. Extract features
        2. StandardScaler normalization
        3. Compute speaker centroids
        4. PCA on centroids to find speaker subspace
        5. Create orthogonal projection matrix: P_perp = I - U @ U.T
        6. Project features and train classifier
        """
        print(f"Extracting features for training... ({len(audio_paths)} samples)")
        self.ssl.cache_hits = 0
        self.ssl.cache_misses = 0

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Training")
        labels = np.array(labels)
        speaker_ids = np.array(speaker_ids)
        y = labels[valid_indices]
        valid_speakers = speaker_ids[valid_indices]

        print(f"  Successfully extracted: {len(X)} / {len(audio_paths)} samples")

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Build speaker map
        spk_map = {}
        for idx, spk in enumerate(valid_speakers):
            spk_map.setdefault(spk, []).append(idx)

        # Compute speaker centroids
        print(f"Computing speaker subspace (removing top {self.n_speaker_components} components)...")
        centroids = np.array([np.mean(X_scaled[ids], axis=0) for ids in spk_map.values()])
        print(f"  Number of speakers: {len(centroids)}")

        # PCA on centroids
        n_comp = min(self.n_speaker_components, len(centroids) - 1, X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(centroids)

        # Create orthogonal projection matrix: P_perp = I - U @ U.T
        U = self.pca.components_.T  # (feature_dim, n_components)
        self.projection_matrix = np.eye(U.shape[0]) - (U @ U.T)

        # Project and train
        print("Projecting features and training classifier...")
        X_proj = X_scaled @ self.projection_matrix
        self.classifier.fit(X_proj, y)
        self.is_fitted = True

        # Store training features for clustering analysis
        self._train_features = X_proj
        self._train_labels = y
        self._train_speakers = valid_speakers

        acc = self.classifier.score(X_proj, y)
        print(f"Training complete. Accuracy: {acc:.4f}")

        return acc

    def get_features(self, audio_paths):
        """Get projected features for clustering analysis"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Features")
        X_scaled = self.scaler.transform(X)
        X_proj = X_scaled @ self.projection_matrix
        return X_proj, valid_indices

    def predict_batch(self, audio_paths):
        """Batch prediction"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        print("Running batch prediction...")
        self.ssl.cache_hits = 0
        self.ssl.cache_misses = 0

        X, valid_indices = self.ssl.extract_batch(audio_paths, desc="Predicting")
        X_proj = self.scaler.transform(X) @ self.projection_matrix

        predictions = self.classifier.predict(X_proj)
        probabilities = self.classifier.predict_proba(X_proj)[:, 1]

        return predictions, probabilities, valid_indices


# =============================================================================
# Metrics Functions
# =============================================================================
def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) in percentage"""
    if len(np.unique(y_true)) < 2:
        return float('nan')

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100  # Return as percentage


def compute_clustering_metrics(features: np.ndarray, labels: np.ndarray, speaker_ids: np.ndarray) -> dict:
    """
    Compute clustering quality metrics in two ways:
    1. Speaker-wise: How well samples cluster by speaker identity
    2. Nat/Syn: How well samples cluster by bonafide vs spoof

    Uses cosine distance on full-dimensional features (no PCA reduction).

    Args:
        features: (N, D) feature matrix
        labels: (N,) binary labels (0=bonafide, 1=spoof)
        speaker_ids: (N,) speaker IDs

    Returns:
        dict with silhouette/davies_bouldin for both nat_syn and speaker clustering
    """
    # Convert speaker_ids to numeric labels
    unique_speakers = np.unique(speaker_ids)
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
    speaker_labels = np.array([speaker_to_idx[spk] for spk in speaker_ids])

    # Nat/Syn clustering metrics (cosine distance, full-dimensional)
    silhouette_nat_syn = silhouette_score(features, labels, metric='cosine')
    davies_bouldin_nat_syn = davies_bouldin_score(features, labels)

    # Speaker-wise clustering metrics
    silhouette_speaker = silhouette_score(features, speaker_labels, metric='cosine')
    davies_bouldin_speaker = davies_bouldin_score(features, speaker_labels)

    return {
        # Nat/Syn metrics (higher silhouette = better, lower DB = better)
        'silhouette_nat_syn': silhouette_nat_syn,
        'davies_bouldin_nat_syn': davies_bouldin_nat_syn,
        # Speaker metrics
        'silhouette_speaker': silhouette_speaker,
        'davies_bouldin_speaker': davies_bouldin_speaker,
    }


# =============================================================================
# Visualization
# =============================================================================
def plot_clustering_visualization(features, labels, title, save_path, max_samples=5000):
    """
    Create 2D scatter plot using t-SNE

    Args:
        features: High-dimensional feature matrix (N, D)
        labels: Binary labels (0=bonafide, 1=spoof)
        title: Plot title
        save_path: Path to save the figure
        max_samples: Maximum samples for t-SNE (for speed)
    """
    # Subsample if too many samples (t-SNE is slow)
    n_samples = len(features)
    if n_samples > max_samples:
        print(f"  Subsampling {max_samples}/{n_samples} for t-SNE visualization...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Apply t-SNE
    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    # Separate by class
    bonafide_mask = labels == 0
    spoof_mask = labels == 1

    plt.scatter(features_2d[bonafide_mask, 0], features_2d[bonafide_mask, 1],
                c='green', label='Bonafide', alpha=0.5, s=20)
    plt.scatter(features_2d[spoof_mask, 0], features_2d[spoof_mask, 1],
                c='red', label='Spoof', alpha=0.5, s=20)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {save_path}")


# =============================================================================
# Dataset Loading
# =============================================================================
def prepare_asvspoof_dataset(asvspoof_root, subset='eval', max_samples=None):
    """
    Load ASVspoof2019 LA dataset

    Args:
        asvspoof_root: ASVspoof LA root directory
        subset: 'train', 'dev', or 'eval'
        max_samples: Maximum number of samples (None for all)

    Returns:
        (audio_paths, labels, speaker_ids, system_ids) tuple
    """
    asvspoof_root = Path(asvspoof_root)

    audio_dir = asvspoof_root / f"ASVspoof2019_LA_{subset}" / "flac"
    protocol_ext = "trn" if subset == "train" else "trl"
    protocol_file = asvspoof_root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{subset}.{protocol_ext}.txt"

    if not protocol_file.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

    print(f"Loading ASVspoof2019 LA {subset} from {protocol_file}...")

    audio_paths, labels, speaker_ids, system_ids = [], [], [], []

    with open(protocol_file, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            parts = line.strip().split()
            speaker_id = parts[0]
            audio_id = parts[1]
            system_id = parts[3]
            label_str = parts[4]

            audio_path = audio_dir / f"{audio_id}.flac"

            if audio_path.exists():
                audio_paths.append(str(audio_path))
                labels.append(0 if label_str == 'bonafide' else 1)
                speaker_ids.append(speaker_id)
                system_ids.append(system_id)

    print(f"  Total: {len(audio_paths)}")
    print(f"  Bonafide: {sum(1 for l in labels if l == 0)}")
    print(f"  Spoof: {sum(1 for l in labels if l == 1)}")
    print(f"  Unique speakers: {len(set(speaker_ids))}")

    return audio_paths, labels, speaker_ids, system_ids


# =============================================================================
# Balanced Sampling
# =============================================================================
def balanced_sample_synthetic(labels, speaker_ids, system_ids, random_state=42):
    """
    Sample synthetic data to match natural count,
    uniformly by (attack_type, speaker) groups.

    Args:
        labels: list of labels (0=bonafide, 1=spoof)
        speaker_ids: list of speaker IDs
        system_ids: list of system/attack type IDs
        random_state: random seed

    Returns:
        nat_indices: indices of all natural samples
        sampled_syn_indices: indices of sampled synthetic samples
    """
    rng = np.random.RandomState(random_state)

    labels = np.array(labels)
    speaker_ids = np.array(speaker_ids)
    system_ids = np.array(system_ids)

    nat_indices = np.where(labels == 0)[0]
    syn_indices = np.where(labels == 1)[0]

    n_target = len(nat_indices)

    # Group synthetic by (system_id, speaker_id)
    groups = {}
    for idx in syn_indices:
        key = (system_ids[idx], speaker_ids[idx])
        groups.setdefault(key, []).append(idx)

    n_groups = len(groups)
    base_count = n_target // n_groups
    remainder = n_target % n_groups

    # Randomly select which groups get an extra sample
    group_keys = sorted(groups.keys())
    shuffled_keys = list(group_keys)
    rng.shuffle(shuffled_keys)
    extra_groups = set(shuffled_keys[:remainder])

    sampled_syn = []
    deficit = 0
    for key in group_keys:
        indices = np.array(groups[key])
        target = base_count + (1 if key in extra_groups else 0)
        if target <= len(indices):
            sampled = rng.choice(indices, target, replace=False)
        else:
            sampled = indices
            deficit += target - len(indices)
        sampled_syn.extend(sampled.tolist())

    # Handle deficit: sample extra from remaining pool
    if deficit > 0:
        sampled_set = set(sampled_syn)
        remaining_pool = [i for i in syn_indices if i not in sampled_set]
        if remaining_pool:
            extra = rng.choice(remaining_pool, min(deficit, len(remaining_pool)), replace=False)
            sampled_syn.extend(extra.tolist())

    # Ensure exactly n_target
    if len(sampled_syn) > n_target:
        sampled_syn = rng.choice(sampled_syn, n_target, replace=False).tolist()

    print(f"  Balanced sampling: {n_target} natural, {len(sampled_syn)} synthetic")
    print(f"  Synthetic groups (attack_type x speaker): {n_groups}, ~{base_count} per group")

    return nat_indices.tolist(), sampled_syn


def balanced_sample_by_speaker(speaker_ids, labels, random_state=42):
    """
    Sample equal number of samples per speaker with 1:1 natural/synthetic ratio.

    For each speaker, samples n_per_class natural + n_per_class synthetic,
    where n_per_class = min across all speakers of min(n_nat_i, n_syn_i).

    Args:
        speaker_ids: array of speaker IDs
        labels: array of labels (0=natural, 1=synthetic)
        random_state: random seed

    Returns:
        sampled_indices: list of sampled indices (into speaker_ids array)
        n_per_class: number of samples per class per speaker
    """
    rng = np.random.RandomState(random_state)
    speaker_ids = np.array(speaker_ids)
    labels = np.array(labels)
    unique_speakers = sorted(np.unique(speaker_ids))

    # For each speaker, find natural and synthetic indices
    speaker_nat = {}
    speaker_syn = {}
    for spk in unique_speakers:
        spk_mask = speaker_ids == spk
        speaker_nat[spk] = np.where(spk_mask & (labels == 0))[0]
        speaker_syn[spk] = np.where(spk_mask & (labels == 1))[0]

    # n_per_class = min across speakers of min(n_nat, n_syn)
    n_per_class = min(
        min(len(speaker_nat[spk]), len(speaker_syn[spk]))
        for spk in unique_speakers
    )

    sampled_indices = []
    for spk in unique_speakers:
        sampled_nat = rng.choice(speaker_nat[spk], n_per_class, replace=False)
        sampled_syn = rng.choice(speaker_syn[spk], n_per_class, replace=False)
        sampled_indices.extend(sorted(sampled_nat.tolist() + sampled_syn.tolist()))

    print(f"  Speaker-balanced sampling: {len(unique_speakers)} speakers, "
          f"{n_per_class} nat + {n_per_class} syn per speaker, {len(sampled_indices)} total")

    return sampled_indices, n_per_class


# =============================================================================
# Silhouette Plot Functions
# =============================================================================
def plot_silhouette_diagram(ax, silhouette_vals, labels, color, label_name,
                            alpha=0.7, show_cluster_labels=True):
    """
    Plot silhouette diagram on given axes.

    Args:
        ax: matplotlib axes
        silhouette_vals: per-sample silhouette values
        labels: cluster labels (0=Natural, 1=Synthetic)
        color: fill color
        label_name: legend label
        alpha: transparency
        show_cluster_labels: whether to annotate cluster names on y-axis
    """
    y_lower = 10
    cluster_names = {0: 'Natural', 1: 'Synthetic'}

    for cluster_id in [0, 1]:
        cluster_mask = labels == cluster_id
        cluster_values = silhouette_vals[cluster_mask]
        cluster_values = np.sort(cluster_values)

        size_cluster = cluster_values.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_values,
            alpha=alpha, color=color,
            label=label_name if cluster_id == 0 else None
        )

        if show_cluster_labels:
            ax.text(-0.05, y_lower + 0.5 * size_cluster, cluster_names[cluster_id],
                    fontsize=10, va='center', ha='right')

        y_lower = y_upper + 10

    return y_lower


def plot_silhouette_by_speaker(ax, silhouette_vals, speaker_ids, color_map,
                                speakers_to_show=None, alpha=0.7):
    """
    Plot silhouette diagram with speaker clusters.

    Args:
        ax: matplotlib axes
        silhouette_vals: per-sample silhouette values
        speaker_ids: speaker label per sample (numpy array)
        color_map: dict mapping speaker_id -> color
        speakers_to_show: list of speakers to include (None = all)
        alpha: transparency
    """
    unique_speakers = sorted(np.unique(speaker_ids))
    if speakers_to_show is not None:
        show_set = set(speakers_to_show)
        unique_speakers = [s for s in unique_speakers if s in show_set]

    y_lower = 10
    for spk in unique_speakers:
        mask = speaker_ids == spk
        cluster_values = np.sort(silhouette_vals[mask])
        size_cluster = len(cluster_values)
        y_upper = y_lower + size_cluster

        spk_mean = np.mean(cluster_values)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_values,
            alpha=alpha, color=color_map[spk]
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster,
                f'{spk} ({spk_mean:.3f})',
                fontsize=8, va='center', ha='right')

        y_lower = y_upper + 10

    return y_lower


def run_silhouette_plot_comparison(args):
    """
    Run silhouette plot comparison between:
      - Baseline: WavLM last layer features
      - Ours: WavLM layers 8,22 concatenated + speaker subspace removal

    Creates two figures:
      1. Overlaid: both methods on the same axes
      2. Side-by-side: baseline (left) and ours (right) subplots

    Training (scaler fit, speaker subspace computation) uses the full dataset.
    Silhouette score is computed on a balanced-sampled subset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full train dataset
    print("\n" + "=" * 60)
    print("Loading ASVspoof19 LA Train Dataset (full)")
    print("=" * 60)
    train_paths, train_labels, train_speakers, train_systems = prepare_asvspoof_dataset(
        args.train_root, subset='train', max_samples=args.max_train_samples
    )
    all_labels = np.array(train_labels)
    all_speakers = np.array(train_speakers)
    all_systems = np.array(train_systems)

    # === Baseline: WavLM last layer ===
    print("\n" + "=" * 60)
    print("Baseline: WavLM last layer (extracting full train set)")
    print("=" * 60)
    baseline_ssl = SSLFeatureExtractor(
        model_name='microsoft/wavlm-large',
        device=device,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    baseline_features_all, baseline_valid_all = baseline_ssl.extract_batch(train_paths, desc="Baseline")

    # Fit scaler on full data
    baseline_scaler = StandardScaler()
    baseline_features_all_scaled = baseline_scaler.fit_transform(baseline_features_all)

    # === Ours: WavLM layers 8,22 + speaker invariant ===
    print("\n" + "=" * 60)
    print(f"Ours: WavLM layers {WAVLM_SPEAKER_LAYERS} (extracting full train set)")
    print("=" * 60)
    ours_ssl = SSLFeatureExtractor(
        model_name='microsoft/wavlm-large',
        device=device,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        layers=WAVLM_SPEAKER_LAYERS
    )
    ours_features_all, ours_valid_all = ours_ssl.extract_batch(train_paths, desc="Ours")

    # Fit scaler on full data
    ours_scaler = StandardScaler()
    ours_features_all_scaled = ours_scaler.fit_transform(ours_features_all)

    # Compute speaker centroids and projection matrix on full data
    print(f"Computing speaker subspace on full data (n_components={args.n_speaker_components})...")
    ours_speakers_all = all_speakers[ours_valid_all]
    spk_map = {}
    for idx, spk in enumerate(ours_speakers_all):
        spk_map.setdefault(spk, []).append(idx)

    centroids = np.array([np.mean(ours_features_all_scaled[ids], axis=0) for ids in spk_map.values()])
    n_comp = min(args.n_speaker_components, len(centroids) - 1, ours_features_all_scaled.shape[1])
    pca_speaker = PCA(n_components=n_comp)
    pca_speaker.fit(centroids)
    U = pca_speaker.components_.T
    projection_matrix = np.eye(U.shape[0]) - (U @ U.T)
    ours_features_all_proj = ours_features_all_scaled @ projection_matrix

    print(f"  Speakers: {len(centroids)}, Components removed: {n_comp}")

    # === Balanced sampling for silhouette measurement ===
    # Use valid indices common to both methods
    baseline_valid_set = set(baseline_valid_all)
    ours_valid_set = set(ours_valid_all)
    common_valid = sorted(baseline_valid_set & ours_valid_set)

    # Build index mapping: original train index -> position in feature arrays
    baseline_idx_map = {orig: pos for pos, orig in enumerate(baseline_valid_all)}
    ours_idx_map = {orig: pos for pos, orig in enumerate(ours_valid_all)}

    common_labels = all_labels[common_valid]
    common_speakers = all_speakers[common_valid]
    common_systems = all_systems[common_valid]

    # Save common features for external speaker analysis (float16 npz for size)
    print("\nSaving common features for speaker analysis...")
    common_baseline_pos = [baseline_idx_map[o] for o in common_valid]
    common_ours_pos = [ours_idx_map[o] for o in common_valid]
    np.savez_compressed(output_dir / 'common_baseline_features.npz',
                        data=baseline_features_all_scaled[common_baseline_pos].astype(np.float16))
    np.savez_compressed(output_dir / 'common_ours_features.npz',
                        data=ours_features_all_proj[common_ours_pos].astype(np.float16))
    np.savez_compressed(output_dir / 'common_labels.npz', data=common_labels)
    np.savez_compressed(output_dir / 'common_speakers.npz', data=common_speakers)
    print(f"Saved common features (.npz float16) to {output_dir}")

    print("\nBalanced sampling of synthetic data for silhouette measurement...")

    nat_sub, syn_sub = balanced_sample_synthetic(
        common_labels.tolist(), common_speakers.tolist(), common_systems.tolist()
    )
    sample_sub = nat_sub + syn_sub  # indices into common_valid

    # Map sampled indices back to feature array positions
    sampled_orig = [common_valid[i] for i in sample_sub]
    baseline_pos = [baseline_idx_map[o] for o in sampled_orig]
    ours_pos = [ours_idx_map[o] for o in sampled_orig]

    baseline_features_sampled = baseline_features_all_scaled[baseline_pos]
    ours_features_sampled = ours_features_all_proj[ours_pos]
    sampled_labels = all_labels[sampled_orig]

    print(f"  Silhouette subset: {len(sampled_labels)} samples "
          f"(Natural={np.sum(sampled_labels == 0)}, Synthetic={np.sum(sampled_labels == 1)})")

    # === Compute per-sample silhouette values (cosine metric) ===
    print("\nComputing silhouette values (cosine metric)...")
    baseline_sil_values = silhouette_samples(baseline_features_sampled, sampled_labels, metric='cosine')
    ours_sil_values = silhouette_samples(ours_features_sampled, sampled_labels, metric='cosine')

    baseline_sil_score = np.mean(baseline_sil_values)
    ours_sil_score = np.mean(ours_sil_values)

    print(f"  Baseline silhouette score: {baseline_sil_score:.4f}")
    print(f"  Ours silhouette score:     {ours_sil_score:.4f}")

    # Per-cluster breakdown
    for cluster_id, name in [(0, 'Natural'), (1, 'Synthetic')]:
        b_mean = np.mean(baseline_sil_values[sampled_labels == cluster_id])
        o_mean = np.mean(ours_sil_values[sampled_labels == cluster_id])
        print(f"  {name}: Baseline={b_mean:.4f}, Ours={o_mean:.4f}")

    # Shared x-axis limits
    x_min = min(baseline_sil_values.min(), ours_sil_values.min()) - 0.05
    x_max = max(baseline_sil_values.max(), ours_sil_values.max()) + 0.05

    # === Plot 1: Overlaid ===
    print("\nGenerating overlaid silhouette plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    plot_silhouette_diagram(ax, baseline_sil_values, sampled_labels,
                            color='tab:blue', label_name='Baseline',
                            alpha=0.45, show_cluster_labels=True)
    plot_silhouette_diagram(ax, ours_sil_values, sampled_labels,
                            color='tab:orange', label_name='Ours',
                            alpha=0.45, show_cluster_labels=False)

    ax.axvline(x=baseline_sil_score, color='tab:blue', linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Baseline mean ({baseline_sil_score:.4f})')
    ax.axvline(x=ours_sil_score, color='tab:orange', linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Ours mean ({ours_sil_score:.4f})')

    ax.set_xlabel('Silhouette Coefficient (cosine)', fontsize=12)
    ax.set_ylabel('Samples (sorted)', fontsize=12)
    ax.set_title('Silhouette Plot: Baseline vs Ours', fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    overlaid_path = output_dir / 'silhouette_comparison_overlaid.png'
    plt.savefig(overlaid_path, dpi=150)
    plt.close()
    print(f"Saved: {overlaid_path}")

    # === Plot 2: Side-by-side subplots ===
    print("Generating side-by-side silhouette plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # Left: Baseline
    plot_silhouette_diagram(ax1, baseline_sil_values, sampled_labels,
                            color='tab:blue', label_name='Baseline', alpha=0.7)
    ax1.axvline(x=baseline_sil_score, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Mean ({baseline_sil_score:.4f})')
    ax1.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax1.set_ylabel('Samples (sorted)', fontsize=11)
    ax1.set_title(f'Baseline (WavLM last layer)', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.set_yticks([])
    ax1.legend(loc='lower right', fontsize=10)

    # Right: Ours
    plot_silhouette_diagram(ax2, ours_sil_values, sampled_labels,
                            color='tab:orange', label_name='Ours', alpha=0.7)
    ax2.axvline(x=ours_sil_score, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Mean ({ours_sil_score:.4f})')
    ax2.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax2.set_ylabel('Samples (sorted)', fontsize=11)
    ax2.set_title(f'Ours (WavLM L8+L22, SI n={args.n_speaker_components})', fontsize=12)
    ax2.set_xlim(x_min, x_max)
    ax2.set_yticks([])
    ax2.legend(loc='lower right', fontsize=10)

    plt.suptitle('Silhouette Plot Comparison (ASVspoof19 LA Train)', fontsize=14, y=1.01)
    plt.tight_layout()

    subplot_path = output_dir / 'silhouette_comparison_subplots.png'
    plt.savefig(subplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {subplot_path}")

    # Save numeric results
    results = {
        'baseline_silhouette': baseline_sil_score,
        'ours_silhouette': ours_sil_score,
        'baseline_silhouette_natural': np.mean(baseline_sil_values[sampled_labels == 0]),
        'baseline_silhouette_synthetic': np.mean(baseline_sil_values[sampled_labels == 1]),
        'ours_silhouette_natural': np.mean(ours_sil_values[sampled_labels == 0]),
        'ours_silhouette_synthetic': np.mean(ours_sil_values[sampled_labels == 1]),
        'n_natural': int(np.sum(sampled_labels == 0)),
        'n_synthetic': int(np.sum(sampled_labels == 1)),
        'n_train_total': len(train_paths),
    }
    results_df = pd.DataFrame([results])
    results_csv = output_dir / 'silhouette_comparison_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Saved: {results_csv}")

    # Save per-sample silhouette values for re-plotting
    np.save(output_dir / 'baseline_sil_values.npy', baseline_sil_values)
    np.save(output_dir / 'ours_sil_values.npy', ours_sil_values)
    np.save(output_dir / 'sampled_labels.npy', sampled_labels)
    print(f"Saved per-sample silhouette values to {output_dir}")

    return results


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_detector(detector, eval_paths, eval_labels, eval_speakers, mode_name):
    """
    Evaluate detector and compute all metrics

    Args:
        detector: Trained detector instance
        eval_paths: List of audio paths
        eval_labels: List of labels (0=bonafide, 1=spoof)
        eval_speakers: List of speaker IDs
        mode_name: Name of the evaluation mode
    """
    # Get predictions
    predictions, probabilities, valid_indices = detector.predict_batch(eval_paths)

    # Filter labels and speakers for valid indices
    eval_labels_valid = np.array(eval_labels)[valid_indices]
    eval_speakers_valid = np.array(eval_speakers)[valid_indices]

    # Compute EER
    eer = compute_eer(eval_labels_valid, probabilities)

    # Compute clustering metrics on TRAINING data
    train_features = detector._train_features
    train_labels = detector._train_labels
    train_speakers = detector._train_speakers
    train_clustering = compute_clustering_metrics(train_features, train_labels, train_speakers)

    # Compute clustering metrics on EVAL data
    print("Computing clustering metrics on eval data...")
    eval_features, _ = detector.get_features(eval_paths)
    eval_clustering = compute_clustering_metrics(eval_features, eval_labels_valid, eval_speakers_valid)

    results = {
        'mode': mode_name,
        'eer': eer,
        'accuracy': accuracy_score(eval_labels_valid, predictions),
        # Training data clustering metrics
        'train_silhouette_nat_syn': train_clustering['silhouette_nat_syn'],
        'train_davies_bouldin_nat_syn': train_clustering['davies_bouldin_nat_syn'],
        'train_silhouette_speaker': train_clustering['silhouette_speaker'],
        'train_davies_bouldin_speaker': train_clustering['davies_bouldin_speaker'],
        # Eval data clustering metrics
        'eval_silhouette_nat_syn': eval_clustering['silhouette_nat_syn'],
        'eval_davies_bouldin_nat_syn': eval_clustering['davies_bouldin_nat_syn'],
        'eval_silhouette_speaker': eval_clustering['silhouette_speaker'],
        'eval_davies_bouldin_speaker': eval_clustering['davies_bouldin_speaker'],
        # For visualization (using eval data - original high-dim features)
        'features': eval_features,
        'labels': eval_labels_valid,
        'speakers': eval_speakers_valid,
    }

    return results


# =============================================================================
# Main
# =============================================================================
def main(args):
    # Silhouette plot comparison mode
    if args.silhouette_plot:
        run_silhouette_plot_comparison(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n" + "="*60)
    print("Loading Training Dataset (ASV19 LA train)")
    print("="*60)
    train_paths, train_labels, train_speakers, _ = prepare_asvspoof_dataset(
        args.train_root, subset='train', max_samples=args.max_train_samples
    )

    print("\n" + "="*60)
    print("Loading Evaluation Dataset (ASV19 LA eval)")
    print("="*60)
    eval_paths, eval_labels, eval_speakers, _ = prepare_asvspoof_dataset(
        args.eval_root, subset='eval', max_samples=args.max_eval_samples
    )

    # Resolve model name from short name if needed
    model_name = SSL_MODELS.get(args.ssl_model, args.ssl_model)
    model_short_name = args.ssl_model if args.ssl_model in SSL_MODELS else model_name.split('/')[-1]

    # Shared SSL feature extractor
    ssl_extractor = SSLFeatureExtractor(
        model_name=model_name,
        device=device,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    all_results = []

    # Mode 1: SSL Baseline (No Speaker Invariant)
    if args.mode in ['baseline', 'both']:
        print("\n" + "="*60)
        print(f"Mode 1: {model_short_name} Baseline (No Speaker Invariant)")
        print("="*60)

        baseline_detector = SSLBaselineDetector(
            ssl_extractor=ssl_extractor,
            device=device
        )

        # Train
        train_acc = baseline_detector.fit(train_paths, train_labels, train_speakers)

        # Evaluate
        print("\nEvaluating on ASV19 LA eval...")
        baseline_results = evaluate_detector(
            baseline_detector, eval_paths, eval_labels, eval_speakers, "baseline"
        )

        # Visualization (t-SNE)
        plot_clustering_visualization(
            baseline_results['features'],
            baseline_results['labels'],
            f"{model_short_name} Baseline (EER={baseline_results['eer']:.2f}%)",
            str(output_dir / f"{model_short_name}_baseline_tsne.png")
        )

        all_results.append(baseline_results)

        print(f"\n[Baseline Results]")
        print(f"  EER: {baseline_results['eer']:.2f}%")
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  [Train - Nat/Syn Clustering]")
        print(f"    Silhouette: {baseline_results['train_silhouette_nat_syn']:.4f}")
        print(f"    Davies-Bouldin: {baseline_results['train_davies_bouldin_nat_syn']:.4f}")
        print(f"  [Train - Speaker Clustering]")
        print(f"    Silhouette: {baseline_results['train_silhouette_speaker']:.4f}")
        print(f"    Davies-Bouldin: {baseline_results['train_davies_bouldin_speaker']:.4f}")
        print(f"  [Eval - Nat/Syn Clustering]")
        print(f"    Silhouette: {baseline_results['eval_silhouette_nat_syn']:.4f}")
        print(f"    Davies-Bouldin: {baseline_results['eval_davies_bouldin_nat_syn']:.4f}")
        print(f"  [Eval - Speaker Clustering]")
        print(f"    Silhouette: {baseline_results['eval_silhouette_speaker']:.4f}")
        print(f"    Davies-Bouldin: {baseline_results['eval_davies_bouldin_speaker']:.4f}")

    # Mode 2: SSL + Speaker Invariant
    if args.mode in ['speaker_invariant', 'both']:
        print("\n" + "="*60)
        print(f"Mode 2: {model_short_name} + Speaker Invariant (n_components={args.n_speaker_components})")
        print("="*60)

        si_detector = SSLSpeakerInvariantDetector(
            n_speaker_components=args.n_speaker_components,
            ssl_extractor=ssl_extractor,
            model_name=model_name,
            device=device,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Train
        train_acc = si_detector.fit(train_paths, train_labels, train_speakers)

        # Evaluate
        print("\nEvaluating on ASV19 LA eval...")
        si_results = evaluate_detector(
            si_detector, eval_paths, eval_labels, eval_speakers, f"speaker_invariant_n{args.n_speaker_components}"
        )

        # Visualization (t-SNE)
        plot_clustering_visualization(
            si_results['features'],
            si_results['labels'],
            f"{model_short_name} + SI (n={args.n_speaker_components}, EER={si_results['eer']:.2f}%)",
            str(output_dir / f"{model_short_name}_si_n{args.n_speaker_components}_tsne.png")
        )

        all_results.append(si_results)

        print(f"\n[Speaker Invariant Results]")
        print(f"  EER: {si_results['eer']:.2f}%")
        print(f"  Accuracy: {si_results['accuracy']:.4f}")
        print(f"  [Train - Nat/Syn Clustering]")
        print(f"    Silhouette: {si_results['train_silhouette_nat_syn']:.4f}")
        print(f"    Davies-Bouldin: {si_results['train_davies_bouldin_nat_syn']:.4f}")
        print(f"  [Train - Speaker Clustering]")
        print(f"    Silhouette: {si_results['train_silhouette_speaker']:.4f}")
        print(f"    Davies-Bouldin: {si_results['train_davies_bouldin_speaker']:.4f}")
        print(f"  [Eval - Nat/Syn Clustering]")
        print(f"    Silhouette: {si_results['eval_silhouette_nat_syn']:.4f}")
        print(f"    Davies-Bouldin: {si_results['eval_davies_bouldin_nat_syn']:.4f}")
        print(f"  [Eval - Speaker Clustering]")
        print(f"    Silhouette: {si_results['eval_silhouette_speaker']:.4f}")
        print(f"    Davies-Bouldin: {si_results['eval_davies_bouldin_speaker']:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame([{
        'mode': r['mode'],
        'eer': r['eer'],
        'accuracy': r['accuracy'],
        # Train clustering metrics
        'train_silhouette_nat_syn': r['train_silhouette_nat_syn'],
        'train_davies_bouldin_nat_syn': r['train_davies_bouldin_nat_syn'],
        'train_silhouette_speaker': r['train_silhouette_speaker'],
        'train_davies_bouldin_speaker': r['train_davies_bouldin_speaker'],
        # Eval clustering metrics
        'eval_silhouette_nat_syn': r['eval_silhouette_nat_syn'],
        'eval_davies_bouldin_nat_syn': r['eval_davies_bouldin_nat_syn'],
        'eval_silhouette_speaker': r['eval_silhouette_speaker'],
        'eval_davies_bouldin_speaker': r['eval_davies_bouldin_speaker'],
    } for r in all_results])

    results_csv_path = output_dir / f"{model_short_name}_clustering_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SSL Model Deepfake Detection with Clustering Metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Build model choices string for help
    model_choices_help = "SSL model to use. Short names: " + ", ".join(SSL_MODELS.keys())

    parser.add_argument('--train_root', type=str, required=True,
                        help='ASVspoof2019 LA root directory for training')
    parser.add_argument('--eval_root', type=str, default=None,
                        help='ASVspoof2019 LA root directory for evaluation (required unless --silhouette_plot)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['baseline', 'speaker_invariant', 'both'],
                        help='Evaluation mode')
    parser.add_argument('--silhouette_plot', action='store_true',
                        help='Run silhouette plot comparison (Baseline vs Ours) on train data')
    parser.add_argument('--ssl_model', type=str, default='hubert-large',
                        help=model_choices_help)
    parser.add_argument('--n_speaker_components', type=int, default=10,
                        help='Number of speaker components to remove (for speaker_invariant mode)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples (None for all)')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Maximum evaluation samples (None for all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for GPU inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for audio loading')
    parser.add_argument('--cache_dir', type=str, default='_cache/ssl',
                        help='Cache directory for features')
    parser.add_argument('--output_dir', type=str, default='results/ssl_clustering',
                        help='Output directory for results')

    args = parser.parse_args()

    if not args.silhouette_plot and args.eval_root is None:
        parser.error("--eval_root is required unless --silhouette_plot is set")

    main(args)
