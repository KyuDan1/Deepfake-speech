"""
WaveFake 데이터셋만 평가하는 스크립트
기존 cross_dataset_eval.py에서 학습된 모델들을 로드하여 WaveFake에 대해서만 평가 수행
결과는 기존 cross_dataset_results.csv에 추가됨
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

class Config:
    WAVEFAKE_ROOT = Path("/mnt/tmp/Deepfake-speech/data/WaveFake")
    MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
    TRAINED_DIR = MODELS_DIR / "trained"
    RAWNET2_DIR = MODELS_DIR / "asvspoof2021_baseline/2021/DF/Baseline-RawNet2"
    AASIST_DIR = MODELS_DIR / "aasist"
    RESULTS_DIR = Path("/mnt/tmp/Deepfake-speech/results")
    CACHE_DIR = Path("/mnt/tmp/Deepfake-speech/cache")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# WaveFake DataLoader
# =============================================================================

def get_wavefake_data(max_samples: int = None) -> Optional[pd.DataFrame]:
    """
    WaveFake 데이터셋 로드 (Generated + Reference)
    """
    if not Config.WAVEFAKE_ROOT.exists():
        print(f"Error: WaveFake not found at {Config.WAVEFAKE_ROOT}")
        return None

    data = []
    print(f"Loading WaveFake from {Config.WAVEFAKE_ROOT}...")

    # 1. Load Reference Data (Real / Bonafide)

    # (1) LJSpeech
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

    # (2) JSUT (Only use 'basic5000')
    jsut_root = Config.WAVEFAKE_ROOT / "jsut_ver1.1"
    jsut_target_dir = jsut_root / "basic5000"

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
    else:
        print(f"  [Warning] JSUT 'basic5000' folder not found at {jsut_target_dir}")

    # 2. Load Generated Data (Fake / Spoof)
    gen_dir = Config.WAVEFAKE_ROOT / "generated_audio"
    if not gen_dir.exists():
        gen_dir = Config.WAVEFAKE_ROOT

    fake_files = []
    for algo_path in gen_dir.iterdir():
        if algo_path.is_dir() and ("ljspeech" in algo_path.name or "jsut" in algo_path.name):
            curr_files = list(algo_path.glob("*.wav"))
            algo_name = algo_path.name
            fake_files.extend([(f, algo_name) for f in curr_files])

    print(f"  [Fake] Generated: found {len(fake_files)} files (across various vocoders)")

    for f, algo_name in fake_files:
        data.append({
            'file_id': f.stem,
            'source': algo_name,
            'label': 'spoof',
            'binary_label': 1,
            'audio_path': str(f)
        })

    if len(data) == 0:
        print("Error: No audio files found in WaveFake directory")
        return None

    df = pd.DataFrame(data)

    # 3. Sampling (Optional)
    if max_samples and len(df) > max_samples:
        df_real = df[df['binary_label'] == 0]
        df_fake = df[df['binary_label'] == 1]

        n_real = min(len(df_real), max_samples // 2)
        n_fake = min(len(df_fake), max_samples // 2)

        df_real = df_real.sample(n=n_real, random_state=42)
        df_fake = df_fake.sample(n=n_fake, random_state=42)

        df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  -> Subsampled to {len(df)} files (Balanced Real/Fake)")

    print(f"Total WaveFake Loaded: {len(df)} (Real: {(df['binary_label']==0).sum()}, Fake: {(df['binary_label']==1).sum()})")
    return df

# =============================================================================
# WavLM Feature Extractor
# =============================================================================

class WavLMFeatureExtractor:
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
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
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
# Model Wrappers (for loading saved models)
# =============================================================================

class WavLMFrozenBaseline:
    def __init__(self, wavlm_extractor: WavLMFeatureExtractor = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wavlm = wavlm_extractor
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_fitted = False

    def load(self, path: str):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.scaler = save_data['scaler']
        self.classifier = save_data['classifier']
        self.is_fitted = save_data['is_fitted']
        print(f"Model loaded from {path}")

    def predict_batch(
        self,
        audio_paths: List[str],
        cache_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call load() first.")

        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.transform(X)

        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        scores = probabilities[:, 1]

        return predictions, scores


class SpeakerInvariantDetector:
    def __init__(
        self,
        n_speaker_components: int = 10,
        wavlm_extractor: WavLMFeatureExtractor = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_speaker_components = n_speaker_components
        self.wavlm = wavlm_extractor
        self.pca = None
        self.projection_matrix = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        self.is_fitted = False

    def load(self, path: str):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.scaler = save_data['scaler']
        self.classifier = save_data['classifier']
        self.projection_matrix = save_data['projection_matrix']
        self.n_speaker_components = save_data['n_speaker_components']
        self.is_fitted = save_data['is_fitted']
        print(f"SI-Detector (n={self.n_speaker_components}) loaded from {path}")

    def predict_batch(
        self,
        audio_paths: List[str],
        cache_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call load() first.")

        X = self.wavlm.extract_batch(audio_paths, cache_path=cache_path)
        X_scaled = self.scaler.transform(X)
        X_proj = X_scaled @ self.projection_matrix

        predictions = self.classifier.predict(X_proj)
        probabilities = self.classifier.predict_proba(X_proj)
        scores = probabilities[:, 1]

        return predictions, scores

# =============================================================================
# RawNet2 & AASIST (End-to-End Models)
# =============================================================================

class RawNet2Baseline:
    def __init__(self, model_dir: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.code_dir = model_dir  # DF/Baseline-RawNet2
        self.weights_path = self.code_dir / "models" / "model_DF_CCE_100_32_0.0001" / "epoch_70.pth"
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import yaml

            # sys.path에 모델 디렉토리 추가
            if str(self.code_dir) not in sys.path:
                sys.path.insert(0, str(self.code_dir))

            from model import RawNet

            config_path = self.code_dir / "model_config_RawNet.yaml"

            if not config_path.exists():
                print(f"RawNet2: Config not found at {config_path}")
                return

            if not self.weights_path.exists():
                print(f"RawNet2: Weights not found at {self.weights_path}")
                return

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)['model']

            self.model = RawNet(config, self.device).to(self.device)
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"RawNet2 loaded successfully from {self.weights_path}")

        except Exception as e:
            print(f"RawNet2 loading failed: {e}")
            self.is_loaded = False

    def predict_batch(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            raise ValueError("RawNet2 not loaded")

        predictions = []
        scores = []

        for path in tqdm(audio_paths, desc="RawNet2 inference"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                audio = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(audio)
                    prob = torch.softmax(output, dim=1)
                    score = prob[0, 1].item()
                    pred = 1 if score > 0.5 else 0

                predictions.append(pred)
                scores.append(score)
            except Exception as e:
                predictions.append(0)
                scores.append(0.5)

        return np.array(predictions), np.array(scores)


class AASISTBaseline:
    def __init__(self, model_dir: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            sys.path.insert(0, str(self.model_dir))
            from models.AASIST import Model as AASIST

            config_path = self.model_dir / "config" / "AASIST.conf"
            model_path = self.model_dir / "models" / "weights" / "AASIST.pth"

            if not model_path.exists():
                model_path = self.model_dir / "AASIST.pth"

            if not config_path.exists() or not model_path.exists():
                print(f"AASIST: Config or model not found")
                return

            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            self.model = AASIST(config['model_config']).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print("AASIST loaded successfully")

        except Exception as e:
            print(f"AASIST loading failed: {e}")
            self.is_loaded = False

    def predict_batch(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            raise ValueError("AASIST not loaded")

        predictions = []
        scores = []

        for path in tqdm(audio_paths, desc="AASIST inference"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                audio = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    _, output = self.model(audio)
                    prob = torch.softmax(output, dim=1)
                    score = prob[0, 1].item()
                    pred = 1 if score > 0.5 else 0

                predictions.append(pred)
                scores.append(score)
            except Exception as e:
                predictions.append(0)
                scores.append(0.5)

        return np.array(predictions), np.array(scores)

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = 0.5
    return eer * 100

def evaluate_detector(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'eer': compute_eer(y_true, y_scores)
    }
    return metrics

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("="*80)
    print("WaveFake Evaluation Script")
    print("="*80)

    # 1. Load WaveFake data
    print("\n[1] Loading WaveFake dataset...")
    wf_data = get_wavefake_data()

    if wf_data is None or len(wf_data) == 0:
        print("Error: WaveFake data not found or empty. Exiting.")
        return

    audio_paths = wf_data['audio_path'].tolist()
    labels = np.array(wf_data['binary_label'].tolist())

    # 2. Initialize WavLM extractor (shared) - 주석 처리: RawNet2는 WavLM 안 씀
    # print("\n[2] Initializing WavLM Feature Extractor...")
    # wavlm_extractor = WavLMFeatureExtractor(device=device)

    # 3. Load models
    print("\n[3] Loading trained models...")
    eval_models = {}

    # # WavLM Baseline
    # wavlm_baseline = WavLMFrozenBaseline(wavlm_extractor=wavlm_extractor, device=device)
    # wavlm_baseline.load(str(Config.TRAINED_DIR / "wavlm_frozen_baseline.pkl"))
    # eval_models['WavLM Baseline'] = wavlm_baseline

    # # SI-Detectors
    # for n in [1, 5, 10, 16]:
    #     si_detector = SpeakerInvariantDetector(
    #         n_speaker_components=n,
    #         wavlm_extractor=wavlm_extractor,
    #         device=device
    #     )
    #     model_path = Config.TRAINED_DIR / f"si_detector_n{n}.pkl"
    #     if model_path.exists():
    #         si_detector.load(str(model_path))
    #         eval_models[f'SI-Detector (n={n})'] = si_detector

    # RawNet2
    rawnet2 = RawNet2Baseline(Config.RAWNET2_DIR, device=device)
    if rawnet2.is_loaded:
        eval_models['RawNet2'] = rawnet2

    # # AASIST
    # aasist = AASISTBaseline(Config.AASIST_DIR, device=device)
    # if aasist.is_loaded:
    #     eval_models['AASIST'] = aasist

    print(f"\nModels to evaluate: {list(eval_models.keys())}")

    # 4. Run evaluation
    print("\n[4] Running evaluation on WaveFake...")
    print("-" * 60)

    results = []
    cache_path = str(Config.CACHE_DIR / "wavefake_wavlm_features.pkl")

    for model_name, model in eval_models.items():
        print(f"\n  Evaluating {model_name}...", end=" ", flush=True)

        try:
            if hasattr(model, 'predict_batch'):
                if 'cache_path' in model.predict_batch.__code__.co_varnames:
                    preds, scores = model.predict_batch(audio_paths, cache_path=cache_path)
                else:
                    print()
                    preds, scores = model.predict_batch(audio_paths)
            else:
                print("[Skip: no predict_batch method]")
                continue

            metrics = evaluate_detector(labels, preds, scores)
            metrics['model'] = model_name
            metrics['dataset'] = 'WaveFake'
            results.append(metrics)

            print(f"Done. -> EER: {metrics['eer']:.2f}% | Acc: {metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"[Error: {e}]")

    # 5. Save results
    print("\n[5] Saving results...")

    if len(results) > 0:
        new_results_df = pd.DataFrame(results)

        # Load existing results
        results_path = Config.RESULTS_DIR / "cross_dataset_results.csv"
        if results_path.exists():
            existing_df = pd.read_csv(results_path)
            # Remove any existing WaveFake results to avoid duplicates
            existing_df = existing_df[existing_df['dataset'] != 'WaveFake']
            # Append new results
            combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        else:
            combined_df = new_results_df

        # Save
        combined_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

        # Print summary
        print("\n" + "="*80)
        print("WaveFake Evaluation Results Summary")
        print("="*80)
        print(new_results_df[['model', 'accuracy', 'eer']].to_string(index=False))
    else:
        print("No results to save.")

    print("\n[Done]")

if __name__ == "__main__":
    main()
