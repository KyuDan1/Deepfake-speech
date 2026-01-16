#!/usr/bin/env python3
"""
Test script: In-The-Wild 데이터셋 로드 및 모델 평가
- RawNet2 Frozen
- AASIST Frozen
- WavLM Frozen Baseline
- Speaker-Invariant Detector

Metrics: EER, Precision, Recall, F1, Accuracy
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import yaml
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# ============================================================================
# Config
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

INTHEWILD_ROOT = Path("/mnt/tmp/Deepfake-speech/data/InTheWild_hf")
MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
TRAINED_DIR = MODELS_DIR / "trained"
CACHE_DIR = Path("/mnt/tmp/Deepfake-speech/cache")

# ============================================================================
# Metrics Functions
# ============================================================================
def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal Error Rate 계산"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
    return eer * 100

def evaluate_detector(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    """종합 평가 메트릭 계산"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'eer': compute_eer(y_true, y_scores)
    }

def print_metrics(metrics: Dict, model_name: str):
    """메트릭 출력"""
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  EER:       {metrics['eer']:.2f}%")

# ============================================================================
# 1. Load In-The-Wild Dataset
# ============================================================================
print("\n" + "=" * 70)
print("Loading In-The-Wild Dataset")
print("=" * 70)

release_dir = INTHEWILD_ROOT / "release_in_the_wild"
if not release_dir.exists():
    release_dir = INTHEWILD_ROOT

meta_path = release_dir / "meta.csv"
print(f"Meta path: {meta_path}")
print(f"Exists: {meta_path.exists()}")

if not meta_path.exists():
    print("[ERROR] meta.csv not found!")
    sys.exit(1)

meta_df = pd.read_csv(meta_path)
print(f"Total samples in meta.csv: {len(meta_df)}")
print(f"Columns: {meta_df.columns.tolist()}")
print(f"\nLabel distribution:")
print(meta_df['label'].value_counts())

# Build DataFrame
data = []
for _, row in meta_df.iterrows():
    file_name = row['file']
    label = row['label']
    audio_path = release_dir / file_name
    if audio_path.exists():
        binary_label = 0 if label == 'bona-fide' else 1
        data.append({
            'file_id': Path(file_name).stem,
            'label': label,
            'binary_label': binary_label,
            'audio_path': str(audio_path)
        })

inthewild_df = pd.DataFrame(data)
print(f"\nLoaded {len(inthewild_df)} samples")
print(f"  Real (bona-fide): {(inthewild_df['binary_label']==0).sum()}")
print(f"  Fake (spoof): {(inthewild_df['binary_label']==1).sum()}")

# Prepare for evaluation
audio_paths = inthewild_df['audio_path'].tolist()
labels = np.array(inthewild_df['binary_label'].tolist())

# Limit samples for faster testing (optional)
MAX_SAMPLES = None  # Set to e.g., 500 for quick test, None for full
if MAX_SAMPLES and len(audio_paths) > MAX_SAMPLES:
    print(f"\n[Note] Limiting to {MAX_SAMPLES} samples for faster testing")
    indices = np.random.choice(len(audio_paths), MAX_SAMPLES, replace=False)
    audio_paths = [audio_paths[i] for i in indices]
    labels = labels[indices]

all_results = []

# ============================================================================
# 2. Load and Evaluate WavLM Frozen Baseline
# ============================================================================
print("\n" + "=" * 70)
print("Testing WavLM Frozen Baseline")
print("=" * 70)

wavlm_model_path = TRAINED_DIR / "wavlm_frozen_baseline.pkl"
print(f"Model path: {wavlm_model_path}")
print(f"Exists: {wavlm_model_path.exists()}")

if wavlm_model_path.exists():
    try:
        # Load WavLM Feature Extractor
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        print("Loading WavLM model...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        wavlm_model.eval()
        for param in wavlm_model.parameters():
            param.requires_grad = False

        # Load trained classifier
        with open(wavlm_model_path, 'rb') as f:
            save_data = pickle.load(f)
        scaler = save_data['scaler']
        classifier = save_data['classifier']

        print("Extracting features and predicting...")
        features = []
        for path in tqdm(audio_paths, desc="WavLM Feature Extraction"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                with torch.no_grad():
                    outputs = wavlm_model(input_values)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                features.append(feat)
            except Exception as e:
                features.append(np.zeros(1024))

        features = np.array(features)
        features_scaled = scaler.transform(features)
        predictions = classifier.predict(features_scaled)
        scores = classifier.predict_proba(features_scaled)[:, 1]

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'WavLM Frozen'
        all_results.append(metrics)
        print_metrics(metrics, 'WavLM Frozen Baseline')

    except Exception as e:
        print(f"[ERROR] WavLM Frozen: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] WavLM Frozen model not found")

# ============================================================================
# 3. Load and Evaluate Speaker-Invariant Detector (best n=10)
# ============================================================================
print("\n" + "=" * 70)
print("Testing Speaker-Invariant Detector")
print("=" * 70)

si_model_path = TRAINED_DIR / "si_detector_n10.pkl"
print(f"Model path: {si_model_path}")
print(f"Exists: {si_model_path.exists()}")

if si_model_path.exists():
    try:
        with open(si_model_path, 'rb') as f:
            si_data = pickle.load(f)

        si_scaler = si_data['scaler']
        si_projection_matrix = si_data['projection_matrix']
        si_classifier = si_data['classifier']

        # Reuse features from WavLM if available
        if 'features' in dir() and features is not None:
            print("Reusing WavLM features...")
            features_scaled = si_scaler.transform(features)
            features_projected = features_scaled @ si_projection_matrix
            predictions = si_classifier.predict(features_projected)
            scores = si_classifier.predict_proba(features_projected)[:, 1]

            metrics = evaluate_detector(labels, predictions, scores)
            metrics['model'] = 'SI Detector (n=10)'
            all_results.append(metrics)
            print_metrics(metrics, 'Speaker-Invariant Detector (n=10)')
        else:
            print("[SKIP] Need WavLM features first")

    except Exception as e:
        print(f"[ERROR] SI Detector: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] SI Detector model not found")

# ============================================================================
# 4. Load and Evaluate RawNet2 Frozen
# ============================================================================
print("\n" + "=" * 70)
print("Testing RawNet2 Frozen Baseline")
print("=" * 70)

rawnet2_code_dir = MODELS_DIR / "asvspoof2021_baseline" / "2021" / "DF" / "Baseline-RawNet2"
rawnet2_weights_path = rawnet2_code_dir / "models" / "model_DF_CCE_100_32_0.0001" / "epoch_70.pth"

print(f"Code dir: {rawnet2_code_dir}")
print(f"Code dir exists: {rawnet2_code_dir.exists()}")
print(f"Weights path: {rawnet2_weights_path}")
print(f"Weights exists: {rawnet2_weights_path.exists()}")

if rawnet2_code_dir.exists() and rawnet2_weights_path.exists():
    try:
        if str(rawnet2_code_dir) not in sys.path:
            sys.path.insert(0, str(rawnet2_code_dir))

        from model import RawNet

        # Load config
        config_path = rawnet2_code_dir / "model_config_RawNet.yaml"
        if not config_path.exists():
            config_path = rawnet2_code_dir / "model_config_RawNet2.yaml"

        print(f"Config path: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', config)
        print(f"Model config loaded")

        # Initialize and load model
        rawnet2_model = RawNet(model_config, device).to(device)
        checkpoint = torch.load(rawnet2_weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            rawnet2_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            rawnet2_model.load_state_dict(checkpoint)
        rawnet2_model.eval()
        print("RawNet2 model loaded successfully!")

        # Inference
        nb_samp = 64000
        predictions = []
        scores = []

        for path in tqdm(audio_paths, desc="RawNet2 Inference"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) < nb_samp:
                    num_repeats = int(nb_samp / len(audio)) + 1
                    audio = np.tile(audio, num_repeats)[:nb_samp]
                else:
                    audio = audio[:nb_samp]

                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = rawnet2_model(audio_tensor)
                    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                spoof_prob = probs[0]  # Index 0 = spoof
                predictions.append(1 if spoof_prob > 0.5 else 0)
                scores.append(float(spoof_prob))
            except Exception as e:
                predictions.append(0)
                scores.append(0.0)

        predictions = np.array(predictions)
        scores = np.array(scores)

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'RawNet2 Frozen'
        all_results.append(metrics)
        print_metrics(metrics, 'RawNet2 Frozen Baseline')

    except Exception as e:
        print(f"[ERROR] RawNet2: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] RawNet2 model files not found")

# ============================================================================
# 5. Load and Evaluate AASIST Frozen
# ============================================================================
print("\n" + "=" * 70)
print("Testing AASIST Frozen Baseline")
print("=" * 70)

aasist_dir = MODELS_DIR / "aasist"
print(f"AASIST dir: {aasist_dir}")
print(f"Exists: {aasist_dir.exists()}")

if aasist_dir.exists():
    try:
        models_dir = aasist_dir / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from AASIST import Model as AASIST
        import json

        # Load config
        config_path = aasist_dir / "config" / "AASIST.conf"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config.get('model_config', config)
        else:
            print(f"Config not found at {config_path}")
            model_config = None

        # Find weights
        weights_path = aasist_dir / "models" / "weights" / "AASIST.pth"
        print(f"Weights path: {weights_path}")
        print(f"Weights exists: {weights_path.exists()}")

        if model_config and weights_path.exists():
            aasist_model = AASIST(model_config).to(device)
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                aasist_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                aasist_model.load_state_dict(checkpoint)
            aasist_model.eval()
            print("AASIST model loaded successfully!")

            # Inference
            nb_samp = 64600
            predictions = []
            scores = []

            for path in tqdm(audio_paths, desc="AASIST Inference"):
                try:
                    audio, _ = librosa.load(path, sr=16000, mono=True)
                    if len(audio) < nb_samp:
                        num_repeats = int(nb_samp / len(audio)) + 1
                        audio = np.tile(audio, num_repeats)[:nb_samp]
                    else:
                        audio = audio[:nb_samp]

                    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, output = aasist_model(audio_tensor)
                        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                    spoof_prob = probs[0]  # Index 0 = spoof
                    predictions.append(1 if spoof_prob > 0.5 else 0)
                    scores.append(float(spoof_prob))
                except Exception as e:
                    predictions.append(0)
                    scores.append(0.0)

            predictions = np.array(predictions)
            scores = np.array(scores)

            metrics = evaluate_detector(labels, predictions, scores)
            metrics['model'] = 'AASIST Frozen'
            all_results.append(metrics)
            print_metrics(metrics, 'AASIST Frozen Baseline')
        else:
            print("[SKIP] AASIST config or weights not found")

    except Exception as e:
        print(f"[ERROR] AASIST: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] AASIST directory not found")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: In-The-Wild Evaluation Results")
print("=" * 70)

if all_results:
    results_df = pd.DataFrame(all_results)
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'eer']
    results_df = results_df[cols]
    print(results_df.to_string(index=False))
else:
    print("No models were successfully evaluated.")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
