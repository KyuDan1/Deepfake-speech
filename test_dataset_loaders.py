#!/usr/bin/env python3
"""
Test script: WaveFake & ASVspoof2021 DF 데이터셋 로드 및 모델 평가
- RawNet2 Frozen
- AASIST Frozen
- WavLM Frozen Baseline
- Speaker-Invariant Detector

Metrics: EER, Precision, Recall, F1, Accuracy

Usage:
    python test_dataset_loaders.py --dataset wavefake
    python test_dataset_loaders.py --dataset asvspoof21
    python test_dataset_loaders.py --dataset all
    python test_dataset_loaders.py --dataset wavefake --max-samples 500
"""

import os
import sys
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
import pickle
import yaml
import json
import librosa

# librosa audioread fallback 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
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

# Paths
DATA_ROOT = Path("/mnt/tmp/Deepfake-speech/data")
MODELS_DIR = Path("/mnt/tmp/Deepfake-speech/models")
TRAINED_DIR = MODELS_DIR / "trained"

# Dataset paths
WAVEFAKE_ROOT = DATA_ROOT / "WaveFake"
ASVSPOOF21_ROOT = DATA_ROOT / "ASVspoof2021"

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
# Dataset Loaders
# ============================================================================
def load_wavefake_dataset(max_samples: int = None) -> Tuple[List[str], np.ndarray]:
    """WaveFake 데이터셋 로드"""
    print("\n" + "=" * 70)
    print("Loading WaveFake Dataset")
    print("=" * 70)

    if not WAVEFAKE_ROOT.exists():
        print(f"[ERROR] WaveFake not found at {WAVEFAKE_ROOT}")
        return None, None

    data = []

    # 1. Real audio: LJSpeech
    ljs_dir = WAVEFAKE_ROOT / "LJSpeech-1.1" / "wavs"
    if ljs_dir.exists():
        files = list(ljs_dir.glob("*.wav"))
        print(f"  [Real] LJSpeech: {len(files)} files")
        for f in files:
            data.append({'audio_path': str(f), 'binary_label': 0, 'source': 'LJSpeech'})

    # 2. Real audio: JSUT (basic5000 only)
    jsut_dir = WAVEFAKE_ROOT / "jsut_ver1.1" / "basic5000"
    if jsut_dir.exists():
        files = list(jsut_dir.rglob("*.wav"))
        print(f"  [Real] JSUT basic5000: {len(files)} files")
        for f in files:
            data.append({'audio_path': str(f), 'binary_label': 0, 'source': 'JSUT'})

    # 3. Fake audio: generated_audio folder
    gen_dir = WAVEFAKE_ROOT / "generated_audio"
    if not gen_dir.exists():
        gen_dir = WAVEFAKE_ROOT

    fake_count = 0
    for algo_path in gen_dir.iterdir():
        if algo_path.is_dir() and ("ljspeech" in algo_path.name.lower() or "jsut" in algo_path.name.lower()):
            files = list(algo_path.glob("*.wav"))
            fake_count += len(files)
            for f in files:
                data.append({'audio_path': str(f), 'binary_label': 1, 'source': algo_path.name})

    print(f"  [Fake] Generated: {fake_count} files")

    if not data:
        print("[ERROR] No audio files found")
        return None, None

    df = pd.DataFrame(data)
    print(f"\nTotal: {len(df)} (Real: {(df['binary_label']==0).sum()}, Fake: {(df['binary_label']==1).sum()})")

    # Sampling
    if max_samples and len(df) > max_samples:
        df_real = df[df['binary_label'] == 0].sample(n=min(max_samples//2, (df['binary_label']==0).sum()), random_state=42)
        df_fake = df[df['binary_label'] == 1].sample(n=min(max_samples//2, (df['binary_label']==1).sum()), random_state=42)
        df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  -> Sampled to {len(df)} files")

    return df['audio_path'].tolist(), df['binary_label'].values


def load_asvspoof21_dataset(max_samples: int = None) -> Tuple[List[str], np.ndarray]:
    """ASVspoof2021 DF 데이터셋 로드"""
    print("\n" + "=" * 70)
    print("Loading ASVspoof2021 DF Dataset")
    print("=" * 70)

    if not ASVSPOOF21_ROOT.exists():
        print(f"[ERROR] ASVspoof2021 not found at {ASVSPOOF21_ROOT}")
        return None, None

    # Find protocol file
    protocol_path = ASVSPOOF21_ROOT / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    if not protocol_path.exists():
        candidates = list(ASVSPOOF21_ROOT.glob("**/trial_metadata.txt"))
        if candidates:
            protocol_path = candidates[0]
        else:
            print("[ERROR] trial_metadata.txt not found")
            return None, None

    print(f"Protocol: {protocol_path}")

    # Scan audio files
    print("Scanning audio files...")
    audio_path_map = {}
    part_folders = [
        "ASVspoof2021_DF_eval_part00",
        "ASVspoof2021_DF_eval_part01",
        "ASVspoof2021_DF_eval_part02",
        "ASVspoof2021_DF_eval_part03"
    ]

    for part in part_folders:
        part_dir = ASVSPOOF21_ROOT / part
        if part_dir.exists():
            for file_path in tqdm(part_dir.rglob("*.flac"), desc=f"Scanning {part}", leave=False):
                audio_path_map[file_path.stem] = str(file_path)

    print(f"Found {len(audio_path_map)} audio files")

    # Build dataframe
    # Protocol format: SPEAKER FILE_ID CODEC SOURCE ATTACK LABEL ...
    # Example: LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof ...
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_id = parts[1]  # DF_E_xxx
                label = parts[5]    # bonafide or spoof
                if file_id in audio_path_map:
                    data.append({
                        'file_id': file_id,
                        'label': label,
                        'binary_label': 0 if label == 'bonafide' else 1,
                        'audio_path': audio_path_map[file_id]
                    })

    df = pd.DataFrame(data)
    print(f"Total: {len(df)} (Real: {(df['binary_label']==0).sum()}, Fake: {(df['binary_label']==1).sum()})")

    # Sampling
    if max_samples and len(df) > max_samples:
        df_real = df[df['binary_label'] == 0].sample(n=min(max_samples//2, (df['binary_label']==0).sum()), random_state=42)
        df_fake = df[df['binary_label'] == 1].sample(n=min(max_samples//2, (df['binary_label']==1).sum()), random_state=42)
        df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  -> Sampled to {len(df)} files")

    return df['audio_path'].tolist(), df['binary_label'].values


# ============================================================================
# Model Evaluators
# ============================================================================
def evaluate_wavlm_frozen(audio_paths: List[str], labels: np.ndarray) -> Optional[Dict]:
    """WavLM Frozen Baseline 평가"""
    print("\n" + "=" * 70)
    print("Testing WavLM Frozen Baseline")
    print("=" * 70)

    model_path = TRAINED_DIR / "wavlm_frozen_baseline.pkl"
    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}")
        return None

    try:
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        print("Loading WavLM model...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        wavlm_model.eval()

        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        scaler = save_data['scaler']
        classifier = save_data['classifier']

        print("Extracting features...")
        features = []
        for path in tqdm(audio_paths, desc="WavLM"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = wavlm_model(inputs.input_values.to(device))
                feat = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                features.append(feat)
            except:
                features.append(np.zeros(1024))

        features = np.array(features)
        features_scaled = scaler.transform(features)
        predictions = classifier.predict(features_scaled)
        scores = classifier.predict_proba(features_scaled)[:, 1]

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'WavLM Frozen'
        print_metrics(metrics, 'WavLM Frozen Baseline')
        return metrics, features

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_si_detector(features: np.ndarray, labels: np.ndarray) -> Optional[Dict]:
    """Speaker-Invariant Detector 평가"""
    print("\n" + "=" * 70)
    print("Testing Speaker-Invariant Detector")
    print("=" * 70)

    model_path = TRAINED_DIR / "si_detector_n10.pkl"
    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}")
        return None

    try:
        with open(model_path, 'rb') as f:
            si_data = pickle.load(f)

        scaler = si_data['scaler']
        projection_matrix = si_data['projection_matrix']
        classifier = si_data['classifier']

        features_scaled = scaler.transform(features)
        features_projected = features_scaled @ projection_matrix
        predictions = classifier.predict(features_projected)
        scores = classifier.predict_proba(features_projected)[:, 1]

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'SI Detector (n=10)'
        print_metrics(metrics, 'Speaker-Invariant Detector (n=10)')
        return metrics

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_rawnet2(audio_paths: List[str], labels: np.ndarray) -> Optional[Dict]:
    """RawNet2 Frozen 평가"""
    print("\n" + "=" * 70)
    print("Testing RawNet2 Frozen Baseline")
    print("=" * 70)

    rawnet2_dir = MODELS_DIR / "asvspoof2021_baseline" / "2021" / "DF" / "Baseline-RawNet2"
    weights_path = rawnet2_dir / "models" / "model_DF_CCE_100_32_0.0001" / "epoch_70.pth"

    if not rawnet2_dir.exists() or not weights_path.exists():
        print(f"[SKIP] RawNet2 not found")
        print(f"  Dir: {rawnet2_dir} (exists: {rawnet2_dir.exists()})")
        print(f"  Weights: {weights_path} (exists: {weights_path.exists()})")
        return None

    try:
        if str(rawnet2_dir) not in sys.path:
            sys.path.insert(0, str(rawnet2_dir))

        from model import RawNet

        # Load config
        config_path = rawnet2_dir / "model_config_RawNet.yaml"
        if not config_path.exists():
            config_path = rawnet2_dir / "model_config_RawNet2.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', config)

        # Load model
        model = RawNet(model_config, device).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("RawNet2 loaded successfully!")

        # Inference
        nb_samp = 64000
        predictions = []
        scores = []

        for path in tqdm(audio_paths, desc="RawNet2"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) < nb_samp:
                    audio = np.tile(audio, int(nb_samp / len(audio)) + 1)[:nb_samp]
                else:
                    audio = audio[:nb_samp]

                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(audio_tensor)
                    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                spoof_prob = probs[0]
                predictions.append(1 if spoof_prob > 0.5 else 0)
                scores.append(float(spoof_prob))
            except:
                predictions.append(0)
                scores.append(0.0)

        predictions = np.array(predictions)
        scores = np.array(scores)

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'RawNet2 Frozen'
        print_metrics(metrics, 'RawNet2 Frozen Baseline')
        return metrics

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_aasist(audio_paths: List[str], labels: np.ndarray) -> Optional[Dict]:
    """AASIST Frozen 평가"""
    print("\n" + "=" * 70)
    print("Testing AASIST Frozen Baseline")
    print("=" * 70)

    aasist_dir = MODELS_DIR / "aasist"
    weights_path = aasist_dir / "models" / "weights" / "AASIST.pth"

    if not aasist_dir.exists():
        print(f"[SKIP] AASIST dir not found: {aasist_dir}")
        return None

    try:
        models_dir = aasist_dir / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from AASIST import Model as AASIST

        # Load config
        config_path = aasist_dir / "config" / "AASIST.conf"
        if not config_path.exists():
            print(f"[SKIP] AASIST config not found: {config_path}")
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = config.get('model_config', config)

        if not weights_path.exists():
            print(f"[SKIP] AASIST weights not found: {weights_path}")
            return None

        # Load model
        model = AASIST(model_config).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("AASIST loaded successfully!")

        # Inference
        nb_samp = 64600
        predictions = []
        scores = []

        for path in tqdm(audio_paths, desc="AASIST"):
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
                if len(audio) < nb_samp:
                    audio = np.tile(audio, int(nb_samp / len(audio)) + 1)[:nb_samp]
                else:
                    audio = audio[:nb_samp]

                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, output = model(audio_tensor)
                    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                spoof_prob = probs[0]
                predictions.append(1 if spoof_prob > 0.5 else 0)
                scores.append(float(spoof_prob))
            except:
                predictions.append(0)
                scores.append(0.0)

        predictions = np.array(predictions)
        scores = np.array(scores)

        metrics = evaluate_detector(labels, predictions, scores)
        metrics['model'] = 'AASIST Frozen'
        print_metrics(metrics, 'AASIST Frozen Baseline')
        return metrics

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main
# ============================================================================
def run_evaluation(dataset_name: str, audio_paths: List[str], labels: np.ndarray):
    """단일 데이터셋에 대한 전체 평가 실행"""
    print("\n" + "#" * 70)
    print(f"# Evaluating on {dataset_name}")
    print("#" * 70)

    all_results = []
    wavlm_features = None

    # 1. RawNet2
    result = evaluate_rawnet2(audio_paths, labels)
    if result:
        all_results.append(result)

    # 2. AASIST
    result = evaluate_aasist(audio_paths, labels)
    if result:
        all_results.append(result)

    # 3. WavLM Frozen
    result = evaluate_wavlm_frozen(audio_paths, labels)
    if result:
        metrics, wavlm_features = result
        all_results.append(metrics)

    # 4. SI Detector (requires WavLM features)
    if wavlm_features is not None:
        result = evaluate_si_detector(wavlm_features, labels)
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {dataset_name} Evaluation Results")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'eer']
        results_df = results_df[[c for c in cols if c in results_df.columns]]
        print(results_df.to_string(index=False))
    else:
        print("No models were successfully evaluated.")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Dataset Loader & Model Evaluation Test")
    parser.add_argument("--dataset", "-d", type=str, default="all",
                        choices=["wavefake", "asvspoof21", "all"],
                        help="Dataset to evaluate")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Max samples per dataset (for quick testing)")
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples or 'All'}")

    all_dataset_results = {}

    # WaveFake
    if args.dataset in ["wavefake", "all"]:
        audio_paths, labels = load_wavefake_dataset(args.max_samples)
        if audio_paths is not None:
            results = run_evaluation("WaveFake", audio_paths, labels)
            all_dataset_results["WaveFake"] = results

    # ASVspoof2021 DF
    if args.dataset in ["asvspoof21", "all"]:
        audio_paths, labels = load_asvspoof21_dataset(args.max_samples)
        if audio_paths is not None:
            results = run_evaluation("ASVspoof2021_DF", audio_paths, labels)
            all_dataset_results["ASVspoof2021_DF"] = results

    # Final Summary
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    for dataset_name, results in all_dataset_results.items():
        print(f"\n{dataset_name}:")
        if results:
            for r in results:
                print(f"  {r['model']}: EER={r['eer']:.2f}%, Acc={r['accuracy']*100:.2f}%")
        else:
            print("  No results")

    print("\n" + "#" * 70)
    print("# EVALUATION COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
