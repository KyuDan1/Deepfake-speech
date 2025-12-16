"""
Deepfake Audio Detector - Inference Script

학습된 SpeakerInvariantDetector 모델을 로드하여 새로운 오디오 파일에 대해 추론합니다.

Usage:
    python inference.py --audio_path /path/to/audio.wav --model_path ./models/detector_n10.pkl

또는 Python 코드에서:
    from inference import DeepfakeDetector

    detector = DeepfakeDetector(model_path="./models/detector_n10.pkl")
    result = detector.predict("new_audio.wav")
    print(f"Is Synthetic: {result['is_fake']}")
    print(f"Confidence: {result['confidence']:.2%}")
"""

import torch
import numpy as np
import librosa
import pickle
import argparse
from pathlib import Path
from typing import Dict, Union, Optional
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class DeepfakeDetector:
    """
    학습된 Speaker-Invariant Deepfake Detector를 로드하여 추론하는 클래스
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Args:
            model_path: 저장된 모델 파일 경로 (.pkl)
            device: 'cuda' 또는 'cpu' (None이면 자동 선택)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 모델 로드
        print(f"Loading model from {model_path}...")
        self._load_model(model_path)

        print("Model loaded successfully!")

    def _load_model(self, model_path: str):
        """저장된 모델과 전처리 파라미터를 로드"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        # WavLM 모델 로드
        model_name = saved_data.get('model_name', 'microsoft/wavlm-large')
        print(f"Loading WavLM model: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 저장된 파라미터 로드
        self.scaler = saved_data['scaler']
        self.projection_matrix = saved_data['projection_matrix']
        self.classifier = saved_data['classifier']
        self.n_speaker_components = saved_data['n_speaker_components']

        print(f"  n_speaker_components: {self.n_speaker_components}")
        print(f"  Feature dimension: {self.projection_matrix.shape[0]}")

    def _extract_feature(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """오디오 파일에서 WavLM feature 추출"""
        try:
            # 오디오 로드
            audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)

            # WavLM feature extraction
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)

            # Mean pooling
            pooled_features = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return pooled_features

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        새로운 오디오 파일에 대해 Deepfake 여부를 예측합니다.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            dict: {
                'is_fake': bool,           # True if fake, False if real
                'label': int,              # 0: Real, 1: Fake
                'confidence': float,       # 예측 확신도 (0~1)
                'probabilities': dict,     # {'real': float, 'fake': float}
                'audio_path': str          # 입력 파일 경로
            }
        """
        # 1. Feature extraction
        feat = self._extract_feature(audio_path)
        if feat is None:
            return {
                'error': 'Feature extraction failed',
                'audio_path': str(audio_path)
            }

        # 2. Scaling
        feat = feat.reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        # 3. Project (Remove Speaker Info)
        feat_projected = feat_scaled @ self.projection_matrix

        # 4. Predict
        prob = self.classifier.predict_proba(feat_projected)[0]
        pred_label = self.classifier.predict(feat_projected)[0]

        # 결과 정리
        result = {
            'is_fake': bool(pred_label == 1),
            'label': int(pred_label),
            'confidence': float(prob[pred_label]),  # 예측된 클래스에 대한 확신도
            'probabilities': {
                'real': float(prob[0]),
                'fake': float(prob[1])
            },
            'audio_path': str(audio_path)
        }

        return result

    def predict_batch(self, audio_paths: list) -> list:
        """
        여러 오디오 파일을 배치로 예측합니다.

        Args:
            audio_paths: 오디오 파일 경로 리스트

        Returns:
            list of dict: 각 파일에 대한 예측 결과 리스트
        """
        results = []
        for path in audio_paths:
            result = self.predict(path)
            results.append(result)
        return results


def save_trained_model(
    detector,
    save_path: str,
    n_speaker_components: int,
    model_name: str = "microsoft/wavlm-large"
):
    """
    학습된 SpeakerInvariantDetector를 저장합니다.

    Args:
        detector: 학습된 SpeakerInvariantDetector 인스턴스
        save_path: 저장할 파일 경로 (.pkl)
        n_speaker_components: 사용한 speaker component 개수
        model_name: 사용한 WavLM 모델 이름
    """
    if not detector.is_fitted:
        raise ValueError("Detector must be fitted before saving!")

    save_data = {
        'scaler': detector.scaler,
        'projection_matrix': detector.projection_matrix,
        'classifier': detector.classifier,
        'n_speaker_components': n_speaker_components,
        'model_name': model_name,
        'pca': detector.pca  # 추가 정보 (분석용)
    }

    # 디렉토리가 없으면 생성
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Model saved to {save_path}")
    print(f"  n_speaker_components: {n_speaker_components}")
    print(f"  Model name: {model_name}")


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description='Deepfake Audio Detector - Inference')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to the audio file to analyze')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pkl)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified.')

    args = parser.parse_args()

    # 오디오 파일 존재 확인
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found at {args.audio_path}")
        return

    # 모델 파일 존재 확인
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Detector 로드
    detector = DeepfakeDetector(model_path=args.model_path, device=args.device)

    # 예측
    print(f"\nAnalyzing audio: {args.audio_path}")
    print("=" * 60)

    result = detector.predict(args.audio_path)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    # 결과 출력
    print(f"\nPrediction: {'FAKE' if result['is_fake'] else 'REAL'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nDetailed Probabilities:")
    print(f"  Real: {result['probabilities']['real']:.4f}")
    print(f"  Fake: {result['probabilities']['fake']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
