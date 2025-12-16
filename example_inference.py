"""
간단한 Inference 예제 스크립트

학습된 Deepfake Detector를 사용하여 새로운 오디오 파일의 진위를 판별합니다.
"""

from inference import DeepfakeDetector
from pathlib import Path

def main():
    # 1. 모델 로드
    model_path = "./models/detector_n10.pkl"  # 학습된 모델 경로

    if not Path(model_path).exists():
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please run detector_evaluation.ipynb first to train and save a model.")
        return

    print("Loading Deepfake Detector...")
    detector = DeepfakeDetector(model_path=model_path)
    print("✅ Model loaded successfully!\n")

    # 2. 단일 파일 예측 예제
    print("="*80)
    print("Example 1: Single Audio File Prediction")
    print("="*80)

    # 여기에 테스트할 오디오 파일 경로를 입력하세요
    test_audio = "my_raw_audio/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"

    if Path(test_audio).exists():
        result = detector.predict(test_audio)

        print(f"\nAudio File: {test_audio}")
        print(f"Prediction: {'🎭 FAKE (Synthetic)' if result['is_fake'] else '✅ REAL (Genuine)'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Detailed Probabilities:")
        print(f"  Real: {result['probabilities']['real']:.4f}")
        print(f"  Fake: {result['probabilities']['fake']:.4f}")
    else:
        print(f"❌ Test file not found: {test_audio}")

    # 3. 배치 예측 예제
    print(f"\n{'='*80}")
    print("Example 2: Batch Prediction (Multiple Files)")
    print("="*80)

    # 여러 파일 경로 리스트
    audio_files = [
        "my_raw_audio/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
        "my_raw_audio/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac",
        "generated_results/speaker_libri_transcript_1089-134686-0000.wav",
    ]

    # 존재하는 파일만 필터링
    existing_files = [f for f in audio_files if Path(f).exists()]

    if existing_files:
        print(f"\nProcessing {len(existing_files)} files...\n")
        results = detector.predict_batch(existing_files)

        for audio_file, result in zip(existing_files, results):
            filename = Path(audio_file).name
            prediction = "FAKE" if result['is_fake'] else "REAL"
            confidence = result['confidence']

            print(f"📄 {filename:50s} -> {prediction:6s} ({confidence:6.2%})")
    else:
        print("❌ No valid audio files found for batch prediction")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
