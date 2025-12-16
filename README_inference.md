# Deepfake Audio Detector - Inference Guide

Speaker-Invariant Deepfake Audio Detector를 사용하여 새로운 오디오 파일의 진위를 판별하는 방법을 설명합니다.

## 📋 목차

1. [모델 학습](#1-모델-학습)
2. [Python 코드에서 사용](#2-python-코드에서-사용)
3. [커맨드라인에서 사용](#3-커맨드라인에서-사용)
4. [예제 스크립트 실행](#4-예제-스크립트-실행)

---

## 1. 모델 학습

먼저 `detector_evaluation.ipynb` 노트북을 실행하여 모델을 학습하고 저장합니다.

```bash
# Jupyter Notebook 실행
jupyter notebook detector_evaluation.ipynb
```

노트북을 끝까지 실행하면:
- 다양한 `n_speaker_components` 값(1, 5, 10, 16)에 대해 모델 학습
- 최고 성능 모델이 `./models/detector_n{best_n}.pkl`로 저장됨
- 예: `./models/detector_n10.pkl`

---

## 2. Python 코드에서 사용

### 2.1 기본 사용법

```python
from inference import DeepfakeDetector

# 모델 로드
detector = DeepfakeDetector(model_path="./models/detector_n10.pkl")

# 단일 파일 예측
result = detector.predict("new_audio.wav")

# 결과 확인
print(f"Is Fake: {result['is_fake']}")           # True/False
print(f"Label: {result['label']}")               # 0: Real, 1: Fake
print(f"Confidence: {result['confidence']:.2%}") # 확신도
print(f"Real Prob: {result['probabilities']['real']:.4f}")
print(f"Fake Prob: {result['probabilities']['fake']:.4f}")
```

### 2.2 배치 예측

```python
from inference import DeepfakeDetector

detector = DeepfakeDetector(model_path="./models/detector_n10.pkl")

# 여러 파일 예측
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = detector.predict_batch(audio_files)

# 결과 출력
for audio_file, result in zip(audio_files, results):
    status = "FAKE" if result['is_fake'] else "REAL"
    conf = result['confidence']
    print(f"{audio_file}: {status} ({conf:.2%})")
```

### 2.3 결과 구조

`predict()` 함수는 다음과 같은 딕셔너리를 반환합니다:

```python
{
    'is_fake': bool,           # True if fake, False if real
    'label': int,              # 0: Real, 1: Fake
    'confidence': float,       # 예측 확신도 (0~1)
    'probabilities': {
        'real': float,         # Real일 확률 (0~1)
        'fake': float          # Fake일 확률 (0~1)
    },
    'audio_path': str          # 입력 파일 경로
}
```

---

## 3. 커맨드라인에서 사용

### 3.1 단일 파일 예측

```bash
python inference.py --audio_path /path/to/audio.wav --model_path ./models/detector_n10.pkl
```

**출력 예시:**
```
Using device: cuda
Loading model from ./models/detector_n10.pkl...
Loading WavLM model: microsoft/wavlm-large...
Model loaded successfully!

Analyzing audio: /path/to/audio.wav
============================================================

Prediction: FAKE
Confidence: 87.34%

Detailed Probabilities:
  Real: 0.1266
  Fake: 0.8734
============================================================
```

### 3.2 GPU/CPU 선택

```bash
# GPU 사용 (기본값)
python inference.py --audio_path audio.wav --model_path ./models/detector_n10.pkl

# CPU 강제 사용
python inference.py --audio_path audio.wav --model_path ./models/detector_n10.pkl --device cpu
```

---

## 4. 예제 스크립트 실행

제공된 예제 스크립트를 실행하여 바로 테스트할 수 있습니다.

```bash
python example_inference.py
```

이 스크립트는:
1. 저장된 모델 로드
2. 단일 파일 예측 예제
3. 배치 예측 예제

를 보여줍니다.

---

## 📊 성능 지표

`detector_evaluation.ipynb`에서 학습 후 각 `n_speaker_components` 값에 대한 성능을 확인할 수 있습니다:

| n_components | Test Accuracy | Precision | Recall | F1-Score |
|--------------|---------------|-----------|--------|----------|
| 1            | ??.??%        | ??.??%    | ??.??% | ??.??%   |
| 5            | ??.??%        | ??.??%    | ??.??% | ??.??%   |
| 10           | ??.??%        | ??.??%    | ??.??% | ??.??%   |
| 16           | ??.??%        | ??.??%    | ??.??% | ??.??%   |

*(노트북 실행 후 실제 값으로 업데이트)*

---

## 🔧 주요 파라미터

### n_speaker_components

Speaker Subspace에서 제거할 주요 차원 수:

- **낮은 값 (1~5)**: 화자 정보가 일부 남음, overfitting 가능성
- **중간 값 (5~10)**: 균형잡힌 성능 (권장)
- **높은 값 (16+)**: 화자 정보 거의 제거, 탐지 성능 저하 가능

---

## 💡 팁

1. **모델 선택**: `detector_evaluation.ipynb` 실행 후 최고 성능 모델 사용
2. **오디오 형식**: WAV, FLAC, MP3 등 librosa가 지원하는 모든 형식 사용 가능
3. **샘플링 레이트**: 내부에서 자동으로 16kHz로 변환됨
4. **배치 처리**: 여러 파일 처리 시 `predict_batch()` 사용 권장

---

## 🐛 문제 해결

### 1. 모델 파일을 찾을 수 없음
```
Error: Model file not found at ./models/detector_n10.pkl
```
→ `detector_evaluation.ipynb`를 먼저 실행하여 모델을 학습하고 저장하세요.

### 2. WavLM 모델 다운로드 오류
```
Error loading WavLM model...
```
→ 인터넷 연결을 확인하고 Hugging Face에 접근 가능한지 확인하세요.

### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
→ `--device cpu` 옵션을 사용하여 CPU에서 실행하세요.

---

## 📚 참고 자료

- **detector_evaluation.ipynb**: 모델 학습 및 평가
- **speaker_subspace_analysis.ipynb**: Speaker Subspace 분석 및 UMAP 시각화
- **inference.py**: Inference 스크립트 (CLI + Python API)
- **example_inference.py**: 사용 예제

---

## 📝 라이센스

이 프로젝트는 연구 목적으로 제공됩니다.
