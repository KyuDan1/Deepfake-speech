# Deepfake Speech Detection - Cross-Dataset Evaluation

ASVspoof2019 LA로 학습된 모델을 다양한 데이터셋(ASVspoof2021 DF, WaveFake, In-The-Wild)에서 평가하는 Cross-Dataset 실험 프로젝트입니다.

## 1. 환경 설정

### 1.1 Conda 환경 생성

```bash
# conda 환경 생성
conda create -n deepfake_speech python=3.12 -y
conda activate deepfake_speech
```

### 1.2 필수 라이브러리 설치

```bash
# PyTorch (CUDA 11.8)
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Transformers & HuggingFace
pip install transformers==4.57.3 huggingface-hub==0.36.0 tokenizers==0.22.1 safetensors==0.7.0

# Audio Processing
pip install librosa==0.11.0 soundfile==0.13.1 audioread==3.1.0

# Data Science
pip install numpy==2.2.6 pandas==2.3.3 scipy==1.16.3
conda install scikit-learn==1.7.2 -c conda-forge -y

# Visualization
pip install matplotlib==3.10.7 seaborn==0.13.2

# Utilities
pip install pyyaml==6.0.3 tqdm==4.67.1 joblib==1.5.2

# Jupyter (optional)
pip install ipykernel jupyter-client jupyter-core kaggle
```

### 1.3 전체 환경 한 번에 설치 (간편 설치)

```bash
# 기본 conda 환경 생성
conda create -n deepfake_speech python=3.12 -y
conda activate deepfake_speech

# scikit-learn, scipy는 conda로 설치 (최적화)
conda install scikit-learn scipy joblib tqdm -c conda-forge -y

# 나머지는 pip로 설치
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa soundfile numpy pandas matplotlib seaborn pyyaml
pip install ipykernel
pip install kaggle
```

### 1.4 시스템 요구사항
- **GPU**: CUDA 11.8 지원 NVIDIA GPU (RTX 3090 이상 권장)
- **RAM**: 32GB 이상
- **저장공간**: 최소 100GB (데이터셋 포함 시)
- **wget**: 데이터셋 다운로드에 필요

```bash
# wget 설치 (Ubuntu/Debian)
sudo apt-get install wget
```

## 2. 프로젝트 구조

```
Deepfake-speech/
├── cross_dataset_eval.py      # 메인 평가 스크립트 (Jupyter Notebook 형식)
├── download_datasets.py       # 데이터셋 다운로드 스크립트
├── data/                      # 데이터셋 저장 위치
│   ├── ASVspoof2021_DF/       # ASVspoof2021 DeepFake
│   ├── WaveFake/              # WaveFake + LJSpeech + JSUT
│   └── InTheWild_hf/          # In-The-Wild 데이터셋
├── models/                    # 사전학습 모델
│   ├── aasist/                # AASIST 모델
│   ├── rawnet2-antispoofing/  # RawNet2 모델
│   └── asvspoof2021_baseline/ # ASVspoof2021 Baseline
├── cache/                     # 피처 캐시
└── results/                   # 평가 결과
```

## 3. 모델

### 3.1 사용 모델
| 모델 | 설명 | 가중치 위치 |
|------|------|-------------|
| **WavLM** | HuggingFace Transformers (자동 다운로드) | `microsoft/wavlm-large` |
| **AASIST** | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | `models/aasist/models/weights/AASIST.pth` |
| **RawNet2** | End-to-end anti-spoofing | `models/rawnet2-antispoofing/pre_trained_DF_RawNet2.pth` |

### 3.2 모델 가중치 확인
```bash
# AASIST 가중치
ls models/aasist/models/weights/
# 예상: AASIST.pth, AASIST-L.pth

# RawNet2 가중치
ls models/rawnet2-antispoofing/
# 예상: pre_trained_DF_RawNet2.pth
```

### 3.3 AASIST 모델 다운로드 (없는 경우)
```bash
# AASIST 저장소에서 가중치 다운로드
cd models/aasist/models/weights/
wget https://github.com/clovaai/aasist/releases/download/v1.0/AASIST.pth
wget https://github.com/clovaai/aasist/releases/download/v1.0/AASIST-L.pth
```

## 4. 데이터셋 구축

### 4.1 download_datasets.py 사용법

```bash
conda activate deepfake_speech

# 모든 데이터셋 다운로드 (순차)
python download_datasets.py --dataset all

# 모든 데이터셋 병렬 다운로드 (권장)
python download_datasets.py --dataset all --parallel

# 개별 데이터셋 다운로드
python download_datasets.py --dataset asvspoof2021   # ASVspoof2021 DF (~34GB)
python download_datasets.py --dataset inthewild      # In-The-Wild (~3GB)
python download_datasets.py --dataset wavefake       # WaveFake + References (~35GB)

# 커스텀 저장 경로 지정
python download_datasets.py --dataset all --data-dir /path/to/data
```

### 4.2 데이터셋 상세 정보

#### ASVspoof2021 DF (DeepFake Track)
- **출처**: [Zenodo](https://zenodo.org/records/4835108)
- **크기**: ~34GB (오디오) + Evaluation Keys
- **내용**: 611,829개 평가 샘플 (bonafide + spoof)
- **포맷**: FLAC

#### In-The-Wild
- **출처**: [HuggingFace](https://huggingface.co/datasets/mueller91/In-The-Wild)
- **크기**: ~3GB
- **내용**: 실제 환경에서 수집된 딥페이크 음성

#### WaveFake
- **출처**: [Zenodo](https://zenodo.org/records/5642694)
- **크기**: ~29GB (생성 오디오) + Reference 데이터
- **내용**:
  - 생성 오디오: MelGAN, Parallel WaveGAN, Multi-band MelGAN 등
  - Reference (Real): LJSpeech (~2.6GB), JSUT (~2GB)

### 4.3 수동 다운로드 (네트워크 문제 시)

```bash
# ASVspoof2021 DF
# Zenodo에서 직접 다운로드: https://zenodo.org/records/4835108
# 다운로드 후 data/ASVspoof2021_DF/ 에 압축 해제

# Evaluation Keys
wget https://www.asvspoof.org/resources/DF-keys-full.tar.gz
tar -xzf DF-keys-full.tar.gz -C data/ASVspoof2021_DF/keys/

# WaveFake
# Zenodo에서 직접 다운로드: https://zenodo.org/records/5642694
# 다운로드 후 data/WaveFake/ 에 압축 해제

# LJSpeech (Reference)
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2 -C data/WaveFake/

# JSUT (Reference)
wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
unzip jsut_ver1.1.zip -d data/WaveFake/
```

### 4.4 데이터 디렉토리 구조 (다운로드 완료 후)

```
data/
├── ASVspoof2021_DF/
│   ├── flac/                          # 오디오 파일
│   │   ├── DF_E_0000001.flac
│   │   └── ...
│   └── keys/
│       └── DF/
│           └── CM/
│               └── trial_metadata.txt  # 평가 프로토콜
├── WaveFake/
│   ├── generated_audio/               # 생성된 가짜 오디오
│   │   ├── ljspeech_melgan/
│   │   ├── ljspeech_parallel_wavegan/
│   │   └── ...
│   ├── LJSpeech-1.1/                  # Reference (Real)
│   │   └── wavs/
│   └── jsut_ver1.1/                   # Reference (Real)
│       └── basic5000/
└── InTheWild_hf/
    └── release_in_the_wild/
        └── ...
```

## 5. 실행 방법

### 5.1 데이터셋 준비
```bash
conda activate deepfake_speech

# 1. 데이터셋 다운로드
python download_datasets.py --dataset all --parallel
```

### 5.2 Cross-Dataset 평가 실행

`cross_dataset_eval.py`는 Jupyter Notebook 형식의 Python 스크립트입니다.

#### 방법 1: Jupyter Notebook으로 실행 (권장)
```bash
conda activate deepfake_speech

# Jupyter kernel 등록
python -m ipykernel install --user --name deepfake_speech --display-name "Deepfake Speech"

# Notebook 서버 실행
jupyter notebook cross_dataset_eval.py
```

#### 방법 2: Python 스크립트로 실행
```bash
conda activate deepfake_speech
python cross_dataset_eval.py
```

#### 방법 3: VS Code에서 실행
1. VS Code에서 `cross_dataset_eval.py` 열기
2. Python 확장 설치
3. Interpreter를 `deepfake_speech` 환경으로 선택
4. 셀 단위로 실행 (# %% 구분자)

### 5.3 경로 설정 변경
`cross_dataset_eval.py`의 `Config` 클래스에서 경로 수정:

```python
class Config:
    # ASVspoof2019 LA (학습 데이터)
    LA_ROOT = Path("/your/path/to/LA")

    # 평가 데이터셋
    DF_ROOT = Path("/your/path/to/data/ASVspoof2021_DF")
    WAVEFAKE_ROOT = Path("/your/path/to/data/WaveFake")
    INTHEWILD_ROOT = Path("/your/path/to/data/InTheWild_hf")

    # 모델
    MODELS_DIR = Path("/your/path/to/models")
```

## 6. 실험 내용

### 6.1 학습/평가 구성
- **학습 데이터**: ASVspoof2019 LA (train)
- **평가 데이터**:
  - ASVspoof2019 LA (eval)
  - ASVspoof2021 DF
  - WaveFake
  - In-The-Wild

### 6.2 평가 모델
1. **Speaker-Invariant Detector** (제안 방법)
2. **WavLM Frozen Baseline**
3. **RawNet2 Pre-trained**
4. **AASIST Pre-trained**

### 6.3 평가 지표
- EER (Equal Error Rate)
- Accuracy
- F1-Score
- ROC Curve

## 7. 문제 해결

### 7.1 CUDA Out of Memory
```python
# batch_size 줄이기
batch_size = 8  # 기본값에서 줄이기

# GPU 메모리 정리
torch.cuda.empty_cache()
```

### 7.2 다운로드 중단 시
```bash
# wget의 -c 옵션으로 이어받기 지원
python download_datasets.py --dataset asvspoof2021
```

### 7.3 파일 권한 문제
```bash
chmod +x download_datasets.py
chmod -R 755 data/
```

### 7.4 librosa 관련 오류
```bash
# soxr 백엔드 설치
pip install soxr

# 또는 resampy 사용
pip install resampy
```

## 8. 참고 자료

- [ASVspoof 2021](https://www.asvspoof.org/)
- [WaveFake Paper](https://arxiv.org/abs/2111.02813)
- [AASIST Paper](https://arxiv.org/abs/2110.01200)
- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus)

## 9. 라이선스

각 데이터셋 및 모델의 라이선스를 따릅니다:
- ASVspoof: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- WaveFake: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- AASIST: MIT License
