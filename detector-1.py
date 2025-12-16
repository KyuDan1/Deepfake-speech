import torch
import numpy as np
import librosa
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

class SpeakerInvariantDetector:
    def __init__(self, model_name="microsoft/wavlm-large", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading WavLM model ({self.device})...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 투영 행렬 (Speaker info 제거용)과 분류기
        self.projection_matrix = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, solver='liblinear')
        self.pca = None
        self.is_fitted = False

    def _extract_feature(self, audio_input):
        """내부용: Wav 파일 경로 또는 numpy array에서 WavLM feature 추출"""
        try:
            # 경로인 경우 로드, 아니면 그대로 사용
            if isinstance(audio_input, (str, Path)):
                audio, _ = librosa.load(str(audio_input), sr=16000, mono=True)
            else:
                audio = audio_input # 이미 numpy array라고 가정

            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)
            
            # (Batch, Time, Dim) -> Mean Pooling -> (Dim,)
            pooled_features = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return pooled_features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def fit(self, audio_paths, labels, speaker_ids, n_speaker_components=10):
        """
        모델 학습 함수 (Projection Matrix 계산 + Classifier 학습)
        
        Args:
            audio_paths: 오디오 파일 경로 리스트
            labels: 0 (Real), 1 (Fake) 등의 레이블 리스트
            speaker_ids: 각 오디오의 화자 ID 리스트 (Speaker Subspace 계산용)
            n_speaker_components: 제거할 화자 정보 차원 수 (PC 개수)
        """
        print("Extracting features for training...")
        X_raw = []
        y = []
        spk_map = {} # {speaker_id: [indices]}
        
        # 1. Feature Extraction
        for idx, (path, label, spk) in enumerate(zip(audio_paths, labels, speaker_ids)):
            feat = self._extract_feature(path)
            if feat is not None:
                X_raw.append(feat)
                y.append(label)
                if spk not in spk_map: spk_map[spk] = []
                spk_map[spk].append(len(X_raw) - 1)
        
        X_raw = np.array(X_raw)
        y = np.array(y)
        
        # 2. Scaling
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # 3. Compute Speaker Subspace (PCA on Speaker Centroids)
        print(f"Computing Speaker Subspace (removing top {n_speaker_components} components)...")
        speaker_centroids = []
        for spk, indices in spk_map.items():
            # 해당 화자의 모든 발화 평균 계산
            centroid = np.mean(X_scaled[indices], axis=0)
            speaker_centroids.append(centroid)
        
        speaker_centroids = np.array(speaker_centroids)
        
        # 화자 평균들에 대해 PCA 수행하여 주요 "화자 방향(Basis)" 찾기
        self.pca = PCA(n_components=n_speaker_components)
        self.pca.fit(speaker_centroids)
        
        # Orthogonal Projection Matrix 생성: P_perp = I - U @ U.T
        # U: Speaker Basis Vectors (n_features, n_components)
        U = self.pca.components_.T 
        I = np.eye(U.shape[0])
        self.projection_matrix = I - (U @ U.T)
        
        # 4. Project Features (Remove Speaker Info)
        # X_proj = X @ P_perp
        X_projected = X_scaled @ self.projection_matrix
        
        # 5. Train Simple Classifier (Logistic Regression)
        print("Training Decision Boundary (Logistic Regression)...")
        self.classifier.fit(X_projected, y)
        self.is_fitted = True
        
        # 학습 결과 요약
        acc = self.classifier.score(X_projected, y)
        print(f"Training Complete. Accuracy on Train Set: {acc:.4f}")

    def predict(self, audio_path):
        """
        Inference 함수: Wav -> Feature -> Scale -> Project -> Predict
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # 1. Extract
        feat = self._extract_feature(audio_path)
        if feat is None: return None
        
        # 2. Scale
        # (1, Dim) 형태로 변환
        feat = feat.reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        
        # 3. Project (Remove Speaker Info)
        # 수학적으로: x_new = x_old @ (I - UU^T)
        feat_projected = feat_scaled @ self.projection_matrix
        
        # 4. Predict
        prob = self.classifier.predict_proba(feat_projected)[0]
        pred_label = self.classifier.predict(feat_projected)[0]
        
        return {
            "label": pred_label,          # 예측 클래스
            "probability": prob,          # [Prob_Class0, Prob_Class1]
            "feature_vector": feat_projected # 투영된 벡터 (시각화용)
        }

# --- 사용 예시 ---

# 1. Detector 인스턴스 생성
detector = SpeakerInvariantDetector()

# 2. 데이터 준비 (사용자의 기존 dataframe_10 활용 예시)
#    X: 경로, y: 레이블(0:Real, 1:Fake), groups: 화자ID
train_paths = []
train_labels = []
train_speakers = []

# (기존 코드의 루프를 사용하여 리스트 구성)
# 예시: dataframe_10이 있다고 가정
print("Preparing Dataset...")
for row in dataframe_10.to_dict('records'):
    # Real Data
    real_path = Path(row['audio_path'])
    if real_path.exists():
        train_paths.append(str(real_path))
        train_labels.append(0) # 0 for Real
        train_speakers.append(row['speaker_id'])
    
    # Fake Data (Gen 2)
    base_name = real_path.stem
    fake_path = Path(f"generated_results/speaker_libri_transcript_{base_name}.wav")
    if fake_path.exists():
        train_paths.append(str(fake_path))
        train_labels.append(1) # 1 for Fake
        train_speakers.append(row['speaker_id']) # 같은 화자 ID를 가짐 (이걸 제거하는게 목표)

# 3. 학습 (이 과정에서 Speaker Subspace가 계산되고 제거됩니다)
#    n_speaker_components=5 ~ 10 정도로 설정하여 주요 화자 특성을 제거
if len(train_paths) > 0:
    detector.fit(train_paths, train_labels, train_speakers, n_speaker_components=10)

    # 4. 추론 (새로운 파일 테스트)
    test_file = train_paths[0] # 테스트용으로 하나 뽑음
    result = detector.predict(test_file)
    
    print("\n--- Inference Result ---")
    print(f"File: {test_file}")
    print(f"Prediction: {'Fake' if result['label'] == 1 else 'Real'}")
    print(f"Confidence (Fake): {result['probability'][1]:.4f}")
else:
    print("No training data found.")