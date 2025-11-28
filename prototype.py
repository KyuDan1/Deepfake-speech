#%%
import torch
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def compute_gradient_features(waveform, sample_rate=16000):
    # 1. Spectrogram 변환 (Luminance 역할)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )(waveform)
    
    # Log Scaling (사람의 청각 및 이미지의 휘도와 유사하게)
    log_mel = torch.log(mel_spectrogram + 1e-9)
    
    # 2. Gradients 계산 (numpy.gradient와 유사)
    # dim=-1: Time axis (d_t), dim=-2: Frequency axis (d_f)
    grad_t = torch.diff(log_mel, dim=-1, prepend=log_mel[..., :1])
    grad_f = torch.diff(log_mel, dim=-2, prepend=log_mel[..., :1, :])
    
    # Gradient Magnitude (변화량의 세기)
    grad_mag = torch.sqrt(grad_t**2 + grad_f**2)
    
    # 3. Flatten & Statistics (벡터화)
    # 전체 픽셀을 다 쓰면 차원이 너무 크므로, 통계적 특징을 사용하거나
    # 고정 길이로 잘라서 Flatten 해야 함. 여기서는 통계적 특징 예시.
    features = torch.tensor([
        grad_mag.mean(), 
        grad_mag.std(),
        grad_mag.max(),
        grad_t.var(),
        grad_f.var()
    ])
    
    return features.numpy()

# --- 가상의 실험 시나리오 ---
# real_audios: [wave1, wave2, ...]
# fake_audios: [wave1, wave2, ...]

features_list = []
labels = [] # 0: Real, 1: Fake

# (데이터 로딩 루프가 있다고 가정)
# for audio in real_audios:
#     feat = compute_gradient_features(audio)
#     features_list.append(feat)
#     labels.append(0)
# for audio in fake_audios:
#     feat = compute_gradient_features(audio)
#     features_list.append(feat)
#     labels.append(1)

# 4. PCA 적용
# X = np.array(features_list)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# 5. 시각화 (Real vs Fake 분리 확인)
# plt.scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], c='blue', label='Real')
# plt.scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], c='red', label='Fake')
# plt.legend()
# plt.show()

# %%
