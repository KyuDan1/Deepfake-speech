"""
MegaTTS3 WaveVAE를 사용하여 LibriSpeech 오디오 파일의 latent representation 생성

이 스크립트는 모든 .flac 파일에 대해 .npy latent 파일을 생성합니다.
"""

import os
import sys
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm

# MegaTTS3 경로 추가
megatts3_root = '/mnt/ddn/kyudan/MegaTTS3'
if megatts3_root not in sys.path:
    sys.path.insert(0, megatts3_root)

from tts.utils.commons.hparams import set_hparams

def load_wavvae_encoder(ckpt_root='/mnt/ddn/kyudan/MegaTTS3/checkpoints', device='cuda'):
    """
    WaveVAE 인코더 로드

    Args:
        ckpt_root: 체크포인트 루트 디렉토리
        device: 'cuda' 또는 'cpu'

    Returns:
        wavvae: WaveVAE 모델 인스턴스
        sr: 샘플링 레이트
    """
    print("=" * 80)
    print("WaveVAE 인코더 로딩 중...")
    print("=" * 80)

    wavvae_exp_name = os.path.join(ckpt_root, 'wavvae')

    # WaveVAE config 로드
    hp_wavvae = set_hparams(f'{wavvae_exp_name}/config.yaml', global_hparams=False)

    # WaveVAE 모델 생성
    from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
    wavvae = WavVAE_V3(hparams=hp_wavvae)

    # 체크포인트 로드
    encoder_ckpt = f'{wavvae_exp_name}/model_only_last.ckpt'
    decoder_ckpt = f'{wavvae_exp_name}/decoder.ckpt'

    if os.path.exists(encoder_ckpt):
        print(f"✅ 인코더 포함 체크포인트 로드: {encoder_ckpt}")
        ckpt = torch.load(encoder_ckpt, map_location='cpu')
        wavvae.load_state_dict(ckpt['state_dict']['model_gen'], strict=True)
        has_encoder = True
    elif os.path.exists(decoder_ckpt):
        print(f"⚠️  Decoder-only 체크포인트 로드: {decoder_ckpt}")
        ckpt = torch.load(decoder_ckpt, map_location='cpu')
        try:
            wavvae.load_state_dict(ckpt['state_dict']['model_gen'], strict=False)
            # 인코더가 로드되었는지 확인
            has_encoder = hasattr(wavvae, 'encoder') and wavvae.encoder is not None
            if has_encoder:
                print(f"✅ 인코더 파라미터 확인됨")
            else:
                print(f"❌ 인코더가 없습니다")
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            raise
    else:
        raise FileNotFoundError(f"WaveVAE 체크포인트를 찾을 수 없습니다: {wavvae_exp_name}")

    wavvae.eval()
    wavvae.to(device)

    sr = 24000  # MegaTTS3 샘플링 레이트
    print(f"✅ WaveVAE 로드 완료 (샘플링 레이트: {sr}Hz)")
    print("=" * 80)

    return wavvae, sr

def encode_audio_to_latent(wavvae, audio_path, sr=24000, device='cuda'):
    """
    오디오 파일을 latent representation으로 인코딩

    Args:
        wavvae: WaveVAE 모델
        audio_path: 오디오 파일 경로
        sr: 샘플링 레이트
        device: 디바이스

    Returns:
        latent: numpy array (1, latent_dim, frames)
    """
    # 오디오 로드
    wav, _ = librosa.load(audio_path, sr=sr)

    # Padding (MegaTTS3 전처리와 동일)
    ws = 480  # win_size
    if len(wav) % ws < ws - 1:
        wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
    wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)

    # Tensor 변환
    wav_tensor = torch.FloatTensor(wav)[None].to(device)  # (1, samples)

    # 인코딩
    with torch.no_grad():
        latent = wavvae.encode_latent(wav_tensor)  # (1, frames, latent_dim)
        latent = latent.permute(0, 2, 1)  # (1, latent_dim, frames)

    # CPU로 이동 및 numpy 변환
    latent_np = latent.cpu().numpy()

    return latent_np

def find_audio_files(base_dir, extensions=['.flac', '.wav']):
    """
    디렉토리에서 오디오 파일 찾기
    """
    audio_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def generate_latents_batch(wavvae, audio_files, sr=24000, device='cuda', skip_existing=True):
    """
    배치로 latent 파일 생성

    Args:
        wavvae: WaveVAE 모델
        audio_files: 오디오 파일 경로 리스트
        sr: 샘플링 레이트
        device: 디바이스
        skip_existing: 이미 존재하는 .npy 파일 건너뛰기
    """
    created_count = 0
    skipped_count = 0
    failed_count = 0

    for audio_path in tqdm(audio_files, desc="Encoding audio files"):
        # .npy 파일 경로
        npy_path = audio_path.replace('.flac', '.npy').replace('.wav', '.npy')

        # 이미 존재하면 건너뛰기
        if skip_existing and os.path.exists(npy_path):
            skipped_count += 1
            continue

        try:
            # 인코딩
            latent = encode_audio_to_latent(wavvae, audio_path, sr=sr, device=device)

            # 저장
            np.save(npy_path, latent)
            created_count += 1

        except Exception as e:
            print(f"\n❌ 실패: {audio_path}")
            print(f"   에러: {e}")
            failed_count += 1

    print("\n" + "=" * 80)
    print("Latent 생성 완료!")
    print("=" * 80)
    print(f"  ✅ 생성: {created_count}개")
    print(f"  ⏭️  건너뜀: {skipped_count}개")
    print(f"  ❌ 실패: {failed_count}개")
    print("=" * 80)

if __name__ == "__main__":
    # 설정
    LIBRISPEECH_DIR = "/mnt/ddn/kyudan/Deepfake-speech/my_raw_audio/LibriSpeech"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n디바이스: {DEVICE}")
    print(f"LibriSpeech 디렉토리: {LIBRISPEECH_DIR}\n")

    # WaveVAE 로드
    try:
        wavvae, sr = load_wavvae_encoder(device=DEVICE)
    except Exception as e:
        print(f"\n❌ WaveVAE 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. MegaTTS3 전체 체크포인트 다운로드 (인코더 포함)")
        print("   - 링크: https://huggingface.co/ByteDance/MegaTTS3")
        print("2. 또는 다른 TTS 모델 사용")
        sys.exit(1)

    # 오디오 파일 찾기
    print("\nLibriSpeech 파일 스캔 중...")
    audio_files = find_audio_files(LIBRISPEECH_DIR)
    print(f"총 {len(audio_files)}개의 오디오 파일 발견\n")

    if len(audio_files) == 0:
        print("❌ 오디오 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # Latent 생성
    generate_latents_batch(wavvae, audio_files, sr=sr, device=DEVICE, skip_existing=True)

    print("\n모든 작업이 완료되었습니다!")
