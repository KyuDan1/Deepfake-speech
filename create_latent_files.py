"""
LibriSpeech 오디오 파일에 대한 latent 파일(.npy) 생성 스크립트

MegaTTS3는 decoder-only 모드에서 실행되므로,
각 프롬프트 음성 파일에 대한 latent representation이 필요합니다.

해결 방법:
1. WaveVAE 인코더를 사용하여 직접 생성
2. MegaTTS3 데모에서 제공하는 방법 사용
3. 다른 TTS 모델 사용 (encoder가 있는 버전)
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

def check_encoder_availability():
    """
    WaveVAE 인코더 사용 가능 여부 확인
    """
    encoder_ckpt = os.path.join(megatts3_root, 'checkpoints/wavvae/model_only_last.ckpt')
    decoder_only = os.path.join(megatts3_root, 'checkpoints/wavvae/decoder.ckpt')

    if os.path.exists(encoder_ckpt):
        print("✅ WaveVAE 인코더 사용 가능 (model_only_last.ckpt)")
        return True
    elif os.path.exists(decoder_only):
        print("❌ WaveVAE decoder만 사용 가능 (encoder 없음)")
        print(f"   현재 파일: {decoder_only}")
        print(f"   필요 파일: {encoder_ckpt}")
        return False
    else:
        print("❌ WaveVAE 체크포인트를 찾을 수 없습니다.")
        return False

def create_dummy_latents_from_audio(audio_paths, output_dir=None):
    """
    대안: 오디오 길이에 맞는 더미 latent 생성

    경고: 이 방법은 임시 방편이며, 실제 음성 클로닝 품질이 매우 낮을 수 있습니다.
    실제 사용을 위해서는 WaveVAE 인코더가 필요합니다.
    """
    print("\n⚠️  경고: WaveVAE 인코더가 없으므로 더미 latent를 생성합니다.")
    print("   이는 테스트용이며, 실제 음성 품질은 보장되지 않습니다.\n")

    sr = 24000  # MegaTTS3 샘플링 레이트
    vae_stride = 4
    hop_size = 4

    created_count = 0
    for audio_path in tqdm(audio_paths, desc="Creating dummy latents"):
        # .npy 파일 경로
        npy_path = audio_path.replace('.flac', '.npy').replace('.wav', '.npy')

        # 이미 존재하면 건너뛰기
        if os.path.exists(npy_path):
            continue

        try:
            # 오디오 로드
            wav, _ = librosa.load(audio_path, sr=sr)

            # latent 크기 계산 (MegaTTS3 기준)
            # latent shape: [1, latent_dim, frames]
            # frames = audio_samples / (hop_size * vae_stride)
            latent_dim = 128  # WaveVAE latent dimension
            frames = len(wav) // (hop_size * vae_stride)

            # 더미 latent 생성 (작은 랜덤 값)
            dummy_latent = np.random.randn(1, latent_dim, frames).astype(np.float32) * 0.01

            # .npy 파일로 저장
            np.save(npy_path, dummy_latent)
            created_count += 1

        except Exception as e:
            print(f"  ❌ 실패: {audio_path} - {e}")

    print(f"\n✅ 총 {created_count}개의 더미 latent 파일 생성 완료")
    print(f"   주의: 이는 실제 WaveVAE 인코더로 생성한 latent가 아닙니다!")

def find_librispeech_files(base_dir):
    """
    LibriSpeech 디렉토리에서 모든 .flac 파일 찾기
    """
    flac_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_files.append(os.path.join(root, file))
    return flac_files

if __name__ == "__main__":
    print("=" * 80)
    print("LibriSpeech Latent 파일 생성 스크립트")
    print("=" * 80)

    # 인코더 확인
    has_encoder = check_encoder_availability()

    if not has_encoder:
        print("\n해결 방법:")
        print("1. MegaTTS3 전체 체크포인트 다운로드 (인코더 포함 버전)")
        print("   - 링크: https://huggingface.co/ByteDance/MegaTTS3")
        print("2. 또는 이 스크립트로 더미 latent 생성 (품질 저하)")
        print("3. 또는 다른 TTS 모델 사용 (예: CosyVoice, F5-TTS 등)")

        response = input("\n더미 latent를 생성하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("종료합니다.")
            sys.exit(0)

    # LibriSpeech 파일 찾기
    librispeech_dir = "/mnt/ddn/kyudan/Deepfake-speech/my_raw_audio/LibriSpeech"

    print(f"\nLibriSpeech 디렉토리 스캔 중: {librispeech_dir}")
    audio_files = find_librispeech_files(librispeech_dir)
    print(f"총 {len(audio_files)}개의 .flac 파일 발견")

    if len(audio_files) == 0:
        print("❌ 오디오 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # Latent 생성
    if not has_encoder:
        create_dummy_latents_from_audio(audio_files)
    else:
        print("✅ 인코더를 사용한 실제 latent 생성 기능은 구현 예정")
        print("   현재는 더미 latent 생성만 지원합니다.")

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)
