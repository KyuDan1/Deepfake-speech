"""
MegaTTS3 배치 생성 스크립트
LibriSpeech 전사 문장을 MegaTTS3로 음성 합성
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# MegaTTS3 경로 추가
megatts3_root = '/mnt/ddn/kyudan/MegaTTS3'
if megatts3_root not in sys.path:
    sys.path.insert(0, megatts3_root)

import torch
from tts.infer_cli import MegaTTS3DiTInfer
from tts.utils.audio_utils.io import save_wav

# 설정
OUTPUT_DIR = "generated_results"
CHECKPOINT_ROOT = os.path.join(megatts3_root, 'checkpoints')

# 결과 저장 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resolve_path(path_str):
    """경로를 절대 경로로 변환"""
    return str(Path(path_str).resolve())

def initialize_megatts3(device='cuda', precision=torch.float16):
    """
    MegaTTS3 모델 초기화

    Args:
        device: 'cuda' 또는 'cpu'
        precision: torch.float16 (GPU) 또는 torch.float32 (CPU)

    Returns:
        MegaTTS3DiTInfer 인스턴스
    """
    print("MegaTTS3 모델 로딩 중...")
    start_time = time.time()

    infer_ins = MegaTTS3DiTInfer(
        device=device,
        ckpt_root=CHECKPOINT_ROOT,
        precision=precision
    )

    elapsed = time.time() - start_time
    print(f"✅ 모델 로딩 완료 (소요시간: {elapsed:.2f}초)")

    return infer_ins

def preprocess_prompt(infer_ins, prompt_wav_path):
    """
    프롬프트 음성 전처리

    Args:
        infer_ins: MegaTTS3DiTInfer 인스턴스
        prompt_wav_path: 프롬프트 음성 파일 경로

    Returns:
        resource_context: 전처리된 리소스 컨텍스트
    """
    # 음성 파일 읽기
    with open(prompt_wav_path, 'rb') as file:
        audio_bytes = file.read()

    # latent 파일 경로 (WaveVAE 인코더가 없으므로 필수)
    # .wav, .flac 등 확장자를 .npy로 변경
    prompt_path = Path(prompt_wav_path)
    latent_file_path = str(prompt_path.with_suffix('.npy'))

    # latent 파일이 없는 경우 None으로 설정 (에러 발생 가능)
    if not os.path.exists(latent_file_path):
        print(f"⚠️  경고: latent 파일이 없습니다: {latent_file_path}")
        print(f"   프롬프트 음성에 대한 .npy 파일이 필요합니다.")
        latent_file_path = None

    # 전처리 수행
    resource_context = infer_ins.preprocess(
        audio_bytes,
        latent_file=latent_file_path,
        topk_dur=1
    )

    return resource_context

def generate_audio(infer_ins, resource_context, text, filename,
                   time_step=32, p_w=2.0, t_w=3.0):
    """
    MegaTTS3로 음성을 생성하고 파일로 저장

    Args:
        infer_ins: MegaTTS3DiTInfer 인스턴스
        resource_context: 전처리된 프롬프트 컨텍스트
        text: 합성할 텍스트
        filename: 저장할 파일 이름
        time_step: Diffusion Transformer 추론 스텝 (기본: 32)
        p_w: Intelligibility Weight (기본: 2.0)
        t_w: Similarity Weight (기본: 3.0)

    Returns:
        file_path: 저장된 파일 경로
    """
    try:
        start_time = time.time()
        print(f"생성 중... Text: {text[:50]}...")

        # 음성 생성
        wav_bytes = infer_ins.forward(
            resource_context,
            text,
            time_step=time_step,
            p_w=p_w,
            t_w=t_w
        )

        # 파일 저장
        file_path = os.path.join(OUTPUT_DIR, filename)
        save_wav(wav_bytes, file_path)

        elapsed = time.time() - start_time
        print(f"✅ 저장 완료: {file_path} (소요시간: {elapsed:.2f}초)")

        return file_path

    except Exception as e:
        print(f"❌ 생성 실패: {e}")
        return None

def batch_generate_with_cache(infer_ins, input_data, time_step=32, p_w=2.0, t_w=3.0):
    """
    동일한 화자의 음성은 프롬프트를 캐싱하여 배치 생성

    Args:
        infer_ins: MegaTTS3DiTInfer 인스턴스
        input_data: 입력 데이터 리스트 (각 항목은 text, filename, prompt_wav_path 포함)
        time_step: Diffusion 스텝 수
        p_w: Intelligibility Weight
        t_w: Similarity Weight
    """
    # 프롬프트 캐시 (같은 프롬프트를 재사용)
    prompt_cache = {}

    for idx, item in enumerate(input_data, start=1):
        print(f"\n[{idx:02d}/{len(input_data):02d}]")
        print(f"  텍스트: {item['text'][:70]}...")
        print(f"  프롬프트 음성: {item['prompt_wav_path']}")

        # 프롬프트 캐시 확인
        prompt_wav_path = item['prompt_wav_path']
        if prompt_wav_path not in prompt_cache:
            print(f"  프롬프트 전처리 중...")
            try:
                resource_context = preprocess_prompt(infer_ins, prompt_wav_path)
                prompt_cache[prompt_wav_path] = resource_context
            except Exception as e:
                print(f"  ❌ 프롬프트 전처리 실패: {e}")
                continue
        else:
            print(f"  캐시된 프롬프트 사용")
            resource_context = prompt_cache[prompt_wav_path]

        # 음성 생성
        generate_audio(
            infer_ins,
            resource_context,
            text=item["text"],
            filename=item["filename"],
            time_step=time_step,
            p_w=p_w,
            t_w=t_w
        )

# 메인 실행 코드는 별도 스크립트나 노트북에서 실행
if __name__ == "__main__":
    print("이 스크립트는 모듈로 임포트하여 사용하세요.")
    print("노트북에서 사용 예시:")
    print("""
    from megatts3_batch_generation import initialize_megatts3, batch_generate_with_cache

    # 모델 초기화
    infer_ins = initialize_megatts3(device='cuda', precision=torch.float16)

    # 입력 데이터 준비
    input_data = [
        {
            'text': '생성할 텍스트',
            'filename': 'output.wav',
            'prompt_wav_path': '/path/to/prompt.wav'
        },
        # ... 더 많은 항목
    ]

    # 배치 생성
    batch_generate_with_cache(infer_ins, input_data, time_step=32, p_w=2.0, t_w=3.0)
    """)
