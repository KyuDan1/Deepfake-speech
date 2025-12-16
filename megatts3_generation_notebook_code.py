"""
노트북에서 바로 실행할 수 있는 MegaTTS3 배치 생성 코드
기존 vLLM 서버 방식을 MegaTTS3 직접 호출로 대체
"""

# ============================================================================
# 1. 필수 라이브러리 임포트 및 MegaTTS3 설정
# ============================================================================
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

# ============================================================================
# 2. 설정
# ============================================================================
OUTPUT_DIR = "generated_results"
CHECKPOINT_ROOT = os.path.join(megatts3_root, 'checkpoints')

# 결과 저장 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MegaTTS3 생성 파라미터
TIME_STEP = 32      # Diffusion 스텝 수 (높을수록 품질 향상, 시간 증가)
P_W = 2.0          # Intelligibility Weight (명료도)
T_W = 3.0          # Similarity Weight (유사도)

def resolve_path(path_str):
    """경로를 절대 경로로 변환"""
    return str(Path(path_str).resolve())

# ============================================================================
# 3. MegaTTS3 모델 초기화
# ============================================================================
print("=" * 80)
print("MegaTTS3 모델 로딩 중...")
print("=" * 80)

start_time = time.time()

infer_ins = MegaTTS3DiTInfer(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ckpt_root=CHECKPOINT_ROOT,
    precision=torch.float16 if torch.cuda.is_available() else torch.float32
)

elapsed = time.time() - start_time
print(f"\n✅ 모델 로딩 완료 (소요시간: {elapsed:.2f}초)\n")

# ============================================================================
# 4. 프롬프트 전처리 함수
# ============================================================================
def preprocess_prompt(prompt_wav_path):
    """
    프롬프트 음성 전처리

    Args:
        prompt_wav_path: 프롬프트 음성 파일 경로

    Returns:
        resource_context: 전처리된 리소스 컨텍스트
    """
    # 음성 파일 읽기
    with open(prompt_wav_path, 'rb') as file:
        audio_bytes = file.read()

    # latent 파일 경로 (.npy 파일)
    # .wav, .flac 등 확장자를 .npy로 변경
    prompt_path = Path(prompt_wav_path)
    latent_file_path = str(prompt_path.with_suffix('.npy'))

    # latent 파일이 없는 경우 경고
    if not os.path.exists(latent_file_path):
        print(f"⚠️  경고: latent 파일이 없습니다: {latent_file_path}")
        print(f"   WaveVAE 인코더 없이 실행하려면 .npy 파일이 필요합니다.")
        latent_file_path = None

    # 전처리 수행
    resource_context = infer_ins.preprocess(
        audio_bytes,
        latent_file=latent_file_path,
        topk_dur=1
    )

    return resource_context

# ============================================================================
# 5. 음성 생성 및 저장 함수
# ============================================================================
def generate_and_save_audio(resource_context, text, filename,
                           time_step=TIME_STEP, p_w=P_W, t_w=T_W):
    """
    MegaTTS3로 음성을 생성하고 파일로 저장

    Args:
        resource_context: 전처리된 프롬프트 컨텍스트
        text: 합성할 텍스트
        filename: 저장할 파일 이름
        time_step: Diffusion Transformer 추론 스텝 (기본: 32)
        p_w: Intelligibility Weight (기본: 2.0)
        t_w: Similarity Weight (기본: 3.0)

    Returns:
        file_path: 저장된 파일 경로 또는 None (실패 시)
    """
    try:
        start_time = time.time()

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
        print(f"    ✅ 저장 완료: {file_path} (소요시간: {elapsed:.2f}초)")

        return file_path

    except Exception as e:
        print(f"    ❌ 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 6. 입력 데이터 준비
# ============================================================================
# dataframe_10과 dataframe이 이미 정의되어 있다고 가정
# 아래 코드를 실제 노트북 환경에 맞게 수정하세요

input_rows = dataframe_10[['transcript', 'audio_path', 'speaker_id']].to_dict('records')

# 각 speaker_id당 하나의 프롬프트 음성/텍스트를 선택
speaker_prompt_lookup = {}
for row in dataframe[['speaker_id', 'transcript', 'audio_path']].to_dict('records'):
    if row['speaker_id'] not in speaker_prompt_lookup:
        speaker_prompt_lookup[row['speaker_id']] = {
            'prompt_text': row['transcript'],
            'prompt_wav_path': resolve_path(row['audio_path'])
        }

input_data = []
for idx, row in enumerate(input_rows):
    absolute_audio_path = resolve_path(row['audio_path'])
    base_name = Path(absolute_audio_path).stem
    prompt = speaker_prompt_lookup.get(row['speaker_id'], {
        'prompt_text': row['transcript'],
        'prompt_wav_path': absolute_audio_path
    })
    input_data.append({
        "text": row['transcript'],
        "filename": f"speaker_libri_transcript_{base_name}.wav",
        "prompt_wav_path": prompt['prompt_wav_path'],
    })

# ============================================================================
# 7. 메인 실행 루프 (프롬프트 캐싱 포함)
# ============================================================================
print("=" * 80)
print(f"총 {len(input_data)}개의 오디오 생성을 시작합니다.")
print("각 샘플은 동일한 speaker_id의 실음성을 프롬프트로 사용합니다.")
print("=" * 80)

# 프롬프트 캐시 (같은 프롬프트 음성을 재사용하여 성능 향상)
prompt_cache = {}
success_count = 0
fail_count = 0

for idx, item in enumerate(input_data, start=1):
    print(f"\n[{idx:02d}/{len(input_data):02d}]")
    print(f"  텍스트: {item['text'][:70]}...")
    print(f"  프롬프트 음성: {item['prompt_wav_path']}")

    try:
        # 프롬프트 캐시 확인
        prompt_wav_path = item['prompt_wav_path']
        if prompt_wav_path not in prompt_cache:
            print(f"  → 프롬프트 전처리 중...")
            resource_context = preprocess_prompt(prompt_wav_path)
            prompt_cache[prompt_wav_path] = resource_context
        else:
            print(f"  → 캐시된 프롬프트 사용")
            resource_context = prompt_cache[prompt_wav_path]

        # 음성 생성
        result = generate_and_save_audio(
            resource_context,
            text=item["text"],
            filename=item["filename"],
            time_step=TIME_STEP,
            p_w=P_W,
            t_w=T_W
        )

        if result:
            success_count += 1
        else:
            fail_count += 1

    except Exception as e:
        print(f"  ❌ 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        fail_count += 1

# ============================================================================
# 8. 결과 요약
# ============================================================================
print("\n" + "=" * 80)
print("모든 작업이 완료되었습니다.")
print("=" * 80)
print(f"  총 {len(input_data)}개 중:")
print(f"    ✅ 성공: {success_count}개")
print(f"    ❌ 실패: {fail_count}개")
print(f"  결과 디렉토리: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 80)
