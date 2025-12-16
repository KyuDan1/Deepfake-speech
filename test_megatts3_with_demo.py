"""
MegaTTS3 테스트 스크립트 - 데모 프롬프트 사용
LibriSpeech 텍스트를 MegaTTS3로 합성하되, 프롬프트는 데모 파일 사용
"""
import os
import sys
import pandas as pd

# MegaTTS3 경로 추가
megatts3_root = '/mnt/ddn/kyudan/MegaTTS3'
if megatts3_root not in sys.path:
    sys.path.insert(0, megatts3_root)

import torch
from tts.infer_cli import MegaTTS3DiTInfer
from tts.utils.audio_utils.io import save_wav

# 설정
OUTPUT_DIR = "generated_results_demo_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 초기화
print("=" * 80)
print("MegaTTS3 모델 로딩 중...")
print("=" * 80)

infer_ins = MegaTTS3DiTInfer(
    device='cuda',
    ckpt_root='/mnt/ddn/kyudan/MegaTTS3/checkpoints',
    precision=torch.float16
)

print("\n✅ 모델 로딩 완료")

# 데모 프롬프트 전처리 (영어)
demo_prompt_path = '/mnt/ddn/kyudan/MegaTTS3/assets/English_prompt.wav'
demo_latent_path = '/mnt/ddn/kyudan/MegaTTS3/assets/English_prompt.npy'

print(f"\n프롬프트 전처리 중: {demo_prompt_path}")
with open(demo_prompt_path, 'rb') as f:
    audio_bytes = f.read()

resource_context = infer_ins.preprocess(
    audio_bytes,
    latent_file=demo_latent_path,
    topk_dur=1
)
print("✅ 프롬프트 전처리 완료")

# LibriSpeech 메타데이터 로드
metadata_path = '/mnt/ddn/kyudan/Deepfake-speech/metadata/librispeech_test_clean.tsv'
df = pd.read_csv(metadata_path, sep='\t')

print(f"\n총 {len(df)}개 샘플 중 처음 10개만 테스트합니다.")
print("=" * 80)

# 처음 10개만 테스트
for idx, row in df.head(10).iterrows():
    text = row['text']
    audio_path = row['audio_path']

    # 파일명 생성
    filename = os.path.basename(audio_path).replace('.flac', '_generated.wav')

    print(f"\n[{idx+1:02d}/10]")
    print(f"  텍스트: {text[:70]}...")

    try:
        # 음성 생성
        wav_bytes = infer_ins.forward(
            resource_context,
            text,
            time_step=32,
            p_w=2.0,
            t_w=3.0
        )

        # 저장
        output_path = os.path.join(OUTPUT_DIR, filename)
        save_wav(wav_bytes, output_path)
        print(f"  ✅ 저장: {output_path}")

    except Exception as e:
        print(f"  ❌ 실패: {e}")

print("\n" + "=" * 80)
print("테스트 완료!")
print(f"결과 디렉토리: {OUTPUT_DIR}")
print("=" * 80)
