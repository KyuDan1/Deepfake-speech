"""
LibriSpeech DataFrame Builder

LibriSpeech 데이터셋의 transcript와 audio path를 pandas DataFrame으로 변환하는 모듈입니다.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Optional, Union


def build_librispeech_dataframe(
    librispeech_root: Union[str, Path],
    subset: str = "test-clean"
) -> pd.DataFrame:
    """
    LibriSpeech 데이터셋을 pandas DataFrame으로 로드합니다.
    
    Parameters
    ----------
    librispeech_root : str or Path
        LibriSpeech 루트 디렉토리 경로 (예: "./my_librispeech/LibriSpeech")
    subset : str
        사용할 서브셋 이름 (예: "test-clean", "train-clean-100", etc.)
    
    Returns
    -------
    pd.DataFrame
        다음 컬럼을 가진 DataFrame:
        - speaker_id (int): 화자 ID
        - chapter_id (int): 챕터 ID
        - utterance_id (str): 발화 ID (예: "1089-134686-0000")
        - transcript (str): 전사 텍스트
        - audio_path (str): 오디오 파일 경로 (.flac)
    
    Examples
    --------
    >>> df = build_librispeech_dataframe("./my_librispeech/LibriSpeech", "test-clean")
    >>> df.head()
    """
    root = Path(librispeech_root)
    subset_path = root / subset
    
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset path not found: {subset_path}")
    
    records = []
    
    # speaker_id 폴더들 순회
    for speaker_dir in sorted(subset_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        
        # chapter_id 폴더들 순회
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
                
            chapter_id = chapter_dir.name
            
            # trans.txt 파일 찾기
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            
            if not trans_file.exists():
                continue
            
            # trans.txt 파일 파싱
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 형식: "1089-134686-0000 HE HOPED THERE WOULD BE..."
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    
                    utterance_id = parts[0]
                    transcript = parts[1]
                    
                    # 오디오 파일 경로
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    
                    if audio_path.exists():
                        records.append({
                            'speaker_id': int(speaker_id),
                            'chapter_id': int(chapter_id),
                            'utterance_id': utterance_id,
                            'transcript': transcript,
                            'audio_path': str(audio_path)
                        })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print(f"Warning: No data found in {subset_path}")
    else:
        print(f"Loaded {len(df)} utterances from {subset_path}")
        print(f"  - Speakers: {df['speaker_id'].nunique()}")
        print(f"  - Chapters: {df['chapter_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    # 테스트 실행
    df = build_librispeech_dataframe("./my_librispeech/LibriSpeech", "test-clean")
    print("\nDataFrame sample:")
    print(df.head(10))

