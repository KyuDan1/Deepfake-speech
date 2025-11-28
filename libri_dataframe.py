import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def _parse_transcript_file(file_path: Path) -> Dict[str, str]:
    """Parse a single *.trans.txt file and return {utterance_id: transcript}."""
    transcripts: Dict[str, str] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                utt_id, text = line.split(" ", 1)
            except ValueError:
                continue  # malformed line
            transcripts[utt_id] = text.strip()
    return transcripts


def build_librispeech_dataframe(
    librispeech_root: str,
    subset: Optional[str] = "test-clean"
) -> pd.DataFrame:
    """
    Convert LibriSpeech audio + transcript pairs to a pandas DataFrame.

    Args:
        librispeech_root: Path up to the LibriSpeech directory (e.g. ./my_raw_audio/LibriSpeech)
        subset: Optional split under the root (e.g. test-clean, train-clean-100). If None, the
                function walks every transcript file under the root.

    Returns:
        pd.DataFrame with columns:
            - speaker_id
            - chapter_id
            - utterance_id
            - audio_path
            - transcript
    """
    root = Path(librispeech_root)
    search_dir = root / subset if subset else root

    if not search_dir.exists():
        raise FileNotFoundError(f"Could not find directory: {search_dir}")

    entries: List[Dict[str, str]] = []
    for transcript_path in sorted(search_dir.rglob("*.trans.txt")):
        transcripts = _parse_transcript_file(transcript_path)
        chapter_dir = transcript_path.parent

        for utt_id, text in transcripts.items():
            audio_file = chapter_dir / f"{utt_id}.flac"
            speaker_id, chapter_id, utterance_id = utt_id.split("-")

            entry = {
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "audio_path": str(audio_file),
                "transcript": text,
            }

            if not audio_file.exists():
                entry["missing_audio"] = True
            entries.append(entry)

    df = pd.DataFrame(entries)
    return df


if __name__ == "__main__":
    # Example: create a dataframe for ./my_raw_audio/LibriSpeech/test-clean
    dataframe = build_librispeech_dataframe(
        librispeech_root="./my_raw_audio/LibriSpeech",
        subset="test-clean"
    )
    print(dataframe.head())
