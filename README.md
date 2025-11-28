# Deepfake-speech

## LibriSpeech dataframe helper

To build a dataframe that pairs every audio file in `my_raw_audio/LibriSpeech` with its
matching transcript (e.g. entries from `61-70968.trans.txt`), run:

```bash
python libri_dataframe.py
```

or import the helper in your own code:

```python
from libri_dataframe import build_librispeech_dataframe

df = build_librispeech_dataframe("./my_raw_audio/LibriSpeech", subset="test-clean")
print(df.head())
```

The dataframe includes the `transcript` column alongside speaker/chapter metadata and
absolute audio paths for each `.flac` file.
