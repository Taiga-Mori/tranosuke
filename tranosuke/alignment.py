from pathlib import Path

import librosa
import pandas as pd
import pydomino
import soundfile as sf

from tranosuke.config import get_app_paths
from tranosuke.utils import adjust_ipu_time, float_to_timecode


def _align_phoneme_sequence(audio_segment, sample_rate: int, phoneme_sequence: str, iterations: int, aligner) -> list:
    if audio_segment.ndim > 1:
        audio_segment = audio_segment.mean(axis=1)
    resampled = librosa.resample(audio_segment.astype("float32"), orig_sr=sample_rate, target_sr=16000)
    alignment = aligner.align(resampled, phoneme_sequence, iterations)
    return [item for item in alignment if item[-1] != "pau"]


def build_phoneme_alignment(
    audio_path: str | Path,
    df_ipu: pd.DataFrame,
    df_morph: pd.DataFrame,
    iterations: int = 3,
) -> pd.DataFrame:
    source = Path(audio_path).expanduser().resolve()
    audio, sample_rate = sf.read(source)
    model_path = get_app_paths().cache_dir / "phoneme_transition_model.onnx"
    aligner = pydomino.Aligner(str(model_path))

    phoneme_sequences = (
        df_morph.groupby("ipuID")["phonemes"].apply(lambda values: " ".join(values)).to_dict()
    )

    rows = []
    for _, ipu_row in df_ipu.iterrows():
        if pd.isna(ipu_row["ipu"]):
            continue

        ipu_id = ipu_row["ipuID"]
        phoneme_sequence = phoneme_sequences.get(ipu_id, "").strip()
        if not phoneme_sequence:
            continue

        start_delay = 0.0
        end_buffer = 0.2
        start_sample = max(int((ipu_row["startTime"] + start_delay) * sample_rate), 0)
        end_sample = min(int((ipu_row["endTime"] + end_buffer) * sample_rate), len(audio))
        if end_sample <= start_sample:
            continue

        try:
            phonemes = _align_phoneme_sequence(
                audio[start_sample:end_sample],
                sample_rate,
                phoneme_sequence,
                iterations,
                aligner,
            )
        except Exception as error:
            print(error)
            continue

        for start_time, end_time, phoneme in phonemes:
            rows.append(
                [
                    ipu_row["filename"],
                    ipu_row["speaker"],
                    ipu_id,
                    round(float(start_time) + ipu_row["startTime"] + start_delay, 4),
                    round(float(end_time) + ipu_row["startTime"] + start_delay, 4),
                    phoneme,
                ]
            )

    df_phon = pd.DataFrame(
        rows,
        columns=["filename", "speaker", "ipuID", "startTime", "endTime", "phoneme"],
    )
    if df_phon.empty:
        return pd.DataFrame(
            columns=["filename", "speaker", "tier", "ipuID", "phonemeID", "startTime", "endTime", "phoneme"]
        )

    df_phon["timestamp"] = df_phon["startTime"].apply(float_to_timecode)
    df_phon["phonemeID"] = df_phon["timestamp"].astype(str) + df_phon["speaker"].astype(str)
    df_phon["tier"] = "Phoneme_" + df_phon["speaker"].astype(str)
    return df_phon[
        ["filename", "speaker", "tier", "ipuID", "phonemeID", "startTime", "endTime", "phoneme"]
    ]


def build_word_alignment(df_morph: pd.DataFrame, df_phon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phoneme_groups = {
        ipu_id: group.sort_values("startTime").reset_index(drop=True)
        for ipu_id, group in df_phon.groupby("ipuID")
    }

    for ipu_id, morph_group in df_morph.groupby("ipuID"):
        phoneme_group = phoneme_groups.get(ipu_id)
        if phoneme_group is None or phoneme_group.empty:
            continue

        phoneme_list = [item.lower() for item in phoneme_group["phoneme"].tolist()]
        start_times = phoneme_group["startTime"].tolist()
        end_times = phoneme_group["endTime"].tolist()
        phoneme_index = 0

        for _, word_row in morph_group.sort_values("nth").iterrows():
            word_phonemes = [item.lower() for item in str(word_row["phonemes"]).split() if item]
            if not word_phonemes:
                continue

            word_start = None
            word_end = None
            match_index = 0

            while phoneme_index < len(phoneme_list) and match_index < len(word_phonemes):
                if phoneme_list[phoneme_index] == word_phonemes[match_index]:
                    if match_index == 0:
                        word_start = start_times[phoneme_index]
                    word_end = end_times[phoneme_index]
                    match_index += 1
                phoneme_index += 1

            if word_start is None or word_end is None:
                continue
            rows.append([ipu_id, word_start, word_end, word_row["nth"]])

    df_word_timing = pd.DataFrame(rows, columns=["ipuID", "startTime", "endTime", "nth"])
    df_word = pd.merge(df_morph, df_word_timing, on=["ipuID", "nth"], how="left")
    df_word["timestamp"] = df_word["startTime"].apply(float_to_timecode)
    df_word["wordID"] = df_word["timestamp"].astype(str) + df_word["speaker"].astype(str)
    df_word["tier"] = "Word_" + df_word["speaker"].astype(str)
    return df_word[
        [
            "filename",
            "speaker",
            "tier",
            "ipuID",
            "wordID",
            "startTime",
            "endTime",
            "orth",
            "pos",
            "pos1",
            "pos2",
            "pos3",
            "cForm",
            "cType",
            "lemma",
            "ruby",
            "pron",
            "phonemes",
            "nth",
            "len",
        ]
    ]


def build_word_to_ipu(df_word: pd.DataFrame) -> pd.DataFrame:
    """Build an explicit mapping table between words and IPUs."""
    if df_word.empty:
        return pd.DataFrame(
            columns=["filename", "speaker", "wordID", "ipuID", "startTime", "endTime", "orth", "nth", "len"]
        )

    return df_word[
        ["filename", "speaker", "wordID", "ipuID", "startTime", "endTime", "orth", "nth", "len"]
    ].copy()


def align_phonemes_and_words(
    audio_path: str | Path,
    df_ipu: pd.DataFrame,
    df_morph: pd.DataFrame,
    output_dir: str | Path | None = None,
    iterations: int = 3,
) -> dict[str, pd.DataFrame | Path]:
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(audio_path).expanduser().resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)

    df_phon = build_phoneme_alignment(audio_path, df_ipu, df_morph, iterations=iterations)
    df_word = build_word_alignment(df_morph[df_morph["orth"] != "¥"].copy(), df_phon)
    df_word_to_ipu = build_word_to_ipu(df_word)
    df_ipu_adjusted = adjust_ipu_time(df_ipu, df_phon)
    df_ipu_adjusted["ipu"] = df_ipu_adjusted["ipu"].str.replace("¥", "", regex=False)

    phoneme_csv = target_dir / "phoneme.csv"
    word_csv = target_dir / "word.csv"
    word2ipu_csv = target_dir / "word2ipu.csv"
    ipu_csv = target_dir / "ipu.csv"

    df_phon.to_csv(phoneme_csv, encoding="utf-8_sig", index=False)
    df_word.to_csv(word_csv, encoding="utf-8_sig", index=False)
    df_word_to_ipu.to_csv(word2ipu_csv, encoding="utf-8_sig", index=False)
    df_ipu_adjusted.to_csv(ipu_csv, encoding="utf-8_sig", index=False)

    return {
        "phoneme_csv": phoneme_csv,
        "word_csv": word_csv,
        "word2ipu_csv": word2ipu_csv,
        "ipu_csv": ipu_csv,
        "phoneme_df": df_phon,
        "word_df": df_word,
        "word2ipu_df": df_word_to_ipu,
        "ipu_df": df_ipu_adjusted,
    }
