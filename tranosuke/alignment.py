import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from tranosuke.config import get_app_paths
from tranosuke.utils import adjust_ipu_time, float_to_timecode


def _align_phoneme_sequence(
    audio_segment,
    sample_rate: int,
    phoneme_sequence: str,
    iterations: int,
    model_path: Path,
) -> list:
    worker_path = Path(__file__).with_name("alignment_worker.py")
    with tempfile.TemporaryDirectory(prefix="tranosuke_align_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        request_path = tmp_path / "request.json"
        segment_path = tmp_path / "segment.wav"
        output_path = tmp_path / "output.json"
        request_path.write_text(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "phoneme_sequence": phoneme_sequence,
                    "iterations": iterations,
                }
            )
        )
        sf.write(segment_path, audio_segment, sample_rate)
        result = subprocess.run(
            [sys.executable, str(worker_path), str(request_path), str(segment_path), str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "pydomino alignment subprocess failed "
                f"with code {result.returncode}: {result.stderr or result.stdout}"
            )
        alignment = json.loads(output_path.read_text())
    return [item for item in alignment if item[-1] != "pau"]


def build_phoneme_alignment(
    audio_path: str | Path,
    df_ipu: pd.DataFrame,
    df_morph: pd.DataFrame,
    iterations: int = 3,
    alignment_buffer_s: float = 0.1,
) -> pd.DataFrame:
    source = Path(audio_path).expanduser().resolve()
    audio, sample_rate = sf.read(source)
    model_path = get_app_paths().phoneme_model_path
    phoneme_sequences = (
        df_morph.groupby("ipuID")["phonemes"].apply(lambda values: " ".join(values)).to_dict()
    )
    sorted_ipus = df_ipu.sort_values(["speaker", "startTime", "endTime"]).reset_index(drop=True)

    rows = []
    for _, ipu_row in sorted_ipus.iterrows():
        if pd.isna(ipu_row["ipu"]):
            continue

        ipu_id = ipu_row["ipuID"]
        phoneme_sequence = phoneme_sequences.get(ipu_id, "").strip()
        if not phoneme_sequence:
            continue

        ipu_start = max(float(ipu_row["startTime"]), 0.0)
        ipu_end = min(float(ipu_row["endTime"]), len(audio) / sample_rate)
        previous_ipus = sorted_ipus[sorted_ipus["endTime"] <= ipu_start]
        next_ipus = sorted_ipus[sorted_ipus["startTime"] >= ipu_end]
        previous_boundary = float(previous_ipus["endTime"].max()) if not previous_ipus.empty else 0.0
        next_boundary = float(next_ipus["startTime"].min()) if not next_ipus.empty else len(audio) / sample_rate
        max_left_buffer = max(ipu_start - previous_boundary, 0.0)
        max_right_buffer = max(next_boundary - ipu_end, 0.0)
        max_available_buffer = max(max_left_buffer, max_right_buffer, alignment_buffer_s)
        buffer_attempts = [alignment_buffer_s]
        for multiplier in [2.0, 4.0]:
            buffer_attempts.append(alignment_buffer_s * multiplier)
        buffer_attempts.append(max_available_buffer)
        buffer_attempts = sorted({round(max(buffer, 0.0), 3) for buffer in buffer_attempts})

        phonemes = None
        alignment_start = ipu_start
        for buffer_s in buffer_attempts:
            attempt_start = max(ipu_start - buffer_s, previous_boundary, 0.0)
            attempt_end = min(ipu_end + buffer_s, next_boundary, len(audio) / sample_rate)
            start_sample = max(int(attempt_start * sample_rate), 0)
            end_sample = min(int(attempt_end * sample_rate), len(audio))
            if end_sample <= start_sample:
                continue

            try:
                phonemes = _align_phoneme_sequence(
                    audio[start_sample:end_sample],
                    sample_rate,
                    phoneme_sequence,
                    iterations,
                    model_path,
                )
                alignment_start = attempt_start
                if buffer_s != alignment_buffer_s:
                    print(f"alignment retry succeeded: {ipu_id} buffer={buffer_s:.3f}s")
                break
            except Exception as error:
                print(f"alignment failed: {ipu_id} buffer={buffer_s:.3f}s {error}")

        if phonemes is None:
            continue

        for start_time, end_time, phoneme in phonemes:
            rows.append(
                [
                    ipu_row["filename"],
                    ipu_row["speaker"],
                    ipu_id,
                    round(float(start_time) + alignment_start, 4),
                    round(float(end_time) + alignment_start, 4),
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
    alignment_buffer_s: float = 0.1,
) -> dict[str, pd.DataFrame | Path]:
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(audio_path).expanduser().resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)

    df_phon = build_phoneme_alignment(
        audio_path, df_ipu, df_morph, iterations=iterations, alignment_buffer_s=alignment_buffer_s
    )
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
