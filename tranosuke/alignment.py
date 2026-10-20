import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from tranosuke.config import get_app_paths
from tranosuke.utils import float_to_timecode, timecode_integer_digits


class _PydominoWorker:
    """pydomino を別プロセスで動かし、IPU をまたいで使い回す。

    pydomino はセグメンテーション違反で落ちることがあるので本体とはプロセスを分ける。
    落ちたときはその IPU を失敗として扱い、次の呼び出しで起動し直す。
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.worker_path = Path(__file__).with_name("alignment_worker.py")
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tranosuke_align_")
        self.segment_path = Path(self.tmp_dir.name) / "segment.wav"
        self.stderr_path = Path(self.tmp_dir.name) / "worker_stderr.log"
        self.process: subprocess.Popen | None = None

    def _start(self) -> None:
        self.stderr_file = self.stderr_path.open("w")
        self.process = subprocess.Popen(
            [sys.executable, str(self.worker_path), "--serve", str(self.model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            encoding="utf-8",
        )

    def _stderr_tail(self) -> str:
        self.stderr_file.flush()
        lines = self.stderr_path.read_text(errors="replace").splitlines()
        return "\n".join(line for line in lines[-5:] if "pkg_resources" not in line)

    def align(self, audio_segment, sample_rate: int, phoneme_sequence: str, min_aligned_timeframe: int) -> list:
        if self.process is None or self.process.poll() is not None:
            self._start()
        sf.write(self.segment_path, audio_segment, sample_rate)
        request = {
            "segment_path": str(self.segment_path),
            "phoneme_sequence": phoneme_sequence,
            "min_aligned_timeframe": min_aligned_timeframe,
        }
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError:
            line = ""
        if not line:
            returncode = self.process.wait()
            self.stderr_file.close()
            self.process = None
            raise RuntimeError(f"pydomino worker exited with code {returncode}: {self._stderr_tail()}")

        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return [item for item in response["alignment"] if item[-1] != "pau"]

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()
        if self.process is not None:
            self.stderr_file.close()
        self.process = None
        self.tmp_dir.cleanup()


# pydomino の min_aligned_timeframe: pau 以外の各音素に割り当てる最低フレーム数（1フレーム = 10ms）
DEFAULT_MIN_ALIGNED_TIMEFRAME = 1
# 漢字・ひらがな・カタカナ（長音符・踊り字・〇 を含む）以外の文字
NON_JAPANESE_CHARACTER_PATTERN = re.compile(r"[^\u3041-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF々〇〆]")
# 音素が作れなかった語に均等割り当てで区間を与えるための仮の音素
UNKNOWN_PHONEME = "unk"
# アラインメント再試行で IPU の前後に付ける余白の上限（秒）
MAX_ALIGNMENT_BUFFER_S = 1.0


def _word_phonemes(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text or UNKNOWN_PHONEME


def _uniform_phoneme_alignment(phoneme_sequence: str, duration_s: float) -> list:
    # 境界をミリ秒単位の整数にそろえる。ID は開始時刻のミリ秒から作るので、端数があると隣どうしで同じ ID になりうる
    phonemes = phoneme_sequence.split()
    total_ms = _duration_ms(duration_s)
    boundaries = [round(index * total_ms / len(phonemes)) / 1000 for index in range(len(phonemes) + 1)]
    return [(boundaries[index], boundaries[index + 1], phoneme) for index, phoneme in enumerate(phonemes)]


def _duration_ms(duration_s: float) -> int:
    return int(round(max(duration_s, 0.0) * 1000))


def build_phoneme_alignment(
    audio_path: str | Path,
    df_ipu: pd.DataFrame,
    df_morph: pd.DataFrame,
    min_aligned_timeframe: int = DEFAULT_MIN_ALIGNED_TIMEFRAME,
    alignment_buffer_s: float = 0.1,
    timecode_digits: int | None = None,
) -> pd.DataFrame:
    source = Path(audio_path).expanduser().resolve()
    audio, sample_rate = sf.read(source)
    if timecode_digits is None:
        timecode_digits = timecode_integer_digits(len(audio) / sample_rate)
    worker = _PydominoWorker(get_app_paths().phoneme_model_path)
    phoneme_sequences = (
        df_morph.groupby("IPUID")["phonemes"]
        .apply(lambda values: " ".join(_word_phonemes(value) for value in values))
        .to_dict()
    )
    sorted_ipus = df_ipu.sort_values(["speaker", "startTime", "endTime"]).reset_index(drop=True)

    try:
        rows = _align_ipus(
            sorted_ipus, phoneme_sequences, audio, sample_rate, worker, min_aligned_timeframe, alignment_buffer_s
        )
    finally:
        worker.close()

    df_phon = pd.DataFrame(
        rows,
        columns=["filename", "speaker", "IPUID", "startTime", "endTime", "phoneme", "aligned"],
    )
    df_phon = _assign_time_ids(df_phon, "phonemeID", timecode_digits)
    df_phon["tier"] = "Phoneme_" + df_phon["speaker"].astype(str)
    return df_phon[
        ["filename", "speaker", "tier", "IPUID", "phonemeID", "startTime", "endTime", "phoneme", "aligned"]
    ]


def _align_ipus(
    sorted_ipus: pd.DataFrame,
    phoneme_sequences: dict,
    audio,
    sample_rate: int,
    worker: _PydominoWorker,
    min_aligned_timeframe: int,
    alignment_buffer_s: float,
) -> list:
    rows = []
    for _, ipu_row in sorted_ipus.iterrows():
        if pd.isna(ipu_row["IPU"]):
            continue

        ipu_id = ipu_row["IPUID"]
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
        # 隣の IPU までの空きが長いと区間が数十秒になり pydomino がメモリを使い果たすので上限を設ける
        max_available_buffer = min(
            max(max_left_buffer, max_right_buffer, alignment_buffer_s),
            max(MAX_ALIGNMENT_BUFFER_S, alignment_buffer_s),
        )
        buffer_attempts = [alignment_buffer_s]
        for multiplier in [2.0, 4.0]:
            buffer_attempts.append(alignment_buffer_s * multiplier)
        buffer_attempts.append(max_available_buffer)
        buffer_attempts = sorted({round(max(buffer, 0.0), 3) for buffer in buffer_attempts})

        phonemes = None
        alignment_start = ipu_start
        # 漢字・ひらがな・カタカナ以外を含む IPU は読みが信頼できないので、pydomino を使わず均等割り当てにする
        # 音素が作れなかった語（仮の音素 UNKNOWN_PHONEME）を含む IPU も pydomino には渡せない
        if NON_JAPANESE_CHARACTER_PATTERN.search(str(ipu_row["IPU"])) or UNKNOWN_PHONEME in phoneme_sequence.split():
            print(f"alignment not attempted (non-Japanese characters): {ipu_id}")
            buffer_attempts = []
        for buffer_s in buffer_attempts:
            attempt_start = max(ipu_start - buffer_s, previous_boundary, 0.0)
            attempt_end = min(ipu_end + buffer_s, next_boundary, len(audio) / sample_rate)
            start_sample = max(int(attempt_start * sample_rate), 0)
            end_sample = min(int(attempt_end * sample_rate), len(audio))
            if end_sample <= start_sample:
                continue

            try:
                phonemes = worker.align(
                    audio[start_sample:end_sample],
                    sample_rate,
                    phoneme_sequence,
                    min_aligned_timeframe,
                )
                alignment_start = attempt_start
                if buffer_s != alignment_buffer_s:
                    print(f"alignment retry succeeded: {ipu_id} buffer={buffer_s:.3f}s")
                break
            except Exception as error:
                print(f"alignment failed: {ipu_id} buffer={buffer_s:.3f}s {error}")

        aligned = phonemes is not None
        if not aligned:
            # 1音素 1ms（MIN_EDGE_ITEM_DURATION_S）も確保できないほど短い IPU は、均等割り当てもできないのでスキップする
            if _duration_ms(ipu_end - ipu_start) < len(phoneme_sequence.split()):
                print(f"alignment skipped (IPU too short for its phonemes): {ipu_id}")
                continue
            # IPU 全体でアラインメントに失敗した場合は、IPU 区間に音素を均等に割り当てる
            alignment_start = ipu_start
            phonemes = _uniform_phoneme_alignment(phoneme_sequence, ipu_end - ipu_start)
            print(f"alignment fallback (uniform): {ipu_id}")

        for start_time, end_time, phoneme in phonemes:
            rows.append(
                [
                    ipu_row["filename"],
                    ipu_row["speaker"],
                    ipu_id,
                    round(float(start_time) + alignment_start, 4),
                    round(float(end_time) + alignment_start, 4),
                    phoneme,
                    aligned,
                ]
            )
    return rows


MIN_EDGE_ITEM_DURATION_S = 0.001


def _infer_timecode_digits(df: pd.DataFrame) -> int:
    """Fallback when the audio length is unknown: size IDs to the latest time in the table."""
    if "endTime" not in df.columns:
        return timecode_integer_digits(0.0)
    latest = pd.to_numeric(df["endTime"], errors="coerce").max()
    return timecode_integer_digits(0.0 if pd.isna(latest) else float(latest))


def _assign_time_ids(df: pd.DataFrame, id_column: str, timecode_digits: int) -> pd.DataFrame:
    """開始時刻（秒の整数部 timecode_digits 桁 + ミリ秒3桁）と話者から ID を付ける。"""
    df = df.copy()
    df[id_column] = [
        f"{float_to_timecode(start_time, timecode_digits)}{speaker}"
        for start_time, speaker in zip(df["startTime"], df["speaker"].astype(str))
    ]
    return df


def _force_group_edges_to_ipu_boundaries(
    df_items: pd.DataFrame,
    df_ipu: pd.DataFrame,
    id_column: str,
    timecode_digits: int | None = None,
) -> pd.DataFrame:
    """Force first/last item times in each IPU to match the IPU boundary.

    Items aligned into the buffer outside the IPU are packed inside the boundary in order,
    each keeping at least MIN_EDGE_ITEM_DURATION_S, with as little movement as possible.
    """
    if df_items.empty:
        return df_items.copy()
    if timecode_digits is None:
        timecode_digits = _infer_timecode_digits(df_ipu)

    adjusted = df_items.copy()
    ipu_boundaries = df_ipu.set_index("IPUID")[["startTime", "endTime"]].to_dict("index")

    for ipu_id, group in adjusted.groupby("IPUID", sort=False):
        boundaries = ipu_boundaries.get(ipu_id)
        if boundaries is None:
            continue

        # 行の順は pydomino の出力順（語は語順）なので、時刻で並べ替えずにその順のまま詰める
        timed_group = group.dropna(subset=["startTime", "endTime"])
        if timed_group.empty:
            continue

        ipu_start = float(boundaries["startTime"])
        ipu_end = float(boundaries["endTime"])
        indices = list(timed_group.index)
        starts = timed_group["startTime"].astype(float).tolist()
        ends = timed_group["endTime"].astype(float).tolist()

        # 先頭は IPU 開始に合わせ、以降は前の要素と重ならないよう後ろへずらす
        previous_end = ipu_start
        for position in range(len(indices)):
            start = ipu_start if position == 0 else max(starts[position], previous_end)
            end = max(ends[position], start + MIN_EDGE_ITEM_DURATION_S)
            starts[position], ends[position] = round(start, 4), round(end, 4)
            previous_end = ends[position]

        # 末尾は IPU 終了に合わせ、以前の要素は次の要素と重ならないよう前へずらす
        next_start = ipu_end
        for position in reversed(range(len(indices))):
            end = ipu_end if position == len(indices) - 1 else min(ends[position], next_start)
            start = min(starts[position], end - MIN_EDGE_ITEM_DURATION_S)
            starts[position], ends[position] = round(start, 4), round(end, 4)
            next_start = starts[position]

        adjusted.loc[indices, "startTime"] = starts
        adjusted.loc[indices, "endTime"] = ends

    return _assign_time_ids(adjusted, id_column, timecode_digits)


def build_word_alignment(
    df_morph: pd.DataFrame,
    df_phon: pd.DataFrame,
    timecode_digits: int | None = None,
) -> pd.DataFrame:
    if timecode_digits is None:
        timecode_digits = _infer_timecode_digits(df_phon)
    rows = []
    # 音素は pydomino の出力順のまま照合する（時刻で並べ替えると、時刻が同じ音素の順が入れ替わりうる）
    phoneme_groups = {ipu_id: group.reset_index(drop=True) for ipu_id, group in df_phon.groupby("IPUID", sort=False)}

    for ipu_id, morph_group in df_morph.groupby("IPUID"):
        phoneme_group = phoneme_groups.get(ipu_id)
        if phoneme_group is None or phoneme_group.empty:
            continue

        phoneme_list = [item.lower() for item in phoneme_group["phoneme"].tolist()]
        start_times = phoneme_group["startTime"].tolist()
        end_times = phoneme_group["endTime"].tolist()
        aligned_flags = phoneme_group["aligned"].astype(bool).tolist()
        phoneme_index = 0

        for _, word_row in morph_group.sort_values("nth").iterrows():
            word_phonemes = [item.lower() for item in _word_phonemes(word_row["phonemes"]).split()]

            word_start = None
            word_end = None
            word_aligned = True
            match_index = 0

            while phoneme_index < len(phoneme_list) and match_index < len(word_phonemes):
                if phoneme_list[phoneme_index] == word_phonemes[match_index]:
                    if match_index == 0:
                        word_start = start_times[phoneme_index]
                    word_end = end_times[phoneme_index]
                    word_aligned = word_aligned and aligned_flags[phoneme_index]
                    match_index += 1
                phoneme_index += 1

            if word_start is None or word_end is None:
                continue
            # 語は構成する音素がすべてアラインメントに成功したときだけ成功とみなす
            rows.append([ipu_id, word_start, word_end, word_row["nth"], word_aligned])

    df_word_timing = pd.DataFrame(rows, columns=["IPUID", "startTime", "endTime", "nth", "aligned"])
    df_word = pd.merge(df_morph, df_word_timing, on=["IPUID", "nth"], how="left")
    df_word["aligned"] = df_word["aligned"].fillna(False).astype(bool)
    df_word = _assign_time_ids(df_word, "wordID", timecode_digits)
    df_word["tier"] = "Word_" + df_word["speaker"].astype(str)
    return df_word[
        [
            "filename",
            "speaker",
            "tier",
            "IPUID",
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
            "aligned",
            "nth",
            "len",
        ]
    ]


def build_word_to_ipu(df_word: pd.DataFrame) -> pd.DataFrame:
    """Build a compact mapping table between words and IPUs."""
    if df_word.empty:
        return pd.DataFrame(columns=["filename", "wordID", "IPUID", "nth", "len"])

    mapping = df_word[["filename", "wordID", "IPUID", "nth", "len"]].copy()
    return mapping


def build_phoneme_to_ipu(df_phon: pd.DataFrame) -> pd.DataFrame:
    """Build a compact mapping table between phonemes and IPUs."""
    if df_phon.empty:
        return pd.DataFrame(columns=["filename", "phonemeID", "IPUID", "nth", "len"])

    mapping = df_phon[["filename", "phonemeID", "IPUID"]].copy()
    mapping["nth"] = mapping.groupby("IPUID").cumcount() + 1
    mapping["len"] = mapping.groupby("IPUID")["phonemeID"].transform("size")
    return mapping


def align_phonemes_and_words(
    audio_path: str | Path,
    df_ipu: pd.DataFrame,
    df_morph: pd.DataFrame,
    output_dir: str | Path | None = None,
    min_aligned_timeframe: int = DEFAULT_MIN_ALIGNED_TIMEFRAME,
    alignment_buffer_s: float = 0.1,
) -> dict[str, pd.DataFrame | Path]:
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(audio_path).expanduser().resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)

    timecode_digits = timecode_integer_digits(sf.info(str(Path(audio_path).expanduser().resolve())).duration)
    df_phon = build_phoneme_alignment(
        audio_path,
        df_ipu,
        df_morph,
        min_aligned_timeframe=min_aligned_timeframe,
        alignment_buffer_s=alignment_buffer_s,
        timecode_digits=timecode_digits,
    )
    df_phon = _force_group_edges_to_ipu_boundaries(df_phon, df_ipu, "phonemeID", timecode_digits)
    df_word = build_word_alignment(df_morph, df_phon, timecode_digits)
    # 音素が1つもない IPU（スキップした IPU）の語は出力しない
    df_word = df_word[df_word["IPUID"].isin(df_phon["IPUID"])].copy()
    df_word = _force_group_edges_to_ipu_boundaries(df_word, df_ipu, "wordID", timecode_digits)
    df_word_to_ipu = build_word_to_ipu(df_word)
    df_phon_to_ipu = build_phoneme_to_ipu(df_phon)
    df_ipu_output = df_ipu.copy()

    phoneme_csv = target_dir / "phoneme.csv"
    word_csv = target_dir / "word.csv"
    word2ipu_csv = target_dir / "word2IPU.csv"
    phoneme2ipu_csv = target_dir / "phoneme2IPU.csv"
    ipu_csv = target_dir / "IPU.csv"

    df_phon_output = df_phon.drop(columns=["IPUID"], errors="ignore")
    df_word_output = df_word.drop(columns=["IPUID", "nth", "len"], errors="ignore")

    df_phon_output.to_csv(phoneme_csv, encoding="utf-8_sig", index=False)
    df_word_output.to_csv(word_csv, encoding="utf-8_sig", index=False)
    df_word_to_ipu.to_csv(word2ipu_csv, encoding="utf-8_sig", index=False)
    df_phon_to_ipu.to_csv(phoneme2ipu_csv, encoding="utf-8_sig", index=False)
    df_ipu_output.to_csv(ipu_csv, encoding="utf-8_sig", index=False)

    return {
        "phoneme_csv": phoneme_csv,
        "word_csv": word_csv,
        "word2ipu_csv": word2ipu_csv,
        "phoneme2ipu_csv": phoneme2ipu_csv,
        "ipu_csv": ipu_csv,
        "phoneme_df": df_phon,
        "word_df": df_word,
        "word2ipu_df": df_word_to_ipu,
        "phoneme2ipu_df": df_phon_to_ipu,
        "ipu_df": df_ipu_output,
    }
