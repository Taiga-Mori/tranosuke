from pathlib import Path

import pandas as pd

from tranosuke.utils import float_to_timecode


INTERJECTION_WORDS = {
    "ああ", "あー", "あーん", "あい", "あっ", "あ", "あら", "うーん", "うん", "うお", "うむ", "うわ",
    "ええ", "えー", "えっ", "え", "おい", "おーい", "おお", "おー", "おっ", "お", "ね", "ねえ", "はあ",
    "はー", "はい", "ふう", "ふー", "ふむ", "ふむふむ", "ふん", "ふんふん", "ふーん", "へい", "へー",
    "へえ", "ほい", "ほう", "ほー", "ほほう", "ほほー", "まあ", "まー", "むむ", "もう", "もー",
    "もしもし", "いや", "やあ", "わあ", "わー", "わあい", "わーい", "わあん", "わーん",
}
CONNECTIVE_WORDS = {
    "あと", "あるいは", "一方", "が", "かといって", "けど", "けれど", "けれども", "けども", "さて",
    "さらに", "しかし", "しかしながら", "しかも", "したがって", "すなわち", "すると", "そこで",
    "そして", "それか", "それから", "それじゃ", "それで", "それとは", "それとも", "ただ", "とくに",
    "ところが", "ところで", "といっても", "とすると", "と同時に", "なお", "なおかつ", "なんですが",
    "なんですけど", "なんですけれど", "ほいから", "ほんじゃ", "ほんで",
}
HIGH_INDEPENDENCE_ENDINGS = {"が", "けど", "けども", "けれど", "けれども", "し"}
LOW_INDEPENDENCE_ENDINGS = {
    "から", "ので", "んで", "たら", "で", "て", "っていう", "とか", "みたいな", "ような", "ながら",
}
END_PARTICLES = {"ね", "よ", "か", "な", "の", "さ", "じゃん", "っけ", "もん", "かしら", "わ", "や", "ん", "ぜ", "ぞ", "け"}
LEXICAL_AIZUCHI = {"そうそう", "そうか", "なるほど", "まあね", "ね"}


def _join_surface(words: pd.DataFrame) -> str:
    return "".join(str(value) for value in words["orth"].tolist() if pd.notna(value))


def _normalize(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def _starts_with_interjection_or_connective(next_words: pd.DataFrame | None) -> bool:
    if next_words is None or next_words.empty:
        return False
    first = _normalize(next_words.iloc[0]["orth"])
    return first in INTERJECTION_WORDS or first in CONNECTIVE_WORDS


def _is_interjection_only(ipu_words: pd.DataFrame) -> bool:
    orths = [_normalize(value) for value in ipu_words["orth"].tolist()]
    if not orths:
        return False
    return all(word in INTERJECTION_WORDS for word in orths)


def _is_explicit_sentence_final(ipu_words: pd.DataFrame) -> bool:
    if ipu_words.empty:
        return False

    surface = _join_surface(ipu_words)
    last = ipu_words.iloc[-1]
    last_orth = _normalize(last["orth"])
    last_pos = _normalize(last["pos"])
    last_pos1 = _normalize(last["pos1"])

    if surface in LEXICAL_AIZUCHI:
        return True
    if last_pos == "助動詞":
        return True
    if last_orth in END_PARTICLES:
        return True
    if last_pos == "助詞" and last_pos1 == "終助詞":
        return True
    if last_pos in {"名詞", "副詞"} and len(ipu_words) == 1:
        return True

    if len(ipu_words) >= 2:
        prev = ipu_words.iloc[-2]
        prev_pos = _normalize(prev["pos"])
        if last_orth in END_PARTICLES and prev_pos in {"動詞", "形容詞", "助動詞", "名詞", "副詞"}:
            return True

    return False


def _is_topic_intro(ipu_words: pd.DataFrame) -> bool:
    orths = [_normalize(value) for value in ipu_words["orth"].tolist()]
    if len(orths) < 2:
        return False
    tail2 = "".join(orths[-2:])
    tail3 = "".join(orths[-3:]) if len(orths) >= 3 else ""
    return tail2 in {"のは", "ことは"} or tail3 in {"というのは", "っていうのは"}


def _is_summary_intro(next_words: pd.DataFrame | None) -> bool:
    if next_words is None or next_words.empty:
        return False
    text = _join_surface(next_words.iloc[:3])
    return any(text.startswith(prefix) for prefix in {"つまり", "要するに", "というわけ", "ってわけ", "反省", "結局"})


def _summarize_ipus_from_words(df_word: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for ipu_id, group in df_word.sort_values(["startTime", "endTime", "wordID"]).groupby("ipuID", sort=False):
        valid_starts = group["startTime"].dropna()
        valid_ends = group["endTime"].dropna()
        summaries.append(
            {
                "filename": group.iloc[0]["filename"],
                "speaker": group.iloc[0]["speaker"],
                "ipuID": ipu_id,
                "startTime": valid_starts.min() if not valid_starts.empty else pd.NA,
                "endTime": valid_ends.max() if not valid_ends.empty else pd.NA,
                "ipu": _join_surface(group),
                "wordCount": len(group),
            }
        )
    return pd.DataFrame(summaries)


def _decide_boundary(
    current_words: pd.DataFrame,
    next_words: pd.DataFrame | None,
    current_summary: pd.Series,
    next_summary: pd.Series | None,
) -> tuple[bool, str]:
    if next_summary is None:
        return True, "END"

    speaker_changed = _normalize(current_summary["speaker"]) != _normalize(next_summary["speaker"])
    last_orth = _normalize(current_words.iloc[-1]["orth"]) if not current_words.empty else ""

    if _is_interjection_only(current_words):
        return True, "R"
    if last_orth in HIGH_INDEPENDENCE_ENDINGS:
        return True, "L_SYNTACTIC_HIGH_SUBORDINATE"
    if last_orth in LOW_INDEPENDENCE_ENDINGS and (speaker_changed or _starts_with_interjection_or_connective(next_words)):
        return True, "L_SYNTACTIC_LOW_SUBORDINATE"
    if _is_topic_intro(current_words):
        return True, "L_DISCOURSE_TOPIC"
    if _is_summary_intro(next_words):
        return True, "L_DISCOURSE_SUMMARY"
    if _is_explicit_sentence_final(current_words):
        return True, "L_SYNTACTIC_FINAL"
    if speaker_changed:
        return True, "L_INTERACTION_TURN"
    return False, "CONTINUE"


def build_luus(df_word: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build heuristic LUUs from word timing and part-of-speech information.
    """
    if df_word.empty:
        empty_luu = pd.DataFrame(
            columns=["filename", "speaker", "tier", "luuID", "startTime", "endTime", "luu", "boundaryType", "ipuCount", "wordCount"]
        )
        empty_word_map = pd.DataFrame(
            columns=["filename", "speaker", "wordID", "ipuID", "luuID", "startTime", "endTime", "orth", "nth", "len"]
        )
        return empty_luu, empty_word_map

    df_word_sorted = df_word.sort_values(["startTime", "endTime", "wordID"]).reset_index(drop=True)
    df_ipu_summary = _summarize_ipus_from_words(df_word_sorted)

    luu_rows = []
    word2luu_rows = []
    current_ipu_ids: list[str] = []
    current_word_rows: list[pd.DataFrame] = []
    current_text_parts: list[str] = []

    word_groups = {ipu_id: group.copy() for ipu_id, group in df_word_sorted.groupby("ipuID", sort=False)}

    for index, ipu_summary in df_ipu_summary.iterrows():
        ipu_id = ipu_summary["ipuID"]
        current_words = word_groups[ipu_id]
        current_ipu_ids.append(ipu_id)
        current_word_rows.append(current_words)
        current_text_parts.append(_normalize(ipu_summary["ipu"]))

        next_summary = df_ipu_summary.iloc[index + 1] if index + 1 < len(df_ipu_summary) else None
        next_words = word_groups[next_summary["ipuID"]] if next_summary is not None else None
        should_close, boundary_type = _decide_boundary(current_words, next_words, ipu_summary, next_summary)

        if not should_close:
            continue

        luu_words = pd.concat(current_word_rows, ignore_index=True)
        speaker = _normalize(luu_words.iloc[0]["speaker"])
        valid_starts = luu_words["startTime"].dropna()
        valid_ends = luu_words["endTime"].dropna()
        luu_start = valid_starts.min() if not valid_starts.empty else pd.NA
        luu_end = valid_ends.max() if not valid_ends.empty else pd.NA
        luu_id = f"{float_to_timecode(luu_start)}{speaker}" if pd.notna(luu_start) else f"unknown{speaker}{len(luu_rows)+1}"
        luu_text = "".join(part for part in current_text_parts if part)

        luu_rows.append(
            {
                "filename": luu_words.iloc[0]["filename"],
                "speaker": speaker,
                "tier": f"LUU_{speaker}",
                "luuID": luu_id,
                "startTime": luu_start,
                "endTime": luu_end,
                "luu": luu_text,
                "boundaryType": boundary_type,
                "ipuCount": len(current_ipu_ids),
                "wordCount": len(luu_words),
            }
        )

        for _, word_row in luu_words.iterrows():
            word2luu_rows.append(
                {
                    "filename": word_row["filename"],
                    "speaker": word_row["speaker"],
                    "wordID": word_row["wordID"],
                    "ipuID": word_row["ipuID"],
                    "luuID": luu_id,
                    "startTime": word_row["startTime"],
                    "endTime": word_row["endTime"],
                    "orth": word_row["orth"],
                    "nth": word_row["nth"],
                    "len": word_row["len"],
                }
            )

        current_ipu_ids = []
        current_word_rows = []
        current_text_parts = []

    df_luu = pd.DataFrame(luu_rows)
    df_word2luu = pd.DataFrame(word2luu_rows)
    return df_luu, df_word2luu


def build_luus_from_word_csv(input_csv_path: str | Path, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame | Path]:
    source = Path(input_csv_path).expanduser().resolve()
    df_word = pd.read_csv(source)
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    df_luu, df_word2luu = build_luus(df_word)
    luu_csv = target_dir / "luu.csv"
    word2luu_csv = target_dir / "word2luu.csv"

    df_luu.to_csv(luu_csv, encoding="utf-8_sig", index=False)
    df_word2luu.to_csv(word2luu_csv, encoding="utf-8_sig", index=False)

    return {
        "luu_csv": luu_csv,
        "word2luu_csv": word2luu_csv,
        "luu_df": df_luu,
        "word2luu_df": df_word2luu,
    }
