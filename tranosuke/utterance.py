import os
import imageio_ffmpeg as ffmpeg

# ffmpeg実行パスを取得
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel
from pathlib import Path, PosixPath
from typing import List, Dict
import pandas as pd

try:
    from tranosuke.utils import *
except:
    from utils import *



def transcribe_utterance(
    audio_path: PosixPath,
    model_name: str = "turbo",
    max_gap_ms: int = 200,
    device: str = "cpu",
    beam_size: int = 5,
) -> pd.DataFrame:
    """
    Args:
        audio_path: 音声ファイルのパス
        model_name: faster-whisper のモデル名（例 "small", "medium"...）
        max_gap_ms: 単語間のギャップがこのミリ秒以下なら同一発話として結合
        device: "cpu" or "cuda"
        beam_size: デコーダの beam size（精度/速度トレードオフ）
    
    Returns:
        pandas.DataFrame で列 ["file", "Utterance_x", "utteranceID", "startTime", "endTime", "utterance"] （start,end は秒, float）
    """

    # 音声ファイルの名前
    audio_filename = audio_path.stem

    # モデルロード
    model = WhisperModel(model_name, device=device)

    # 音声を transcribe（word_timestamps=True で単語ごとの start/end を取得）
    segments, _ = model.transcribe(
        audio_path,
        beam_size=beam_size,
        word_timestamps=True
    )

    # 全単語を時系列で収集（segment 内の words を順に取る）
    words: List[Dict] = []
    for seg in segments:
        # seg.words は [{'start':..., 'end':..., 'word':...}, ...] のはず
        seg_words = getattr(seg, "words", None)
        if seg_words is None:
            # もし構造が dict の場合などに備えて柔軟に扱う
            seg_words = seg.get("words", []) if isinstance(seg, dict) else []
        for w in seg_words:
            # w が dict または object-like の場合に対応
            if isinstance(w, dict):
                words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]})
            else:
                # object with attributes
                words.append({"start": float(getattr(w, "start")), "end": float(getattr(w, "end")), "word": getattr(w, "word")})

    if len(words) == 0:
        # 単語が取れなかった場合は空の DataFrame を返す
        return pd.DataFrame(columns=["start", "end", "text"])

    # 単語間のギャップでグループ化
    utterances = []
    cur_words = [words[0]["word"]]
    cur_start = words[0]["start"]
    cur_end = words[0]["end"]
    prev_end = words[0]["end"]

    max_gap_s = max_gap_ms / 1000.0

    for w in words[1:]:
        gap = float(w["start"]) - float(prev_end)
        if gap <= max_gap_s:
            # 同じ発話に結合
            # gap が 0.1 秒以上なら "¥" を追加
            if gap >= 0.1:
                pause_word = {
                    "word": "¥",
                    "start": prev_end,
                    "end": w["start"]
                }
                # ¥も単語列に追加
                cur_words.append(pause_word["word"])
            # 次の単語を追加
            cur_words.append(w["word"])
            cur_end = w["end"]
        else:
            # 発話を確定して次へ
            text = _clean_join(cur_words)
            utterances.append({
                "startTime": cur_start,
                "endTime": cur_end,
                "utterance": text
            })
            # 新しい発話スタート
            cur_words = [w["word"]]
            cur_start = w["start"]
            cur_end = w["end"]

        prev_end = w["end"]

    # 最後の発話を追加
    if cur_words:
        text = _clean_join(cur_words)
        utterances.append({"startTime": cur_start, "endTime": cur_end, "utterance": text})

    df = pd.DataFrame(utterances, columns=["startTime", "endTime", "utterance"])
    df["timestamp"] = df["startTime"].apply(float_to_timecode)
    df["filename"] = audio_filename
    df["utteranceID"] = df["timestamp"].astype(str) + audio_filename
    df["tier"] = f"Utterance_{audio_filename}"
    df = df[["filename", "tier", "utteranceID", "startTime", "endTime", "utterance"]]
    return df


def _clean_join(words: List[str]) -> str:
    """
    単語リストを自然に連結する小さなヘルパー。
    whisper の出力には先頭にスペースがついた単語や
    サブトークン分割（例: "ing" が単独）などが混在する場合があるため、
    基本はスペースで結合して余分な空白を詰める処理をする。
    必要ならここを拡張して区切り記号処理や句読点の整形を行ってください。
    """
    text = "".join(w.strip() for w in words if w is not None)

    # スペースの除去
    text = text.replace(" ", "")

    # 句読点や記号の除去
    text = text.replace("、", "")
    text = text.replace("。", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("〜", "")

    return text

if __name__ == '__main__':
    print(
        transcribe_utterance(
            audio_path=Path("C:/Users/mori/Downloads/sample.mp3"),
            model_name = "large-v3"
        )
    )