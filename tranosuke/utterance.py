import os
from faster_whisper import WhisperModel
from pathlib import Path, PosixPath
import torchaudio
import pandas as pd
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
import tempfile
import soundfile as sf
import yaml

from init import *
from utils import *



def merge_consecutive_turns(speaker_diarization):
    """
    同じ話者が連続する区間をまとめる
    """
    merged_turns = []

    # タイムライン順に取得（yield_label=True で speaker 情報を含む）
    speaker_turns = sorted(
        list(speaker_diarization.itertracks(yield_label=True)),
        key=lambda x: x[0].start
    )

    if not speaker_turns:
        return merged_turns

    prev_turn, _, prev_speaker = speaker_turns[0]

    for turn, _, speaker in speaker_turns[1:]:
        # 同一話者が直後に続く場合、区間を拡張
        if speaker == prev_speaker:
            prev_turn = Segment(prev_turn.start, max(prev_turn.end, turn.end))
        else:
            merged_turns.append((prev_turn, prev_speaker))
            prev_turn, prev_speaker = turn, speaker

    merged_turns.append((prev_turn, prev_speaker))
    return merged_turns



def transcribe_utterance(
    audio_path: Path,
    model_name: str = "turbo",
    max_gap_ms: int = 200,
    device: str = "cpu",
    beam_size: int = 5,
) -> pd.DataFrame:
    """
    fast-whisperで単語を書き起こし、話者埋め込みとクラスタリングによって
    話者ごとに0.2秒以内の単語を結合して発話化する。

    Returns:
        DataFrame(["filename", "tier", "utteranceID", "startTime", "endTime", "speaker", "utterance"])
    """
    audio_path = Path(audio_path)
    filename = audio_path.stem

    # 1. 音声読み込み
    wav, sr = torchaudio.load(str(audio_path))
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0).numpy()

    # 2. Whisperモデル
    model = WhisperModel(model_name, device=device)

    # 3. Pyannote話者分離
    with open(CONFIG_PATH, encoding='utf-8')as f:
            config = yaml.safe_load(f)
    HUGGINGFACE_ACCESS_TOKEN = config["HUGGINGFACE_ACCESS_TOKEN"]

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HUGGINGFACE_ACCESS_TOKEN
    )

    with ProgressHook() as hook:
        diarization = pipeline(str(audio_path), hook=hook)

    merged_turns = merge_consecutive_turns(diarization.speaker_diarization)

    utterances = []

    # 4. 各話者区間ごとに単語認識
    for turn, speakerID in merged_turns:
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        num = int(speakerID[-2:])
        speaker = chr(ord("A") + num)

        segment_audio = wav[start_sample:end_sample]
        
        # 一時ファイルで書き出し
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            tmp_path = tmp_wav_file.name
            sf.write(tmp_path, segment_audio, sr)

        # Whisper で単語単位認識
        segments, _ = model.transcribe(
            tmp_wav_file.name,
            beam_size=beam_size,
            word_timestamps=True,
            language="ja",
            initial_prompt="あっの、あの、あの〜、あのぅ、あのう、あのぉ、あのぉ〜、あのお、あのー、あんの、あんのー、あーの、あーのー、あーんのー、あ、あぁ〜、あん、あー、あっと、あっとー、あとー、あんと、あんーっと、あーっと、あーと、あーとー、あーんと、あーんーと、い、いー、ううんっと、うんっと、うんと、うんとー、うんーと、うーんっと、うーんと、うーんとー、んっと、んっとー、んと、んとー、んーっと、んーっとー、んーと、んーとー、う、うー、うーっと、うーと、えいと、え、え〜、えええ、えー、え〜と、ええっと、ええっとっ、ええっとー、ええと、ええとー、ええーと、えっと、えっとお、えっとー、えと、えとー、えーっと、えーっとっ、えーっとー、えーと、えーとー、お、おー、こうと、そっの、その、そのう、そのー、そん、そんの、そーの、そーのー、っと、と、とー、ま、まー、ん、ん〜、んー",
        )

        # 一時ファイルの削除
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        words = []
        for seg in segments:
            seg_words = getattr(seg, "words", [])
            for w in seg_words:
                start = float(w["start"] if isinstance(w, dict) else w.start)
                end = float(w["end"] if isinstance(w, dict) else w.end)
                word = (w["word"] if isinstance(w, dict) else w.word).strip()
                # 音声全体タイムスタンプに合わせる
                words.append({"start": start + turn.start, "end": end + turn.start, "word": word})

        if not words:
            continue

        # 5. 単語間0.2秒以内で結合
        max_gap_s = max_gap_ms / 1000.0
        cur_words = [words[0]["word"]]
        cur_start = words[0]["start"]
        cur_end = words[0]["end"]
        prev_end = cur_end

        for w in words[1:]:
            gap = w["start"] - prev_end
            # ポーズ用の記号（最終的には消される）
            if gap <= max_gap_s:
                if gap >= 0.1:
                    cur_words.append("¥")
                cur_words.append(w["word"])
                cur_end = w["end"]
            else:
                utterances.append({
                    "filename": filename,
                    "speaker": speaker,
                    "tier": f"Utterance_{speaker}",
                    "utteranceID": f"{float_to_timecode(cur_start)}{speaker}",
                    "startTime": round(cur_start, 3),
                    "endTime": round(cur_end, 3),
                    "utterance": _clean_join(cur_words)
                })
                cur_words = [w["word"]]
                cur_start = w["start"]
                cur_end = w["end"]
            prev_end = w["end"]

        # 最後の発話を追加
        if cur_words:
            utterances.append({
                "filename": filename,
                "speaker": speaker,
                "tier": f"Utterance_{speaker}",
                "utteranceID": f"{float_to_timecode(cur_start)}{speaker}",
                "startTime": round(cur_start, 3),
                "endTime": round(cur_end, 3),
                "utterance": _clean_join(cur_words)
            })

    df = pd.DataFrame(utterances)
    return df



def _clean_join(words):
    """不要な空白や記号を除去して自然に結合"""
    text = "".join(w.strip() for w in words if w)
    for ch in [" ", "、", "。", ",", ".", ":", ";", "〜"]:
        text = text.replace(ch, "")
    return text



if __name__ == '__main__':
    df = transcribe_utterance(
            audio_path="./sample/sample.wav",
            )
    print(df)
    df.to_csv("./sample/utterance.csv", index=False)
