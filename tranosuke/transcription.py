import os
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf
import torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from tranosuke.config import get_app_paths, read_user_config
from tranosuke.media import convert_media_to_wavs
from tranosuke.utils import float_to_timecode


DEFAULT_FILLER_PROMPT = (
    "あっの、あの、あの〜、あのぅ、あのう、あのぉ、あのぉ〜、あのお、あのー、"
    "あんの、あんのー、あーの、あーのー、あーんのー、あ、あぁ〜、あん、あー、"
    "あっと、あっとー、あとー、あんと、あんーっと、あーっと、あーと、あーとー、"
    "あーんと、あーんーと、い、いー、ううんっと、うんっと、うんと、うんとー、"
    "うんーと、うーんっと、うーんと、うーんとー、んっと、んっとー、んと、んとー、"
    "んーっと、んーっとー、んーと、んーとー、う、うー、うーっと、うーと、えいと、"
    "え、え〜、えええ、えー、え〜と、ええっと、ええっとっ、ええっとー、ええと、"
    "ええとー、ええーと、えっと、えっとお、えっとー、えと、えとー、えーっと、"
    "えーっとっ、えーっとー、えーと、えーとー、お、おー、こうと、そっの、その、"
    "そのう、そのー、そん、そんの、そーの、そーのー、っと、と、とー、ま、まー、"
    "ん、ん〜、んー"
)


def merge_consecutive_turns(speaker_diarization) -> list[tuple[Segment, str]]:
    speaker_turns = sorted(
        list(speaker_diarization.itertracks(yield_label=True)),
        key=lambda item: item[0].start,
    )
    if not speaker_turns:
        return []

    merged_turns = []
    prev_turn, _, prev_speaker = speaker_turns[0]
    for turn, _, speaker in speaker_turns[1:]:
        if speaker == prev_speaker:
            prev_turn = Segment(prev_turn.start, max(prev_turn.end, turn.end))
        else:
            merged_turns.append((prev_turn, prev_speaker))
            prev_turn, prev_speaker = turn, speaker
    merged_turns.append((prev_turn, prev_speaker))
    return merged_turns


def _speaker_label_to_name(speaker_id: str) -> str:
    suffix = speaker_id.split("_")[-1]
    index = int(suffix) if suffix.isdigit() else 0
    return chr(ord("A") + index)


def _load_huggingface_token() -> str:
    token = read_user_config().get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        raise ValueError("Hugging Face のアクセストークンが未設定です。")
    return token


def _normalize_ipu_text(words: list[str]) -> str:
    text = "".join(word.strip() for word in words if word)
    for char in [" ", "、", "。", ",", ".", ":", ";", "〜"]:
        text = text.replace(char, "")
    return text


def transcribe_ipus(
    audio_path: str | Path,
    model_name: str = "turbo",
    max_pause_ms: int = 200,
    beam_size: int = 5,
    language: str = "ja",
) -> pd.DataFrame:
    """
    Perform diarization and whisper transcription, then group words into IPUs.
    """
    source = Path(audio_path).expanduser().resolve()
    wav, sample_rate = torchaudio.load(str(source))
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0).numpy()

    paths = get_app_paths()
    model = WhisperModel(model_name, device=paths.device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=_load_huggingface_token(),
    )

    with ProgressHook() as hook:
        diarization = pipeline(str(source), hook=hook)

    ipus = []
    max_gap_s = max_pause_ms / 1000.0

    for turn, speaker_id in merge_consecutive_turns(diarization.speaker_diarization):
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        segment_audio = wav[start_sample:end_sample]
        speaker = _speaker_label_to_name(speaker_id)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            tmp_path = tmp_wav_file.name

        try:
            sf.write(tmp_path, segment_audio, sample_rate)
            segments, _ = model.transcribe(
                tmp_path,
                beam_size=beam_size,
                word_timestamps=True,
                language=language,
                initial_prompt=DEFAULT_FILLER_PROMPT,
            )

            words = []
            for segment in segments:
                for word_info in getattr(segment, "words", []):
                    start = float(word_info["start"] if isinstance(word_info, dict) else word_info.start)
                    end = float(word_info["end"] if isinstance(word_info, dict) else word_info.end)
                    word = (word_info["word"] if isinstance(word_info, dict) else word_info.word).strip()
                    if word:
                        words.append(
                            {"start": start + turn.start, "end": end + turn.start, "word": word}
                        )

            if not words:
                continue

            current_words = [words[0]["word"]]
            current_start = words[0]["start"]
            current_end = words[0]["end"]
            previous_end = current_end

            for word in words[1:]:
                gap = word["start"] - previous_end
                if gap <= max_gap_s:
                    if gap >= 0.1:
                        current_words.append("¥")
                    current_words.append(word["word"])
                    current_end = word["end"]
                else:
                    ipus.append(
                        {
                            "filename": source.stem,
                            "speaker": speaker,
                            "tier": f"IPU_{speaker}",
                            "ipuID": f"{float_to_timecode(current_start)}{speaker}",
                            "startTime": round(current_start, 3),
                            "endTime": round(current_end, 3),
                            "ipu": _normalize_ipu_text(current_words),
                        }
                    )
                    current_words = [word["word"]]
                    current_start = word["start"]
                    current_end = word["end"]
                previous_end = word["end"]

            ipus.append(
                {
                    "filename": source.stem,
                    "speaker": speaker,
                    "tier": f"IPU_{speaker}",
                    "ipuID": f"{float_to_timecode(current_start)}{speaker}",
                    "startTime": round(current_start, 3),
                    "endTime": round(current_end, 3),
                    "ipu": _normalize_ipu_text(current_words),
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return pd.DataFrame(ipus)


def transcribe_media_to_ipu_csv(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = "turbo",
    beam_size: int = 5,
) -> tuple[Path, pd.DataFrame]:
    """
    Convert media to wav if needed, transcribe it, and save ipu.csv.
    """
    source = Path(input_path).expanduser().resolve()
    conversion = convert_media_to_wavs(source, output_dir=output_dir, split_channels=False)
    df_ipu = transcribe_ipus(
        conversion.mixed_mono_wav,
        model_name=model_name,
        beam_size=beam_size,
    )
    csv_path = conversion.mixed_mono_wav.parent / "ipu.csv"
    df_ipu.to_csv(csv_path, encoding="utf-8_sig", index=False)
    return csv_path, df_ipu
