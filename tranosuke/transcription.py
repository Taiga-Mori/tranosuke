import os
import inspect
from collections.abc import Callable
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from tranosuke.config import detect_device, get_app_paths, read_user_config
from tranosuke.media import convert_media_to_wavs
from tranosuke.utils import float_to_timecode


ProgressCallback = Callable[[float, str], None]
DEFAULT_PAUSE_THRESHOLD_MS = 200
SILENCE_THRESHOLD_DB = 35.0
MIN_SPEECH_CHUNK_S = 0.05


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


def merge_consecutive_turns(speaker_diarization, max_gap_s: float = 0.0) -> list[tuple[Segment, str]]:
    speaker_turns = sorted(
        list(speaker_diarization.itertracks(yield_label=True)),
        key=lambda item: item[0].start,
    )
    if not speaker_turns:
        return []

    merged_turns = []
    prev_turn, _, prev_speaker = speaker_turns[0]
    for turn, _, speaker in speaker_turns[1:]:
        gap = float(turn.start) - float(prev_turn.end)
        if speaker == prev_speaker and gap <= max_gap_s:
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


def _resolve_whisper_device(device: str) -> str:
    """Map unsupported Torch devices to faster-whisper compatible devices."""
    if device == "mps":
        return "cpu"
    return device


def _prepare_runtime_device(device: str | None, device_index: int | None) -> tuple[str, int | None]:
    paths = get_app_paths()
    runtime_device = device or paths.device
    if runtime_device != "cuda":
        return runtime_device, None

    if detect_device() != "cuda":
        return "cpu", None

    import torch

    device_count = torch.cuda.device_count()
    index = 0 if device_index is None else int(device_index)
    if index < 0 or index >= device_count:
        return "cpu", None

    torch.cuda.set_device(index)
    torch.empty(1, device=f"cuda:{index}")
    return "cuda", index


def _patch_hf_hub_download_auth_keyword() -> None:
    import huggingface_hub
    import pyannote.audio.core.inference as pyannote_inference
    import pyannote.audio.core.model as pyannote_model
    import pyannote.audio.core.pipeline as pyannote_pipeline
    import pyannote.audio.pipelines.speaker_verification as pyannote_speaker_verification

    original = huggingface_hub.hf_hub_download
    if "use_auth_token" in inspect.signature(original).parameters:
        return
    if getattr(original, "_tranosuke_auth_keyword_compat", False):
        return

    def compatible_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original(*args, **kwargs)

    compatible_hf_hub_download._tranosuke_auth_keyword_compat = True
    huggingface_hub.hf_hub_download = compatible_hf_hub_download
    for module in [
        pyannote_inference,
        pyannote_model,
        pyannote_pipeline,
        pyannote_speaker_verification,
    ]:
        if hasattr(module, "hf_hub_download"):
            module.hf_hub_download = compatible_hf_hub_download


def _allow_pyannote_torch_checkpoint_globals() -> None:
    import torch
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Problem, Resolution, Specifications])


def _load_diarization_pipeline(token: str, device: str, device_index: int | None = None) -> Pipeline:
    _patch_hf_hub_download_auth_keyword()
    _allow_pyannote_torch_checkpoint_globals()
    kwargs = {}
    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    if "token" in parameters:
        kwargs["token"] = token
    else:
        kwargs["use_auth_token"] = token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", **kwargs)

    if device == "cuda":
        import torch

        pipeline.to(torch.device(f"cuda:{device_index or 0}"))
    return pipeline


def _detect_non_silent_chunks(
    audio_segment: np.ndarray,
    sample_rate: int,
    absolute_start: float,
    min_silence_s: float = DEFAULT_PAUSE_THRESHOLD_MS / 1000.0,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> list[tuple[float, float]]:
    if audio_segment.size == 0:
        return []

    audio = audio_segment.astype("float32", copy=False)
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return []

    frame_length = max(int(0.02 * sample_rate), 1)
    hop_length = max(int(0.01 * sample_rate), 1)
    threshold = peak * (10 ** (-silence_threshold_db / 20.0))

    speech_frames = []
    for start_sample in range(0, len(audio), hop_length):
        end_sample = min(start_sample + frame_length, len(audio))
        frame = audio[start_sample:end_sample]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms > threshold:
            speech_frames.append((start_sample, end_sample))

    if not speech_frames:
        return [(absolute_start, absolute_start + len(audio) / sample_rate)]

    intervals = []
    current_start, current_end = speech_frames[0]
    min_silence_samples = int(min_silence_s * sample_rate)
    min_speech_samples = int(MIN_SPEECH_CHUNK_S * sample_rate)

    for start_sample, end_sample in speech_frames[1:]:
        if start_sample - current_end < min_silence_samples:
            current_end = max(current_end, end_sample)
        else:
            if current_end - current_start >= min_speech_samples:
                intervals.append((current_start, current_end))
            current_start, current_end = start_sample, end_sample

    if current_end - current_start >= min_speech_samples:
        intervals.append((current_start, current_end))

    if not intervals:
        return [(absolute_start, absolute_start + len(audio) / sample_rate)]

    return [
        (
            absolute_start + start_sample / sample_rate,
            absolute_start + min(end_sample, len(audio)) / sample_rate,
        )
        for start_sample, end_sample in intervals
    ]


def _merge_speech_chunks(
    speech_chunks: list[dict[str, float | str]],
    max_gap_s: float,
) -> list[dict[str, float | str]]:
    sorted_chunks = sorted(speech_chunks, key=lambda item: (float(item["start"]), float(item["end"])))
    if not sorted_chunks:
        return []

    merged = [dict(sorted_chunks[0])]
    for chunk in sorted_chunks[1:]:
        previous = merged[-1]
        gap = float(chunk["start"]) - float(previous["end"])
        if chunk["speaker"] == previous["speaker"] and gap <= max_gap_s:
            previous["end"] = max(float(previous["end"]), float(chunk["end"]))
        else:
            merged.append(dict(chunk))
    return merged


def _transcribe_segment_text(
    model: WhisperModel,
    tmp_path: str,
    beam_size: int,
    language: str,
) -> str:
    segments, _ = model.transcribe(
        tmp_path,
        beam_size=beam_size,
        word_timestamps=False,
        language=language,
        initial_prompt=DEFAULT_FILLER_PROMPT,
    )
    return "".join(segment.text.strip() for segment in segments if segment.text).strip()


def _is_cuda_device_ordinal_error(error: RuntimeError) -> bool:
    message = str(error)
    return "cudaErrorInvalidDevice" in message or "invalid device ordinal" in message


def _report_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, value)), message)


def transcribe_ipus(
    audio_path: str | Path,
    model_name: str = "turbo",
    pause_threshold_ms: int = DEFAULT_PAUSE_THRESHOLD_MS,
    beam_size: int = 5,
    language: str = "ja",
    device: str | None = None,
    device_index: int | None = None,
    segment_buffer_s: float = 0.1,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Perform diarization and whisper transcription, then group words into IPUs.
    """
    source = Path(audio_path).expanduser().resolve()
    _report_progress(progress_callback, 0.01, "音声を読み込んでいます")
    waveform, sample_rate = torchaudio.load(str(source))
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    wav = waveform.squeeze(0).numpy()

    _report_progress(progress_callback, 0.05, "モデルを準備しています")
    runtime_device, device_index = _prepare_runtime_device(device, device_index)
    whisper_device = _resolve_whisper_device(runtime_device)
    whisper_kwargs = {"device": whisper_device}
    if whisper_device == "cuda" and device_index is not None:
        whisper_kwargs["device_index"] = device_index
    model = WhisperModel(model_name, **whisper_kwargs)
    fallback_cpu_model = None
    pipeline = _load_diarization_pipeline(_load_huggingface_token(), runtime_device, device_index)

    _report_progress(progress_callback, 0.12, "話者分離を実行しています")
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

    ipus = []
    pause_threshold_s = max(pause_threshold_ms, 0) / 1000.0
    merged_turns = merge_consecutive_turns(diarization.speaker_diarization, max_gap_s=pause_threshold_s)
    total_turns = max(len(merged_turns), 1)
    _report_progress(progress_callback, 0.25, f"話者区間を処理しています 0/{len(merged_turns)}")

    speech_chunks = []
    for turn_index, (turn, speaker_id) in enumerate(merged_turns, start=1):
        turn_start = float(turn.start)
        turn_end = float(turn.end)
        previous_turns = [
            other_turn
            for other_turn, _ in merged_turns
            if float(other_turn.end) <= turn_start
        ]
        next_turns = [
            other_turn
            for other_turn, _ in merged_turns
            if float(other_turn.start) >= turn_end
        ]
        previous_end = max((float(other_turn.end) for other_turn in previous_turns), default=0.0)
        next_start = min(
            (float(other_turn.start) for other_turn in next_turns),
            default=len(wav) / sample_rate,
        )
        detection_start = max(turn_start - segment_buffer_s, previous_end, 0.0)
        detection_end = min(turn_end + segment_buffer_s, next_start, len(wav) / sample_rate)
        start_sample = max(int(detection_start * sample_rate), 0)
        end_sample = min(int(detection_end * sample_rate), len(wav))
        detection_audio = wav[start_sample:end_sample]
        speaker = _speaker_label_to_name(speaker_id)

        chunks = _detect_non_silent_chunks(
            detection_audio,
            sample_rate,
            detection_start,
            min_silence_s=pause_threshold_s,
        )
        chunks = [(start, end) for start, end in chunks if end > start]
        for speech_start, speech_end in chunks:
            speech_chunks.append(
                {
                    "speaker": speaker,
                    "speaker_id": speaker_id,
                    "start": speech_start,
                    "end": speech_end,
                }
            )

        _report_progress(
            progress_callback,
            0.25 + 0.20 * (turn_index / total_turns),
            f"無音区間を検出しています {turn_index}/{len(merged_turns)}",
        )

    speech_chunks = _merge_speech_chunks(speech_chunks, pause_threshold_s)
    total_chunks = max(len(speech_chunks), 1)
    for chunk_index, chunk in enumerate(speech_chunks, start=1):
        speech_start = float(chunk["start"])
        speech_end = float(chunk["end"])
        previous_chunks = [
            other_chunk
            for other_chunk in speech_chunks
            if other_chunk is not chunk and float(other_chunk["end"]) <= speech_start
        ]
        next_chunks = [
            other_chunk
            for other_chunk in speech_chunks
            if other_chunk is not chunk and float(other_chunk["start"]) >= speech_end
        ]
        previous_chunk_end = max(
            (float(other_chunk["end"]) for other_chunk in previous_chunks),
            default=0.0,
        )
        next_chunk_start = min(
            (float(other_chunk["start"]) for other_chunk in next_chunks),
            default=len(wav) / sample_rate,
        )
        transcribe_start = max(speech_start - segment_buffer_s, previous_chunk_end, 0.0)
        transcribe_end = min(speech_end + segment_buffer_s, next_chunk_start, len(wav) / sample_rate)
        chunk_start_sample = max(int(transcribe_start * sample_rate), 0)
        chunk_end_sample = min(int(transcribe_end * sample_rate), len(wav))
        chunk_audio = wav[chunk_start_sample:chunk_end_sample]
        if chunk_audio.size == 0:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            tmp_path = tmp_wav_file.name

        try:
            sf.write(tmp_path, chunk_audio, sample_rate)
            try:
                text = _transcribe_segment_text(model, tmp_path, beam_size, language)
            except RuntimeError as error:
                if whisper_device != "cuda" or not _is_cuda_device_ordinal_error(error):
                    raise
                if fallback_cpu_model is None:
                    fallback_cpu_model = WhisperModel(model_name, device="cpu")
                text = _transcribe_segment_text(fallback_cpu_model, tmp_path, beam_size, language)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        text = _normalize_ipu_text([text])
        if not text:
            continue

        speaker = str(chunk["speaker"])
        ipus.append(
            {
                "filename": source.stem,
                "speaker": speaker,
                "tier": f"IPU_{speaker}",
                "IPUID": f"{float_to_timecode(speech_start)}{speaker}",
                "startTime": round(speech_start, 3),
                "endTime": round(speech_end, 3),
                "IPU": text,
            }
        )

        _report_progress(
            progress_callback,
            0.45 + 0.53 * (chunk_index / total_chunks),
            f"書き起こしています {chunk_index}/{len(speech_chunks)}",
        )

    _report_progress(progress_callback, 1.0, "IPU書き起こしが完了しました")
    return pd.DataFrame(ipus)


def transcribe_media_to_ipu_csv(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = "turbo",
    beam_size: int = 5,
    pause_threshold_ms: int = DEFAULT_PAUSE_THRESHOLD_MS,
    device: str | None = None,
    device_index: int | None = None,
    segment_buffer_s: float = 0.1,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Convert media to wav if needed, transcribe it, and save IPU.csv.
    """
    source = Path(input_path).expanduser().resolve()
    _report_progress(progress_callback, 0.0, "wavへ変換しています")
    conversion = convert_media_to_wavs(source, output_dir=output_dir, split_channels=False)

    def transcribe_progress(value: float, message: str) -> None:
        _report_progress(progress_callback, 0.1 + 0.85 * value, message)

    df_ipu = transcribe_ipus(
        conversion.mixed_mono_wav,
        model_name=model_name,
        beam_size=beam_size,
        pause_threshold_ms=pause_threshold_ms,
        device=device,
        device_index=device_index,
        segment_buffer_s=segment_buffer_s,
        progress_callback=transcribe_progress,
    )
    csv_path = conversion.mixed_mono_wav.parent / "IPU.csv"
    _report_progress(progress_callback, 0.98, "CSVへ保存しています")
    df_ipu.to_csv(csv_path, encoding="utf-8_sig", index=False)
    _report_progress(progress_callback, 1.0, "IPU書き起こしCSVを保存しました")
    return csv_path, df_ipu
