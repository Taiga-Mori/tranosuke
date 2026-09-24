import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from tranosuke.config import get_app_paths


@dataclass(frozen=True)
class MediaConversionResult:
    input_path: Path
    mixed_mono_wav: Path
    channel_wavs: list[Path]


def _resolve_media_tool(command_name: str) -> str:
    paths = get_app_paths()
    candidates = []
    if paths.system == "Darwin":
        candidates.append(paths.base_dir / "ffmpeg" / "mac" / command_name)
    elif paths.system == "Windows":
        candidates.append(paths.base_dir / "ffmpeg" / "win" / f"{command_name}.exe")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which(command_name)
    if resolved:
        return resolved

    raise FileNotFoundError(f"{command_name} が見つかりません。")


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _probe_audio_channels(input_path: Path) -> int:
    """Count audio channels with ffmpeg only, so ffprobe does not need to be bundled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        probe_wav = Path(temp_dir) / "probe.wav"
        command = [
            _resolve_media_tool("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-t",
            "1",
            "-acodec",
            "pcm_s16le",
            str(probe_wav),
        ]
        try:
            _run_subprocess(command)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"音声ストリームが見つかりません: {input_path}\n{error.stderr}") from error
        return sf.info(str(probe_wav)).channels


def _convert_to_mono(input_path: Path, output_path: Path, sample_rate: int) -> Path:
    command = [
        _resolve_media_tool("ffmpeg"),
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    _run_subprocess(command)
    return output_path


def _extract_single_channel(input_path: Path, output_path: Path, channel_index: int, sample_rate: int) -> Path:
    command = [
        _resolve_media_tool("ffmpeg"),
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        f"pan=mono|c0=c{channel_index}",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    _run_subprocess(command)
    return output_path


def convert_media_to_wavs(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    sample_rate: int = 16000,
    split_channels: bool = True,
) -> MediaConversionResult:
    """Convert a media file into a mixed mono wav and optional per-channel mono wavs."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {source}")

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else source.with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)

    mixed_mono_wav = target_dir / f"{source.stem}_mono.wav"
    channel_count = _probe_audio_channels(source)
    _convert_to_mono(source, mixed_mono_wav, sample_rate)

    channel_wavs: list[Path] = []
    if split_channels and channel_count > 1:
        for channel_index in range(channel_count):
            channel_output = target_dir / f"{source.stem}_ch{channel_index + 1}.wav"
            channel_wavs.append(_extract_single_channel(source, channel_output, channel_index, sample_rate))

    return MediaConversionResult(
        input_path=source,
        mixed_mono_wav=mixed_mono_wav,
        channel_wavs=channel_wavs,
    )
