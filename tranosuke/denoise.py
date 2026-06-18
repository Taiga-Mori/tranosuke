import subprocess
from pathlib import Path

from tranosuke.config import ensure_denoise_runtime
from tranosuke.media import convert_media_to_wavs


def _prepare_wav_for_denoise(input_path: Path, output_dir: str | Path | None = None) -> Path:
    conversion = convert_media_to_wavs(
        input_path,
        output_dir=output_dir,
        sample_rate=48000,
        split_channels=False,
    )
    return conversion.mixed_mono_wav


def _run_deepfilter(prepared_wav: Path) -> Path:
    paths = ensure_denoise_runtime()
    target_dir = prepared_wav.parent / "denoised"
    target_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(paths.deepfilter_binary_path), "-o", str(target_dir), str(prepared_wav)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "DeepFilterNet failed.\n"
            f"Return code: {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )

    output_path = target_dir / prepared_wav.name
    if not output_path.exists():
        raise RuntimeError(
            "DeepFilterNet output file not found.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return output_path


def denoise_wav(input_wav_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Run the official DeepFilterNet CLI on a wav file."""
    source = Path(input_wav_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input wav file not found: {source}")
    if source.suffix.lower() != ".wav":
        raise ValueError(f"Input file must be wav: {source}")

    prepared_wav = _prepare_wav_for_denoise(source, output_dir=output_dir)
    return _run_deepfilter(prepared_wav)


def denoise_media(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Convert any supported media file to mono 48kHz wav, then denoise it."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {source}")

    prepared_wav = _prepare_wav_for_denoise(source, output_dir=output_dir)
    return _run_deepfilter(prepared_wav)
