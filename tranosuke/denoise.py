import logging
import os
import subprocess
from pathlib import Path

from tranosuke.config import get_app_paths
from tranosuke.media import convert_media_to_wavs


def denoise_wav(input_wav_path: str | Path) -> Path:
    """
    Run MediaProcessor on a wav file and return the processed wav path.
    """
    input_path = Path(input_wav_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input wav file not found: {input_path}")
    if input_path.suffix.lower() != ".wav":
        raise ValueError(f"Input file must be wav: {input_path}")

    paths = get_app_paths()
    if not paths.media_processor_path.exists():
        raise FileNotFoundError(f"MediaProcessor not found: {paths.media_processor_path}")
    if not paths.deep_filter_path.exists():
        raise FileNotFoundError(f"DeepFilterNet path not found: {paths.deep_filter_path}")

    env = os.environ.copy()
    env["DEEPFILTERNET_PATH"] = str(paths.deep_filter_path)

    logging.info("Running MediaProcessor on: %s", input_path)
    result = subprocess.run(
        [str(paths.media_processor_path), str(input_path)],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "MediaProcessor failed.\n"
            f"Return code: {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )

    processed_path = None
    for line in result.stdout.splitlines():
        if "Audio processed successfully" in line:
            processed_path = line.split(": ", 1)[1].strip().strip('"')
            break

    if processed_path is None:
        raise RuntimeError(
            "Processed wav path not found in MediaProcessor output.\n"
            f"stdout:\n{result.stdout}"
        )

    resolved_processed_path = Path(processed_path).expanduser().resolve()
    if not resolved_processed_path.exists():
        raise RuntimeError(f"Processed wav file not found: {resolved_processed_path}")

    return resolved_processed_path


def denoise_media(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Convert any supported media file to mono wav if needed, then denoise it.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {source}")

    if source.suffix.lower() == ".wav":
        return denoise_wav(source)

    conversion = convert_media_to_wavs(source, output_dir=output_dir, split_channels=False)
    return denoise_wav(conversion.mixed_mono_wav)
