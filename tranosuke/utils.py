import tarfile
import zipfile
import shutil
from pathlib import Path

import pandas as pd
import requests


DEFAULT_HEADERS = {"User-Agent": "tranosuke"}


def download_file(url: str, destination: str | Path) -> Path:
    """Download a file if it does not already exist."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return target

    response = requests.get(url, stream=True, timeout=120, headers=DEFAULT_HEADERS)
    response.raise_for_status()

    with open(target, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    return target


def download_json(url: str) -> dict:
    response = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    return response.json()


def extract_archive(archive_path: str | Path, destination_dir: str | Path) -> Path:
    archive = Path(archive_path).expanduser().resolve()
    target_dir = Path(destination_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zip_file:
            zip_file.extractall(target_dir)
        return target_dir

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar_file:
            tar_file.extractall(target_dir)
        return target_dir

    raise ValueError(f"Unsupported archive format: {archive}")


def download_and_extract(url: str, destination_dir: str | Path, archive_name: str | None = None) -> Path:
    target_dir = Path(destination_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.iterdir()):
        return target_dir

    file_name = archive_name or Path(url).name or "downloaded.archive"
    archive_path = target_dir.parent / file_name
    download_file(url, archive_path)
    if zipfile.is_zipfile(archive_path) or tarfile.is_tarfile(archive_path):
        extract_archive(archive_path, target_dir)
    else:
        target_path = target_dir / archive_path.name
        shutil.move(str(archive_path), target_path)
        return target_dir
    archive_path.unlink(missing_ok=True)
    return target_dir


DEFAULT_TIMECODE_INTEGER_DIGITS = 3


def timecode_integer_digits(duration_s: float) -> int:
    """Return how many integer-second digits IDs need to stay unique within a recording of this length."""
    integer_part = f"{max(float(duration_s), 0.0):.3f}".split(".")[0]
    return max(DEFAULT_TIMECODE_INTEGER_DIGITS, len(integer_part))


def float_to_timecode(value: float, integer_digits: int = DEFAULT_TIMECODE_INTEGER_DIGITS) -> str | None:
    """Convert seconds into a zero-padded ID chunk: integer_digits of seconds + 3 digits of milliseconds."""
    if value is None:
        return None

    width = integer_digits + 3
    text = f"{value:.3f}"
    if "." not in text:
        return text.zfill(width)

    integer_part, fraction_part = text.split(".")
    if len(integer_part) > integer_digits:
        raise ValueError(f"{value} 秒は {integer_digits} 桁の秒数に収まりません。")
    return integer_part.zfill(integer_digits) + fraction_part


def adjust_ipu_time(df_ipu: pd.DataFrame, df_phon: pd.DataFrame) -> pd.DataFrame:
    """Replace rough IPU timings with the first and last aligned phoneme timings."""
    if df_phon.empty:
        return df_ipu.copy()

    phoneme_ranges = (
        df_phon.groupby("IPUID")
        .agg(startTime_phon=("startTime", "min"), endTime_phon=("endTime", "max"))
        .reset_index()
    )

    adjusted = pd.merge(df_ipu, phoneme_ranges, on="IPUID", how="left")
    adjusted["startTime"] = adjusted["startTime_phon"].combine_first(adjusted["startTime"])
    adjusted["endTime"] = adjusted["endTime_phon"].combine_first(adjusted["endTime"])
    return adjusted.drop(columns=["startTime_phon", "endTime_phon"])
