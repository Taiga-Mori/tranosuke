import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from tranosuke.utils import download


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    cache_dir: Path
    config_path: Path
    media_processor_path: Path
    deep_filter_path: Path
    system: str
    device: str


def detect_device() -> str:
    """Return the best available Torch device without requiring Torch at import time."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_base_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(os.path.abspath("."))


def get_app_paths() -> AppPaths:
    base_dir = get_base_dir()
    cache_dir = Path.home() / ".tranosuke"
    return AppPaths(
        base_dir=base_dir,
        cache_dir=cache_dir,
        config_path=cache_dir / "config.yaml",
        media_processor_path=base_dir / "fast-music-remover/MediaProcessor/build/MediaProcessor",
        deep_filter_path=base_dir / "fast-music-remover/MediaProcessor/res/deep-filter-0.5.6-x86_64-unknown-linux-musl",
        system=platform.system(),
        device=detect_device(),
    )


def initialize_app() -> AppPaths:
    """Prepare cache files and an empty config file."""
    paths = get_app_paths()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)

    download(
        paths.cache_dir / "unidic-csj-202302",
        "https://clrd.ninjal.ac.jp/unidic_archive/2302/unidic-csj-202302.zip",
    )
    download(
        paths.cache_dir / "phoneme_transition_model.onnx",
        "https://github.com/DwangoMediaVillage/pydomino/raw/main/onnx_model/phoneme_transition_model.onnx",
    )

    if not paths.config_path.exists():
        with open(paths.config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump({}, file, allow_unicode=True, sort_keys=False)

    return paths


def read_user_config() -> dict:
    paths = get_app_paths()
    if not paths.config_path.exists():
        return {}
    with open(paths.config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_user_config(config: dict) -> None:
    paths = get_app_paths()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def save_huggingface_token(token: str) -> None:
    config = read_user_config()
    config["HUGGINGFACE_ACCESS_TOKEN"] = token
    write_user_config(config)
