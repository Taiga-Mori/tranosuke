import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from tranosuke.utils import download_and_extract, download_file, download_json


UNIDIC_CSJ_URL = "https://clrd.ninjal.ac.jp/unidic_archive/2302/unidic-csj-202302.zip"
PYDOMINO_MODEL_URL = (
    "https://github.com/DwangoMediaVillage/pydomino/raw/main/onnx_model/phoneme_transition_model.onnx"
)
DEEPFILTERNET_RELEASES_API = "https://api.github.com/repos/Rikorose/DeepFilterNet/releases/latest"
# (system, arm) -> 静的ビルドの ffmpeg 配布URL。macOS/Linux は martin-riedl.de、Windows は gyan.dev。
FFMPEG_DOWNLOAD_URLS = {
    ("Darwin", True): "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip",
    ("Darwin", False): "https://ffmpeg.martin-riedl.de/redirect/latest/macos/amd64/release/ffmpeg.zip",
    ("Linux", True): "https://ffmpeg.martin-riedl.de/redirect/latest/linux/arm64/release/ffmpeg.zip",
    ("Linux", False): "https://ffmpeg.martin-riedl.de/redirect/latest/linux/amd64/release/ffmpeg.zip",
    ("Windows", False): "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    # Windows on ARM は x64 エミュレーションで動く
    ("Windows", True): "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
}


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    cache_dir: Path
    runtime_dir: Path
    models_dir: Path
    tools_dir: Path
    config_path: Path
    unidic_dir: Path
    phoneme_model_path: Path
    deepfilter_dir: Path
    deepfilter_binary_path: Path
    ffmpeg_dir: Path
    ffmpeg_binary_path: Path
    system: str
    machine: str
    device: str


def detect_device() -> str:
    """Return the best available Torch device without importing Torch at module load time."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            return "cuda"
    except Exception:
        pass

    try:
        if torch.backends.mps.is_available():
            torch.empty(1, device="mps")
            return "mps"
    except Exception:
        pass

    return "cpu"


def is_mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


def list_cuda_devices() -> list[dict[str, int | str | None]]:
    try:
        import torch
    except ModuleNotFoundError:
        return []

    try:
        if not torch.cuda.is_available():
            return []

        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_memory, total_memory = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "free_gb": round(free_memory / 1024**3, 1),
                    "total_gb": round(total_memory / 1024**3, 1),
                }
            )
        return devices
    except Exception:
        return []


def get_base_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def get_app_paths() -> AppPaths:
    base_dir = get_base_dir()
    cache_dir = Path.home() / ".tranosuke"
    runtime_dir = cache_dir / "runtime"
    models_dir = cache_dir / "models"
    tools_dir = runtime_dir / "tools"
    system = platform.system()
    machine = platform.machine().lower()

    deepfilter_binary = "deep-filter.exe" if system == "Windows" else "deep-filter"
    ffmpeg_binary = "ffmpeg.exe" if system == "Windows" else "ffmpeg"
    return AppPaths(
        base_dir=base_dir,
        cache_dir=cache_dir,
        runtime_dir=runtime_dir,
        models_dir=models_dir,
        tools_dir=tools_dir,
        config_path=cache_dir / "config.yaml",
        unidic_dir=models_dir / "unidic-csj-202302",
        phoneme_model_path=models_dir / "phoneme_transition_model.onnx",
        deepfilter_dir=tools_dir / "deepfilternet",
        deepfilter_binary_path=tools_dir / "deepfilternet" / deepfilter_binary,
        ffmpeg_dir=tools_dir / "ffmpeg",
        ffmpeg_binary_path=tools_dir / "ffmpeg" / ffmpeg_binary,
        system=system,
        machine=machine,
        device=detect_device(),
    )


def _ensure_directories(paths: AppPaths) -> None:
    for directory in [paths.cache_dir, paths.runtime_dir, paths.models_dir, paths.tools_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def _ensure_config_file(paths: AppPaths) -> None:
    if paths.config_path.exists():
        return
    with open(paths.config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump({}, file, allow_unicode=True, sort_keys=False)


def _ensure_unidic(paths: AppPaths) -> None:
    download_and_extract(UNIDIC_CSJ_URL, paths.unidic_dir)


def _ensure_phoneme_model(paths: AppPaths) -> None:
    download_file(PYDOMINO_MODEL_URL, paths.phoneme_model_path)


def _deepfilter_platform_tokens(paths: AppPaths) -> list[str]:
    if paths.system == "Darwin":
        if paths.machine in {"arm64", "aarch64"}:
            return ["deep-filter", "apple-darwin", "aarch64"]
        return ["deep-filter", "apple-darwin", "x86_64"]

    if paths.system == "Windows":
        return ["deep-filter", "windows", "x86_64"]

    if paths.system == "Linux":
        if paths.machine in {"arm64", "aarch64"}:
            return ["deep-filter", "linux", "aarch64"]
        return ["deep-filter", "linux", "x86_64"]

    raise RuntimeError(f"Unsupported platform for DeepFilterNet: {paths.system} / {paths.machine}")


def _select_deepfilter_asset(paths: AppPaths, release_data: dict) -> dict:
    assets = release_data.get("assets", [])
    tokens = _deepfilter_platform_tokens(paths)

    preferred_assets = []
    fallback_assets = []
    for asset in assets:
        name = asset.get("name", "").lower()
        if "deep-filter" not in name:
            continue
        if all(token in name for token in tokens[1:]):
            preferred_assets.append(asset)
        elif all(token in name for token in tokens[:2]):
            fallback_assets.append(asset)

    for asset in preferred_assets + fallback_assets:
        if asset.get("browser_download_url"):
            return asset

    raise RuntimeError(
        "DeepFilterNet のこの環境向け公式バイナリが見つかりませんでした。"
    )


def _locate_deepfilter_binary(search_dir: Path, system: str) -> Path:
    binary_name = "deep-filter.exe" if system == "Windows" else "deep-filter"
    candidates = [path for path in search_dir.rglob(binary_name) if path.is_file()]
    if not candidates:
        fallback_names = ["deep-filter"]
        if system == "Windows":
            fallback_names.append("deep-filter.exe")
        candidates = [
            path
            for path in search_dir.rglob("*")
            if path.is_file() and any(path.name.startswith(name) for name in fallback_names)
        ]
    if not candidates:
        raise FileNotFoundError(f"DeepFilterNet binary not found under: {search_dir}")
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def _flatten_single_nested_directory(target_dir: Path) -> None:
    children = list(target_dir.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return

    nested_dir = children[0]
    for child in nested_dir.iterdir():
        shutil.move(str(child), target_dir / child.name)
    nested_dir.rmdir()


def ensure_denoise_runtime() -> AppPaths:
    """Download the official DeepFilterNet runtime when it is needed."""
    paths = get_app_paths()
    _ensure_directories(paths)

    if paths.deepfilter_binary_path.exists():
        paths.deepfilter_binary_path.chmod(0o755)
        return paths

    if paths.deepfilter_dir.exists():
        try:
            binary_path = _locate_deepfilter_binary(paths.deepfilter_dir, paths.system)
            if binary_path != paths.deepfilter_binary_path:
                shutil.copy2(binary_path, paths.deepfilter_binary_path)
                binary_path = paths.deepfilter_binary_path
            binary_path.chmod(0o755)
            return paths
        except FileNotFoundError:
            pass

    release_data = download_json(DEEPFILTERNET_RELEASES_API)
    asset = _select_deepfilter_asset(paths, release_data)
    archive_name = asset["name"]
    download_and_extract(asset["browser_download_url"], paths.deepfilter_dir, archive_name=archive_name)
    if not paths.deepfilter_binary_path.exists():
        _flatten_single_nested_directory(paths.deepfilter_dir)
    binary_path = _locate_deepfilter_binary(paths.deepfilter_dir, paths.system)
    if binary_path != paths.deepfilter_binary_path:
        shutil.copy2(binary_path, paths.deepfilter_binary_path)
        binary_path = paths.deepfilter_binary_path
    binary_path.chmod(0o755)
    return paths


def is_runnable(executable: str | Path) -> bool:
    """Check that a binary actually runs here (e.g. an x86_64 build fails on Apple Silicon without Rosetta)."""
    try:
        subprocess.run([str(executable), "-version"], check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def ensure_ffmpeg_runtime() -> Path:
    """Download a static ffmpeg build for this OS/CPU into ~/.tranosuke (once, on first launch)."""
    paths = get_app_paths()
    _ensure_directories(paths)

    if paths.ffmpeg_binary_path.exists() and is_runnable(paths.ffmpeg_binary_path):
        return paths.ffmpeg_binary_path

    is_arm = paths.machine in {"arm64", "aarch64"}
    url = FFMPEG_DOWNLOAD_URLS.get((paths.system, is_arm))
    if url is None:
        raise RuntimeError(f"この環境向けの ffmpeg を自動取得できません: {paths.system} / {paths.machine}")

    # 壊れた・別アーキテクチャの残骸があれば取り直す
    shutil.rmtree(paths.ffmpeg_dir, ignore_errors=True)
    extract_dir = paths.tools_dir / "ffmpeg_download"
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        download_and_extract(url, extract_dir)
        binary_name = paths.ffmpeg_binary_path.name
        candidates = sorted(
            (path for path in extract_dir.rglob(binary_name) if path.is_file()),
            key=lambda path: len(path.parts),
        )
        if not candidates:
            raise FileNotFoundError(f"ダウンロードした ffmpeg が見つかりません: {extract_dir}")
        paths.ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], paths.ffmpeg_binary_path)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    paths.ffmpeg_binary_path.chmod(0o755)
    if not is_runnable(paths.ffmpeg_binary_path):
        raise RuntimeError(f"ダウンロードした ffmpeg を実行できません: {paths.ffmpeg_binary_path}")
    return paths.ffmpeg_binary_path


def initialize_app(include_denoise: bool = False) -> AppPaths:
    """Prepare user config and officially downloadable runtime assets."""
    paths = get_app_paths()
    _ensure_directories(paths)
    _ensure_config_file(paths)
    _ensure_unidic(paths)
    _ensure_phoneme_model(paths)
    ensure_ffmpeg_runtime()

    if include_denoise:
        ensure_denoise_runtime()

    return paths


def read_user_config() -> dict:
    paths = get_app_paths()
    if not paths.config_path.exists():
        return {}
    with open(paths.config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_user_config(config: dict) -> None:
    paths = get_app_paths()
    _ensure_directories(paths)
    with open(paths.config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def save_huggingface_token(token: str) -> None:
    config = read_user_config()
    config["HUGGINGFACE_ACCESS_TOKEN"] = token
    write_user_config(config)
