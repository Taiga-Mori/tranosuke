import os

# MPS 未対応の演算があっても落ちずに CPU で計算させる（torch を import する前に設定が必要）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from tranosuke.alignment import align_phonemes_and_words
from tranosuke.config import (
    AppPaths,
    ensure_denoise_runtime,
    ensure_ffmpeg_runtime,
    get_app_paths,
    initialize_app,
    read_user_config,
    save_huggingface_token,
)
from tranosuke.corpus import build_corpus
from tranosuke.denoise import denoise_media, denoise_wav
from tranosuke.luu import build_luus, build_luus_from_word_csv
from tranosuke.media import convert_media_to_wavs
from tranosuke.morphology import analyze_ipu_csv, analyze_ipus
from tranosuke.transcription import transcribe_ipus, transcribe_media_to_ipu_csv

__all__ = [
    "AppPaths",
    "align_phonemes_and_words",
    "analyze_ipu_csv",
    "analyze_ipus",
    "build_corpus",
    "build_luus",
    "build_luus_from_word_csv",
    "convert_media_to_wavs",
    "denoise_media",
    "denoise_wav",
    "ensure_denoise_runtime",
    "ensure_ffmpeg_runtime",
    "get_app_paths",
    "initialize_app",
    "read_user_config",
    "save_huggingface_token",
    "transcribe_ipus",
    "transcribe_media_to_ipu_csv",
]
