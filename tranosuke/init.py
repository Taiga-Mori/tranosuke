from pathlib import Path
import platform
import yaml
import sys

from utils import *



# 辞書や音素モデルを保存しておくディレクトリ
CACHE_DIR = Path.home() / ".tranosuke"
# ユーザーの情報を保存しておくファイル
CONFIG_PATH = CACHE_DIR / "config.yaml"

# カレントディレクトリ
if hasattr(sys, "_MEIPASS"):
    # アプリ実行時はPyInstallerでの一時展開先
    BASE_PATH = Path(sys._MEIPASS)
else:
    # スクリプト実行時はプロジェクトのディレクトリ
    BASE_PATH = Path(os.path.abspath("."))

# OSの判定
SYSTEM = platform.system()



def initialize():
    """
    - ユーザーのホームディレクトリ直下に .tranosuke フォルダが
      存在しなければ作成する。
    - 辞書がなければダウンロードする。
    - 音素モデルがなければダウンロードする。
    """
    # ユーザーのホームディレクトリ直下に .tranosuke フォルダが存在しなければ作成する
    CACHE_DIR.mkdir(exist_ok=True)

    # 辞書のダウンロード
    download(
        CACHE_DIR / "unidic-csj-202302",
        "https://clrd.ninjal.ac.jp/unidic_archive/2302/unidic-csj-202302.zip"
    )

    # 音素モデルのダウンロード
    download(
        CACHE_DIR / "phoneme_transition_model.onnx",
        "https://github.com/DwangoMediaVillage/pydomino/raw/main/onnx_model/phoneme_transition_model.onnx"
    )

    # ユーザー設定ファイルがなければ作成
    if not CONFIG_PATH.exists():
        config = {}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)



if __name__ == '__main__':
    initialize()
