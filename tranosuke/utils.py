from pathlib import Path
import pandas as pd
import os
import requests
import zipfile
import sys



def download(path: str, url: str):
    """
    指定されたパス（ファイルまたはディレクトリ）が存在しない場合、
    指定されたURLからダウンロードし、zipなら自動解凍する。
    zipファイルの場合、同名のディレクトリを作って展開する。
    """
    path = resource_path(path)
    # すでに存在する場合
    if os.path.exists(path):
        print(f"✅ '{path}' は既に存在します。ダウンロードをスキップします。")
        return

    # ダウンロード先のファイル名を決定
    filename = os.path.basename(url)
    download_path = os.path.join(os.getcwd(), filename)

    print(f"⬇️ '{url}' から '{download_path}' にダウンロードします...")

    # ダウンロード実行
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("✅ ダウンロード完了")

    # zipファイルなら解凍
    if zipfile.is_zipfile(download_path):
        # 展開先ディレクトリ名
        zip_name = os.path.splitext(filename)[0]  # 'xx.zip' -> 'xx'
        extract_dir = os.path.join(os.getcwd(), zip_name)

        os.makedirs(extract_dir, exist_ok=True)
        print(f"📦 zipファイルを '{extract_dir}' に解凍中...")

        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print("✅ 解凍完了")

        # 元のzipファイルを削除（必要ならコメントアウト）
        os.remove(download_path)
        print(f"🧹 '{download_path}' を削除しました。")

    else:
        print("📄 zipファイルではありません。ダウンロードのみ完了。")

def float_to_timecode(value: float) -> str:
    """
    小数点以下3桁を固定して扱い、
    6桁のゼロ埋め文字列に変換する。

    例:
      23.42   -> '023420'
      125.987 -> '125987'
      374.700 -> '374700'
    """
    if value is None:
        return None

    # 小数点以下3桁固定で文字列化（例: 23.42 → '23.420'）
    s = f"{value:.3f}"

    # 小数点を除去
    s = s.replace('.', '')

    # 6桁ゼロ埋め or 超過時切り捨て
    if len(s) < 6:
        s = s.zfill(6)
    elif len(s) > 6:
        s = s[:6]

    return s

def revise_utterance_time(
        df_utt: pd.DataFrame,
        df_phon: pd.DataFrame
        ) -> pd.DataFrame:
    
    df_utt_phon = pd.merge(df_utt, df_phon[["utteranceID", "startTime", "endTime"]], on="utteranceID", how="left", suffixes=("_utt", "_phon"))

    df_utt_adj = (
        df_utt_phon.groupby("utteranceID")
        .apply(lambda g: pd.Series({
            "filename": g["filename"].iloc[0],
            "tier": g["tier"].iloc[0],
            "utterance": g["utterance"].iloc[0],
            "startTime": max(
                g["startTime_utt"].iloc[0],
                g["startTime_phon"].min() if pd.notna(g["startTime_phon"]).any() else g["startTime_utt"].iloc[0]
            ),
            "endTime": min(
                g["endTime_utt"].iloc[0],
                g["endTime_phon"].max() if pd.notna(g["endTime_phon"]).any() else g["endTime_utt"].iloc[0]
            )
        }))
        .reset_index()
    )
    df_utt_adj = df_utt_adj[["filename", "tier", "utteranceID", "startTime", "endTime", "utterance"]]

    return df_utt_adj

def resource_path(relative_path):
    """PyInstallerでも正しくリソースにアクセスできるようにする"""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # PyInstallerでの一時展開先
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    pass