from pathlib import Path
import pandas as pd
import os
import requests
import zipfile



def download(path: str, url: str):
    """
    指定されたパス（ファイルまたはディレクトリ）が存在しない場合、
    指定されたURLからダウンロードし、zipなら自動解凍する。
    Args:
        path (str): 確認したいファイルまたはディレクトリ
        url (str): ダウンロード先のリンク
    """

    path = Path(path).expanduser()
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # すでに存在する場合はスキップ
    if path.exists():
        print(f"'{path}' は既に存在します。ダウンロードをスキップします。")

    else:
        print(f"'{path}' が存在しません。'{url}' から取得します...")

        # ダウンロード先の一時ファイル
        tmp_path = target_dir / os.path.basename(url)

        # ダウンロード実行
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"ダウンロード完了: {tmp_path}")

        # zipファイルなら解凍
        if zipfile.is_zipfile(tmp_path):
            print(f"zipファイルを '{target_dir / path.name}' に解凍中...")
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(target_dir / path.name)
            print("解凍完了")

            # zip削除（必要に応じて保持したいならコメントアウト）
            tmp_path.unlink()
            print(f"一時ファイル '{tmp_path.name}' を削除しました。")

        else:
            print("zipファイルではありません。ダウンロードのみ完了。")



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



def adjust_utterance_time(
        df_utt: pd.DataFrame,
        df_phon: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Whisperの時間はファジーなので発話の開始時間を最初の音素の開始時間に、
    発話の終了時間を最後の音素の終了時間に修正する
    
    Args:
        df_utt (pd.DataFrame): 発話のデータフレーム
        df_phon (pd.DataFrame): 音素のデータフレーム

    Returns:
        df_adjusted (pd.DataFrame): 修正された発話のデータフレーム
    """

    # 音素の最小・最大時刻を utteranceID ごとに取得
    phon_range = (
        df_phon.groupby("utteranceID")
        .agg(startTime_phon=("startTime", "min"),
        endTime_phon=("endTime", "max"))
        .reset_index()
    )
    
    df_adjusted = pd.merge(df_utt, phon_range, on="utteranceID", how="left")

    # 音素の時間に基づいて更新（音素がない場合は元の値を保持）
    df_adjusted["startTime"] = df_adjusted.apply(
        lambda r: r["startTime_phon"] if pd.notna(r["startTime_phon"]) else r["startTime"], axis=1
    )
    df_adjusted["endTime"] = df_adjusted.apply(
        lambda r: r["endTime_phon"] if pd.notna(r["endTime_phon"]) else r["endTime"], axis=1
    )

    # 不要な列を削除
    df_adjusted = df_adjusted.drop(columns=["startTime_phon", "endTime_phon"])

    return df_adjusted



if __name__ == '__main__':
    pass