import subprocess
from pathlib import Path



def convert_to_mono_wav(input_path: str | Path, output_path: str | Path | None = None):
    """
    ffmpegを使って、指定された動画または音声ファイルを
    1チャンネルのwavファイルに変換する。
    
    Args:
        input_path (str | Path): 変換する元のファイルパス（動画・音声いずれも可）
        output_path (str | Path | None, optional): 出力ファイルパス（省略時は同じ場所に同名で拡張子が .wav のファイルを作成）
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    # ffmpeg コマンド
    cmd = [
        "ffmpeg",
        "-y",                    # 上書き許可
        "-i", str(input_path),   # 入力ファイル
        "-ac", "1",              # 1チャンネル（モノラル）
        "-ar", "16000",          # サンプリングレート（必要に応じて変更可）
        "-acodec", "pcm_s16le",  # 16bit PCM（標準的なWAVフォーマット）
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"変換完了: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"変換に失敗しました: {e.stderr.decode('utf-8', errors='ignore')}")
        raise



if __name__ == '__main__':
    pass
