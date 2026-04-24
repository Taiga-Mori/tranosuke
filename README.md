# とらのすけ

!["とらのすけ"](asset/tranosuke.png)

研究用の日本語音声コーパス作成アプリです。  
メディア変換、ノイズ低減、話者分離つきIPU書き起こし、形態素解析、音素・単語アラインメント、LUU作成、コーパス一括作成を、GUI とコマンドラインの両方から扱えるようにしています。

## 主な機能

1. メディアファイルを wav に変換
2. 音声・動画のノイズを低減
3. pyannote + Whisper による話者分離つきIPU書き起こし
4. MeCab + UniDic-CSJ による形態素解析
5. pydomino による音素アラインメントと単語アラインメント
6. 単語区間・品詞情報にもとづく LUU 作成
7. 1〜6 をまとめて実行するコーパス作成

## 現在の構成

責務ごとにモジュールを分けています。

- [tranosuke/media.py](/Users/taigamori/Works/tranosuke/tranosuke/media.py): メディア変換
- [tranosuke/denoise.py](/Users/taigamori/Works/tranosuke/tranosuke/denoise.py): ノイズ低減
- [tranosuke/transcription.py](/Users/taigamori/Works/tranosuke/tranosuke/transcription.py): IPU書き起こし
- [tranosuke/morphology.py](/Users/taigamori/Works/tranosuke/tranosuke/morphology.py): 形態素解析
- [tranosuke/alignment.py](/Users/taigamori/Works/tranosuke/tranosuke/alignment.py): 音素・単語アラインメント
- [tranosuke/luu.py](/Users/taigamori/Works/tranosuke/tranosuke/luu.py): LUU 作成
- [tranosuke/corpus.py](/Users/taigamori/Works/tranosuke/tranosuke/corpus.py): 一括コーパス作成
- [tranosuke/gui.py](/Users/taigamori/Works/tranosuke/tranosuke/gui.py): Streamlit GUI
- [tranosuke/cli.py](/Users/taigamori/Works/tranosuke/tranosuke/cli.py): CLI

## 動作環境

- Python 3.10 系
- `ffmpeg`
- `ffprobe`
- Hugging Face アカウント
- `pyannote/speaker-diarization-community-1` の利用規約への同意

## インストール

```bash
pip install -r requirements.txt
pip install git+https://github.com/DwangoMediaVillage/pydomino
```

`requirements.txt` には現行アプリに必要な主要パッケージだけを入れています。  
初回はモデルや辞書のダウンロードに時間がかかります。

## 初期設定

話者分離には Hugging Face のアクセストークンが必要です。

1. [Hugging Face](https://huggingface.co) でアカウントを作成
2. [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) の利用規約に同意
3. [アクセストークン発行ページ](https://hf.co/settings/tokens) で `Read` 権限のトークンを作成

初期化:

```bash
python -m tranosuke init
```

トークン保存:

```bash
python -m tranosuke token hf_xxx
```

初期化で次を準備します。

- `~/.tranosuke/config.yaml`
- UniDic-CSJ 辞書
- `phoneme_transition_model.onnx`

## GUI の使い方

```bash
python app.py
```

または

```bash
python -m tranosuke gui
```

GUI では以下のタブを使えます。

- `wav変換`
- `ノイズ低減`
- `IPU書き起こし`
- `形態素解析`
- `アラインメント`
- `LUU作成`
- `コーパス作成`

## CLI の使い方

### 1. メディアを wav に変換

```bash
python -m tranosuke convert input.mp4
```

出力:

- 全チャンネルを混ぜたモノラル wav
- 複数チャンネルならチャンネルごとの wav

チャンネル別出力を止めたい場合:

```bash
python -m tranosuke convert input.mp4 --no-split-channels
```

### 2. ノイズを低減

```bash
python -m tranosuke denoise input.wav
```

動画や mp3 などを渡した場合は、まず wav に変換してからノイズ低減します。

### 3. IPU 書き起こし

```bash
python -m tranosuke transcribe input.wav --model-name turbo --beam-size 5
```

処理内容:

- 話者分離
- 話者ごとの Whisper 書き起こし
- 単語タイムスタンプから 200ms 以上の休止で IPU 化
- IPU ごとのユニーク ID 付与
- `ipu.csv` 出力

### 4. 形態素解析

```bash
python -m tranosuke morph /path/to/ipu.csv
```

`ipu.csv` を読み、`morpheme.csv` を出力します。

### 5. 音素・単語アラインメント

```bash
python -m tranosuke align /path/to/audio.wav /path/to/ipu.csv /path/to/morpheme.csv
```

出力:

- `phoneme.csv`
- `word.csv`
- `word2ipu.csv`
- 時刻補正済み `ipu.csv`

### 6. LUU を作成

```bash
python -m tranosuke luu /path/to/word.csv
```

`word.csv` の単語区間・品詞情報・IPU対応をもとに、可能な限りマニュアル準拠の LUU をヒューリスティックに作成します。

出力:

- `luu.csv`
- `word2luu.csv`

### 7. コーパスを一括作成

```bash
python -m tranosuke corpus input.mp4 --model-name turbo --beam-size 5
```

ノイズ低減も含める場合:

```bash
python -m tranosuke corpus input.mp4 --denoise
```

## Python API の使い方

### wav 変換

```python
from tranosuke import convert_media_to_wavs

result = convert_media_to_wavs("input.mp4")
print(result.mixed_mono_wav)
print(result.channel_wavs)
```

### ノイズ低減

```python
from tranosuke import denoise_media, denoise_wav

output_path = denoise_media("input.wav")
print(output_path)

wav_only_output = denoise_wav("input.wav")
print(wav_only_output)
```

### 一括コーパス作成

```python
from tranosuke import build_corpus

result = build_corpus("input.mp4", use_denoise=True)
print(result.ipu_csv)
print(result.morpheme_csv)
print(result.word_csv)
print(result.word2ipu_csv)
print(result.luu_csv)
print(result.word2luu_csv)
print(result.phoneme_csv)
```

## 出力ファイル

主に次のファイルを生成します。

- `*_mono.wav`
- `*_ch1.wav`, `*_ch2.wav`, ...
- `ipu.csv`
- `morpheme.csv`
- `word.csv`
- `word2ipu.csv`
- `luu.csv`
- `word2luu.csv`
- `phoneme.csv`

## 注意点

- Hugging Face トークンが未設定だと話者分離は実行できません
- 長い音声や話者数が多い音声は時間がかかります
- ノイズ低減は `fast-music-remover/MediaProcessor` に依存します
- 精度は音質、雑音、重なりIPUに大きく左右されます
