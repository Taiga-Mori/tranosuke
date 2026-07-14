# とらのすけ

!["とらのすけ"](asset/tranosuke.png)

研究用の日本語音声コーパス作成アプリです。  
メディア変換、ノイズ低減、話者分離つきIPU書き起こし、形態素解析、音素・単語アラインメント、コーパス一括作成を、GUI とコマンドラインの両方から扱えます。

## 主な機能

1. メディアファイルを wav に変換
2. 音声・動画のノイズを低減
3. pyannote + faster-whisper による話者分離つきIPU書き起こし
4. MeCab + UniDic-CSJ による形態素解析
5. pydomino による音素アラインメントと単語アラインメント
6. コーパスCSV一括作成

LUU作成コードとCLIコマンドは残していますが、現在のGUIと通常のコーパス一括作成ではLUUは出力しません。

## 現在の構成

- [tranosuke/media.py](tranosuke/media.py): メディア変換
- [tranosuke/denoise.py](tranosuke/denoise.py): ノイズ低減
- [tranosuke/transcription.py](tranosuke/transcription.py): IPU書き起こし
- [tranosuke/morphology.py](tranosuke/morphology.py): 形態素解析
- [tranosuke/alignment.py](tranosuke/alignment.py): 音素・単語アラインメント
- [tranosuke/luu.py](tranosuke/luu.py): LUU作成コード
- [tranosuke/corpus.py](tranosuke/corpus.py): 一括コーパス作成
- [tranosuke/gui.py](tranosuke/gui.py): Streamlit GUI
- [tranosuke/cli.py](tranosuke/cli.py): CLI
- [scripts/batch_corpus.sh](scripts/batch_corpus.sh): ディレクトリ内wavの一括処理

## 動作環境

- Python 3.10 系
- conda環境名: `tranosuke`
- `ffmpeg`
- `ffprobe`
- Hugging Face アカウント
- `pyannote/speaker-diarization-community-1` の利用規約への同意

## インストール

```bash
pip install -r requirements.txt
pip install git+https://github.com/DwangoMediaVillage/pydomino
```

初回起動時は、必要な辞書・モデル・公式バイナリのダウンロードに時間がかかります。

## 初回起動時の自動セットアップ

アプリ起動時に `~/.tranosuke/` を確認し、足りないものだけを自動でダウンロードします。

現在、自動取得するものは次の通りです。

- UniDic-CSJ 辞書
- `phoneme_transition_model.onnx`
- ノイズ低減実行時のみ、DeepFilterNet の公式バイナリ

`ffmpeg` はアプリ同梱版があればそれを優先し、なければシステムの `ffmpeg` / `ffprobe` を使います。

## Hugging Face トークン設定

話者分離には Hugging Face のアクセストークンが必要です。

1. [Hugging Face](https://huggingface.co) でアカウントを作成
2. [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) の利用規約に同意
3. [アクセストークン発行ページ](https://hf.co/settings/tokens) で `Read` 権限のトークンを作成

CLIでトークン保存:

```bash
python -m tranosuke token hf_xxx
```

## GUI の使い方

```bash
python app.py
```

または

```bash
python -m tranosuke gui
```

GUIでは以下のタブを使えます。

- `wav変換`
- `ノイズ低減`
- `IPU書き起こし`
- `形態素解析`
- `アラインメント`
- `コーパス作成`

`IPU書き起こし` と `コーパス作成` では処理デバイスを選択できます。GPUを使う場合はGUI上で対象GPUを選びます。

## CLI の使い方

### 1. メディアを wav に変換

```bash
python -m tranosuke convert input.mp4
```

出力:

- 全チャンネルを混ぜたモノラルwav
- 複数チャンネルならチャンネルごとのwav

チャンネル別出力を止めたい場合:

```bash
python -m tranosuke convert input.mp4 --no-split-channels
```

### 2. ノイズを低減

```bash
python -m tranosuke denoise input.wav
```

動画やmp3などを渡した場合は、まず48kHzモノラルwavに変換してからノイズ低減します。  
この機能を初めて使うときだけ、DeepFilterNet の公式バイナリを自動ダウンロードします。

### 3. IPU書き起こし

```bash
python -m tranosuke transcribe input.wav --model-name turbo --beam-size 5
```

GPUを指定する場合:

```bash
python -m tranosuke transcribe input.wav --device cuda --device-index 0
```

処理内容:

- pyannoteで話者区間を検出
- 同一話者の短い隙間を結合
- 無音区間検出でIPU候補を分割・微調整
- Whisperにバッファつき音声を渡して書き起こし
- `IPU.csv` を出力

書き起こし時のバッファはWhisperに渡す音声だけに使います。出力されるIPU時刻は無音検出後の区間です。

主なオプション:

```bash
--pause-threshold-ms 200
--segment-buffer 0.1
--device cuda
--device-index 0
```

### 4. 形態素解析

```bash
python -m tranosuke morph /path/to/IPU.csv
```

`IPU.csv` を読み、`morpheme.csv` を出力します。出力列でも `IPUID` を使います。

### 5. 音素・単語アラインメント

```bash
python -m tranosuke align /path/to/audio.wav /path/to/IPU.csv /path/to/morpheme.csv
```

出力:

- `IPU.csv`
- `morpheme.csv` は入力側で作成済み
- `phoneme.csv`
- `word.csv`
- `word2IPU.csv`
- `phoneme2IPU.csv`

現在は、音素アラインメント後にIPU区間そのものは修正しません。  
ただし、各IPU内の最初/最後のwordとphonemeについては、開始/終了時刻をIPU境界へ強制的に合わせます。

### 6. コーパスを一括作成

```bash
python -m tranosuke corpus input.mp4 --model-name turbo --beam-size 5
```

GPU0を使う場合:

```bash
python -m tranosuke corpus input.mp4 --device cuda --device-index 0
```

ノイズ低減も含める場合:

```bash
python -m tranosuke corpus input.mp4 --denoise
```

### 7. LUUを作成する古いCLI

LUU作成コードは残しています。必要な場合だけCLIから個別に実行できます。

```bash
python -m tranosuke luu /path/to/word.csv
```

通常のGUIと `corpus` コマンドでは `luu.csv` / `word2luu.csv` は出力しません。

## ディレクトリ内wavの一括処理

`scripts/batch_corpus.sh` は、指定ディレクトリ直下の `*_AC.wav` と `*_AP.wav` を順番に処理します。

```bash
scripts/batch_corpus.sh ~/Data/ADOS/FirstEncounter
```

GPU0を使う場合:

```bash
scripts/batch_corpus.sh ~/Data/ADOS/FirstEncounter -- --device cuda --device-index 0
```

追加オプションは `--` の後ろに渡します。

```bash
scripts/batch_corpus.sh ~/Data/ADOS/FirstEncounter -- --pause-threshold-ms 200 --segment-buffer 0.1
```

出力先は入力wavの拡張子を除いたディレクトリです。

```text
/path/to/hogehoge_AC.wav -> /path/to/hogehoge_AC/
/path/to/hogehoge_AP.wav -> /path/to/hogehoge_AP/
```

1件失敗しても残りは続行し、最後に失敗ファイル一覧を表示します。

## Python API の使い方

### wav変換

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

result = build_corpus("input.mp4", use_denoise=True, device="cuda", device_index=0)
print(result.ipu_csv)
print(result.morpheme_csv)
print(result.word_csv)
print(result.word2ipu_csv)
print(result.phoneme2ipu_csv)
print(result.phoneme_csv)
```

## 出力ファイル

通常のコーパス作成では主に次のファイルを生成します。

- `*_mono.wav`
- `*_ch1.wav`, `*_ch2.wav`, ...
- `IPU.csv`
- `morpheme.csv`
- `word.csv`
- `word2IPU.csv`
- `phoneme.csv`
- `phoneme2IPU.csv`

### CSV列

`IPU.csv`:

```text
filename,speaker,tier,IPUID,startTime,endTime,IPU
```

`morpheme.csv`:

```text
filename,speaker,IPUID,orth,pos,pos1,pos2,pos3,cForm,cType,lemma,ruby,pron,phonemes,nth,len
```

`word.csv`:

```text
filename,speaker,tier,wordID,startTime,endTime,orth,pos,pos1,pos2,pos3,cForm,cType,lemma,ruby,pron,phonemes
```

`word2IPU.csv`:

```text
filename,wordID,IPUID,nth,len
```

`phoneme.csv`:

```text
filename,speaker,tier,phonemeID,startTime,endTime,phoneme
```

`phoneme2IPU.csv`:

```text
filename,phonemeID,IPUID,nth,len
```

## 注意点

- Hugging Faceトークンが未設定だと話者分離は実行できません
- 長い音声や話者数が多い音声は時間がかかります
- ノイズ低減はDeepFilterNetの公式バイナリに依存します
- GPUを指定しても環境上使えない場合はCPUへフォールバックすることがあります
- 精度は音質、雑音、重なり発話に大きく左右されます
