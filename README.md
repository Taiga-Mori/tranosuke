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

## 動作環境

- Python 3.10 系
- conda環境名: `tranosuke`
- `ffmpeg`（`ffprobe` は不要）
- Hugging Face アカウント
- `pyannote/speaker-diarization-community-1` の利用規約への同意

## インストール

```bash
pip install -r requirements.txt
pip install git+https://github.com/DwangoMediaVillage/pydomino
```

### GPU（CUDA）を使う場合

先に CUDA 版の `torch` / `torchaudio` だけを PyTorch のインデックスから入れ、残りは PyPI から入れます。  
`cu128` の部分は、サーバーの NVIDIA ドライバに合わせて `cu126` などに変えてください。

```bash
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/DwangoMediaVillage/pydomino
```

`torchcodec` は PyPI の CPU 版を使います。CUDA 版（`+cuXXX` 付き）を入れると NVIDIA NPP ライブラリが見つからず、起動時にエラーになります。  
`torchcodec` は音声の読み込みにしか使わないため、CPU 版でも話者分離や書き起こしは GPU で動きます。

すでに CUDA 版が入っている場合は、次のように入れ替えてください。

```bash
pip show torchcodec | grep Version   # +cuXXX が付いていれば CUDA 版
pip uninstall -y torchcodec
pip install torchcodec==0.8.0 --index-url https://pypi.org/simple
```

### setuptools のバージョン

`setuptools` 81 以降では `pkg_resources` が削除されています。`librosa==0.10.1` はこれを使うため、アラインメント時に `No module named 'pkg_resources'` で失敗します。  
`setuptools` は 81 未満にしてください。

```bash
pip install "setuptools<81"
```

### pydomino のビルド

`pydomino` はソースからビルドされるため、`cmake` と C++ コンパイラが必要です。  
Ubuntu / WSL なら `sudo apt install build-essential cmake` で入ります。sudo が使えない場合は、conda-forge でビルド用の環境を作り、そのコンパイラを使います。

```bash
conda create -y -p ./buildenv -c conda-forge cxx-compiler cmake make git
export PATH=$PWD/buildenv/bin:$PATH
export CC=$PWD/buildenv/bin/x86_64-conda-linux-gnu-cc
export CXX=$PWD/buildenv/bin/x86_64-conda-linux-gnu-c++
```

`-static-libstdc++` は付けないでください。同梱の onnxruntime と libstdc++ が混在し、`pydomino.Aligner` の生成時にセグメンテーション違反で落ちます。  
conda のコンパイラでビルドすると、`pydomino_cpp*.so` が `buildenv/lib` を参照したままになります。`buildenv` を消す場合は、インストール後に参照先を外してください（システムの libstdc++ が十分新しい場合）。

```bash
SITE=$(python -c "import site;print(site.getsitepackages()[0])")
patchelf --set-rpath '$ORIGIN/_deps/onnxruntime-src/lib/' $SITE/pydomino/pydomino_cpp*.so
```

新しい GCC では `pydomino` のソースに `#include <cstdint>` が足りず、`'uint32_t' was not declared in this scope` でビルドが失敗します。  
その場合は `CXXFLAGS` で `cstdint` を読み込ませてからインストールしてください。

```bash
CXXFLAGS="-include cstdint" pip install git+https://github.com/DwangoMediaVillage/pydomino
```

初回起動時は、必要な辞書・モデル・公式バイナリのダウンロードに時間がかかります。

## 初回起動時の自動セットアップ

アプリ起動時に `~/.tranosuke/` を確認し、足りないものだけを自動でダウンロードします。

現在、自動取得するものは次の通りです。

- UniDic-CSJ 辞書
- `phoneme_transition_model.onnx`
- ノイズ低減実行時のみ、DeepFilterNet の公式バイナリ

`ffmpeg` はアプリ同梱版があればそれを優先し、なければシステムの `ffmpeg` を使います。

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

ポートは既定で8501を使い、使用中なら8502、8503…と空いているポートを自動で使います。  
ポートを指定する場合は `--port` を付けます（`start.sh` / `start.bat` でも同じです）。

```bash
python app.py --port 8600
```

指定したポートが使用中の場合は、エラーを表示して終了します。

GUIでは以下のタブを使えます。

- `wav変換`
- `ノイズ低減`
- `IPU書き起こし`
- `形態素解析`
- `アラインメント`
- `コーパス作成`

`IPU書き起こし` と `コーパス作成` では処理デバイスを選択できます。GPUを使う場合はGUI上で対象GPUを選びます。  
書き起こしの品質は次の2つから選べます。

| モード | Whisper モデル | ビーム幅 |
| --- | --- | --- |
| スピード優先 | `turbo` | 5 |
| クオリティ優先 | `large-v3` | 10 |

`最小IPU長 秒` より短い区間は書き起こさずに捨てます（既定 0.1 秒）。  
`話者は1人` にチェックを入れると、CLI の `--single-speaker` と同じく全区間を同じ話者として扱います。  
`Whisper に渡す前に音量をそろえる` にチェックを入れると、CLI の `--normalize-volume` と同じく作業用 wav の音量をそろえます。

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

GUI の「クオリティ優先」と同じ設定にする場合:

```bash
python -m tranosuke transcribe input.wav --model-name large-v3 --beam-size 10
```

処理内容:

- pyannoteで話者区間を検出（話者区間が重ならない `exclusive_speaker_diarization` を使用）
- 同一話者の短い隙間を結合
- 無音区間検出でIPU候補を分割・微調整
- `--min-ipu-ms` 未満の区間を捨てる
- Whisperにバッファつき音声を渡して書き起こし
- 書き起こしを正規化（下記）
- `IPU.csv` を出力

重なり発話では Whisper は大きい方の声しか書き起こせないため、話者区間は重ならないものを使います。  
`--single-speaker` を付けると、pyannote が別話者として切り出した区間もすべて同じ話者（`A`）として扱います。話者を置き換えた結果、区間の間隔が `--pause-threshold-ms` 以内になったものはつなげます。話者が1人だけの音声で、pyannote が同じ人を複数の話者に分けてしまう場合に使います（話者分離自体は実行するので、処理時間は短くなりません）。  
`--normalize-volume` を付けると、変換後の作業用 wav（`*_mono.wav`、ノイズ低減をした場合はその出力）の音量をそろえて上書きします。発話部分（20 ミリ秒ごとの音量が最大から 40 dB 以内のフレーム）の RMS が -20 dBFS になるように持ち上げ、ピークが 0.99 を超える場合はそこで止めます。話者分離・書き起こし・アラインメントはすべて音量をそろえた音声を使い、保存される wav もその音声です（入力ファイル自体は変更しません）。録音レベルの小さい音声で、Whisper の聞き取りを安定させるために使います。  
書き起こし時のバッファはWhisperに渡す音声だけに使います。出力されるIPU時刻は無音検出後の区間です。

書き起こしの正規化では、形態素解析と音素アラインメントで扱えるように次の変換をします。

- 算用数字を漢数字にする（`50` → `五十`、`1,000` → `千`、`3.5` → `三点五`、`007` → `〇〇七`）
- 数字と組になる記号を読みにする（`¥500` → `五百円`、`10:30` → `十時三十分`、`2:1` → `二対一`、`3〜5` → `三から五`、`5km` → `五キロメートル`、`30℃` → `三十度`）
- 読み上げられる記号をカタカナにする（`%` → `パーセント`、`@` → `アットマーク`、`&` → `アンド`、`+` → `プラス`、`=` → `イコール` など）
- 仮名の後の波線は引き延ばしとして `ー` にし、それ以外の波線は `から` にする
- 伸ばし棒 `ー` 以外の記号・句読点・空白・文字化けを削除し、先頭の `ー` も削除する

主なオプション:

```bash
--pause-threshold-ms 200
--segment-buffer 0.1
--min-ipu-ms 100
--single-speaker
--normalize-volume
--device cuda
--device-index 0
```

### 4. 書き起こしの正規化

```bash
python -m tranosuke normalize /path/to/IPU.csv
```

既存の `IPU.csv` に、書き起こし時と同じ正規化をかけます。正規化後に空になった IPU と、`--min-ipu-ms`（既定 100）未満の IPU は除きます。`IPUID` は変えません。  
`--output-csv` を指定しない場合は入力ファイルを上書きします。古いバージョンで作った `IPU.csv` から形態素解析・アラインメントをやり直す前に使います。

### 5. 形態素解析

```bash
python -m tranosuke morph /path/to/IPU.csv
```

`IPU.csv` を読み、`morpheme.csv` を出力します。出力列でも `IPUID` を使います。

- 単独の伸ばし棒（`好きで` + `ー`）は直前の形態素に連結し、表層形 `orth` と発音形 `pron` に `ー` を付けます（`でー` / `デー`）。原形 `lemma` と語彙素読み `ruby` は変えません。
- `phonemes` は発音形から作る pydomino 用の音素列です。長音は同じ母音を重ね（`デー` → `d e e`）、撥音は `N`、促音は `cl` です。
- 漢字・仮名以外の文字など、音素が作れない形態素の `phonemes` は空欄になります。

### 6. 音素・単語アラインメント

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

アラインメントの流れ:

1. 各 IPU の音声に前後 0.1 秒の余白を付けて pydomino に渡します。失敗したら余白を 0.2 秒、0.4 秒、隣の IPU までの空き（最大 1 秒）と広げて再試行します。pydomino は1つのプロセスで使い回し、落ちたときだけ起動し直します。
2. 次の IPU は pydomino を使わず、IPU 区間を音素数で等分します（均等割り当て）。
   - 再試行してもアラインメントできなかった IPU
   - 漢字・ひらがな・カタカナ以外の文字を含む IPU
   - 音素が作れない形態素を含む IPU（その形態素には仮の音素 `unk` を1つ割り当てます）
3. 均等割り当てでも1音素 1 ミリ秒を確保できないほど短い IPU はスキップし、その IPU の音素と語は出力しません。
4. 各 IPU の最初の音素の開始と最後の音素の終了を IPU 境界に合わせます。余白にはみ出した音素は、順番を保ったまま IPU の内側へ詰め、どの音素も最低 1 ミリ秒の長さを残します。語も同じように調整します。
5. 語の区間は、その語の音素の区間から作ります。

IPU 区間そのものは修正しません。`phoneme.csv` と `word.csv` の `aligned` 列は、pydomino でアラインメントできたものが `True`、均等割り当てのものが `False` です。語は、構成する音素がすべて `True` のときだけ `True` です。

pydomino の `min_aligned_timeframe`（pau 以外の各音素に割り当てる最低フレーム数、1 フレーム 10 ミリ秒）は 1 にしています。

### 7. コーパスを一括作成

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

`transcribe` と同じく `--pause-threshold-ms`、`--segment-buffer`、`--min-ipu-ms`、`--single-speaker`、`--normalize-volume` も指定できます。

### 8. LUUを作成する古いCLI

LUU作成コードは残しています。必要な場合だけCLIから個別に実行できます。

```bash
python -m tranosuke luu /path/to/word.csv
```

通常のGUIと `corpus` コマンドでは `luu.csv` / `word2luu.csv` は出力しません。

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
filename,speaker,tier,wordID,startTime,endTime,orth,pos,pos1,pos2,pos3,cForm,cType,lemma,ruby,pron,phonemes,aligned
```

`word2IPU.csv`:

```text
filename,wordID,IPUID,nth,len
```

`phoneme.csv`:

```text
filename,speaker,tier,phonemeID,startTime,endTime,phoneme,aligned
```

`phoneme2IPU.csv`:

```text
filename,phonemeID,IPUID,nth,len
```

### ID の形式

`IPUID`・`wordID`・`phonemeID` は「開始時刻 + 話者」です（例: `0084830A` = 84.830 秒に始まる話者 A）。  
開始時刻は「秒の整数部 + ミリ秒3桁」で、整数部の桁数は音声ファイルの長さから決めます（最低3桁）。1000 秒未満の音声なら `084830A`、1000 秒以上 10000 秒未満なら `0084830A` のように、同じファイルの中では桁数がそろいます。

## Docker

リポジトリのルートでビルドします。

```bash
docker build -f docker/Dockerfile -t tranosuke .
docker run --gpus all -p 8501:8501 tranosuke
```

## テスト

```bash
pip install pytest
pytest tests
```

`sample/` の音声を使って、wav変換と形態素解析を数秒で確認します。  
コーパス一括作成まで通す場合は、Hugging Face トークンを設定したうえで次を実行します（Whisper は `tiny` モデルを使います）。

```bash
TRANOSUKE_RUN_SLOW=1 pytest tests
```

GPU を別の処理で使っている場合は、`CUDA_VISIBLE_DEVICES=""` を付けると CPU で実行します。

## 注意点

- Hugging Faceトークンが未設定だと話者分離は実行できません
- 長い音声や話者数が多い音声は時間がかかります
- ノイズ低減はDeepFilterNetの公式バイナリに依存します
- GPUを指定しても環境上使えない場合はCPUへフォールバックすることがあります
- 精度は音質、雑音、重なり発話に大きく左右されます
- Whisper は無音や雑音から「ご視聴ありがとうございました」などの定型文や、同じ語の繰り返しを出力することがあります。とらのすけではこれを判定・除外しません
- 書き起こしの正規化による読みは推測です。単位の読み方（`km` を「キロ」と言うか「キロメートル」と言うか）など、実際の発話と違うことがあります
- 中国語の漢字は日本語の漢字と区別できないため、日本語の読みで推測されます

## 謝辞

とらのすけは、以下のライブラリ、モデル、ツール、データ資源に支えられています。

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) / [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1): 話者分離
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): Whisper書き起こし
- [pydomino](https://github.com/DwangoMediaVillage/pydomino): 音素アラインメント
- [MeCab](https://taku910.github.io/mecab/) / [mecab-python3](https://github.com/SamuraiT/mecab-python3): 形態素解析
- [UniDic-CSJ](https://clrd.ninjal.ac.jp/unidic/): 日本語形態素解析辞書
- [pykakasi](https://github.com/miurahr/pykakasi): かな・ローマ字変換
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet): ノイズ低減
- [FFmpeg](https://ffmpeg.org/): メディア変換
- [PyTorch](https://pytorch.org/) / [torchaudio](https://pytorch.org/audio/stable/index.html): 音声処理とGPU実行基盤
- [librosa](https://librosa.org/), [soundfile](https://python-soundfile.readthedocs.io/), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/): 音声・表形式データ処理
- [Streamlit](https://streamlit.io/): GUI

これらの開発・公開に関わるすべての方々に感謝します。
