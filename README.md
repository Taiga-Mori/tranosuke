# 書き起こしアプリ  とらのすけ ベータ版
!["とらのすけ"](asset/tranosuke.png)

## 何のためのアプリ？
研究用の発話の書き起こしアプリだよ！

## 何ができるの？
動画からwavファイルへの変換、話者分離、発話（200msecの間休止単位）の書き起こし、形態素解析、単語と音素のセグメントができるよ！

## 何がいいの？
- ローカルで動くから個人情報が含まれるデータでも安心！
- 国立国語研究所の現代話し言葉UniDicを使っているので研究向き！

## 事前準備
話者分離に使っているpyannote.audioのモデルをダウンロードするためにHugging Faceのアクセストークンがいるよ！
1. [Hugging Face](https://huggingface.co)にアクセスしてアカウントを作成してね  
2. pyannoteの[利用規約](https://hf.co/pyannote/speaker-diarization-community-1)に同意してね  
3. アクセストークンを[発行](https://hf.co/settings/tokens)してね（一番上のToken typeでReadを選んでね） 

## アプリとして使う （Apple Silicon Mac）
- [ここ](https://drive.google.com/drive/folders/17CF5nJsM1CEM40yz92yw-vfYhettvGfd?usp=sharing)からアプリをダウンロードしてね!
- 警告が出たら「システム設定」 → 「プライバシーとセキュリティ」から解除してね！

## pythonから使う
- 必要なパッケージをインストールしてね
> pip install -r requirements.txt  
- [pydomino](https://github.com/DwangoMediaVillage/pydomino)もインストールしてね！
> pip install git+https://github.com/DwangoMediaVillage/pydomino  
- 下のコマンドで実行してね
> python app.py  
- 最初は辞書やモデルのダウンロードで時間がかかるよ💦
- 長いファイルは短く切った方が精度が上がるかも!
- **精度に関しては自己責任で使ってね！**

## Dockerから使う
1. docker/Dockerfileからイメージをビルドしてね！最初は時間がかかるし容量も30GBくらい使うよ！  
> docker build -t tranosuke -f docker/Dockerfile .  
2. 次はコンテナを起動してね！書き起こしたい音声が/User/username/audioフォルダーにあるとしたら
- GPUなしの場合  
> docker run -p 8501:8501 -v /User/username/audio:/tranosuke/audio tranosuke  
- GPUありの場合  
> docker run --gpus all -p 8501:8501 -v /User/username/audio:/tranosuke/audio tranosuke  
- 起動に成功したらブラウザで[https://localhost:8501](https://localhost:8501)にアクセスしてね！  
- /User/username/audio/audio.wavを書き起こしたい場合は「./audio/audio.wav」と入力してね

## 書き起こし結果
- 入力した音声（映像）ファイルと同じフォルダーに結果のフォルダーができるよ
- その中に変換後のwavファイル、utterance.csv、word.csv、phoneme.csvがあるよ

## 開発環境は？
- Mac
- Python 3.10.18

## どうやって動いてるの？
1. pyannote.audioで各話者の発話区間を推定
2. 各話者の発話区間をfast-whisperで書き起こし
3. 200msec以上のポーズで発話を分割
4. MeCabで形態素解析
5. 形態素の発音形を音素ラベルに変換して強制アラインメントで音素のセグメント
6. 音素のセグメント情報から単語の開始時間と終了時間を逆算
7. 発話の開始時間と終了時間を音素のセグメント情報で修正

## 今後の展望
- ノイズ抑制機能の追加
- 精度の検証

## クレジット
- [pyannote](https://github.com/pyannote/pyannote-audio)
- [fast-whisper](https://github.com/AIXerum/faster-whisper)
- [pydomino](https://github.com/DwangoMediaVillage/pydomino)