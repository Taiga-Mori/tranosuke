# 書き起こしアプリ  とらのすけ ベータ版
!["とらのすけ"](asset/tranosuke.png)

## 何のためのアプリ？
研究用の発話の書き起こしアプリだよ！

## 何ができるの？
発話（200msecの間休止単位）の書き起こし、話者分離、形態素解析、単語と音素のセグメントができるよ！

## 何がいいの？
- ローカルで動くから個人情報が含まれるデータでも安心！
- 国立国語研究所の現代話し言葉UniDicを使っているので研究向き！

## アプリとして使う
[このドライブ](https://drive.google.com/drive/folders/17CF5nJsM1CEM40yz92yw-vfYhettvGfd?usp=sharing)から自分の環境に合ったファイルをダウンロードしてね!

## pythonから使う
- Pythonから使う人は必要なパッケージをインストールしてね
> pip install -r requirements.txt  
- [pydomino](https://github.com/DwangoMediaVillage/pydomino/tree/main#)もインストールしてね！
> pip install git+https://github.com/DwangoMediaVillage/pydomino  
- 下のコマンドで実行してね
> streamlit run tranosuke/main.py  
- 最初は辞書やモデルのダウンロードで時間がかかるよ💦
- 長いファイルは短く切った方が精度が上がるかも
- **精度に関しては自己責任で使ってね！**

## 使い方
- 初めて使う人は「はじめて」を選んでね
- 案内に従ってHugging Faceのアクセストークンを発行してね
- 入力するファイルはwav以外にもmp4やaviでも大丈夫だよ！
- 入力したファイルと同じところにフォルダーができて、その中にutterance.csv、word.csv、phoneme.csvができるよ

## 開発環境は？
- Mac
- Python 3.10.18

## どうやって動いてるの？
1. pyannote.audioで各話者の発話区間を推定
2. 各話者の発話区間をWhisperで書き起こし
3. 200msec以上のポーズで発話を分割
4. MeCabで形態素解析
5. 形態素の発音形を音素ラベルに変換して強制アラインメントで音素のセグメント
6. 音素のセグメント情報から単語の開始時間と終了時間を逆算
7. Whisperの発話の開始時間と終了時間はファジーなのでこちらも音素のセグメント情報で修正