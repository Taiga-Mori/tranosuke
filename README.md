# 書き起こしアプリ  とらのすけ ベータ版
!["とらのすけ"](asset/tranosuke.png)

## 何のためのアプリ？
研究用の発話の書き起こしアプリだよ！

## 何ができるの？
発話（200msecの間休止単位）の書き起こし、形態素解析、単語と音素のセグメントができるよ！

## 何がいいの？
- ローカルで動くから個人情報が含まれるデータでも安心！
- 国立国語研究所の現代話し言葉UniDicを使っているので研究向き！

## どうやって使うの？
- AppleシリコンのMacの人は[このzip](https://drive.google.com/file/d/1fQmajQ6BUJDHBexdn78DO2HBFFyxg0ug/view?usp=share_link)ファイルの中の実行ファイル「とらのすけ」をクリックするだけで使える（かも）!
- Pythonから使う人は必要なパッケージをインストールしてね
- 最初は辞書やモデルのダウンロードで時間がかかかも
- [pydomino](https://github.com/DwangoMediaVillage/pydomino/tree/main#)は自分でインストールしてね！
> pip install -r requirements.txt  
> pip install git+https://github.com/DwangoMediaVillage/pydomino
- 長いファイルは短く切った方が精度が上がるかも
- 基本的に一つの音声ファイルに一人の話者しかいないことを想定してるよ
- **精度に関しては自己責任で使ってね！**

## 開発環境は？
- Python 3.10.18

## どうやって動いてるの？
1. Whisperで書き起こし
2. 200 msec以上のポーズで発話を分割
3. MeCabで形態素解析
4. 形態素の発音系を音素ラベルに変換して強制アラインメントで音素のセグメント
5. 音素のセグメント情報から単語の開始時間と終了時間を逆算
6. Whisperの発話の開始時間と終了時間はファジーなのでこちらも音素のセグメント情報で修正