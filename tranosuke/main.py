import os
import streamlit as st
from pathlib import Path
import yaml

from init import *
from utils import *
from wav import *
from utterance import *
from morpheme import *
from phoneme import *
from word import *



def main():

    # アプリタイトルの表示
    st.title("書き起こしアプリ とらのすけ")

    # 画像ファイルパス
    tranosuke = BASE_PATH / "asset/tranosuke.png"  # ベースイメージ
    tranosuke_greeting = BASE_PATH / "asset/tranosuke_greeting.png"  # ベースイメージ
    tranosuke_transcribing = BASE_PATH / "asset/tranosuke_transcribing.png"  # 処理中イメージ
    tranosuke_confusing = BASE_PATH / "asset/tranosuke_confusing.png"  # 処理中イメージ
    tranosuke_finish = BASE_PATH / "asset/tranosuke_finish.png"  # 完了イメージ

    # 画像表示用のプレースホルダ
    img_placeholder = st.empty()
    img_placeholder.image(tranosuke)



    # --- セッション状態の初期化 ---
    # init = 初回起動時（ユーザー設定ファイルの中にhugging faceのアクセストークンがない場合）
    # start = 初期状態
    # processing = 処理中
    # error = エラー発生時
    # done = 完了時

    if "state" not in st.session_state:
        st.session_state.state = None

    # 開始画面
    if st.session_state.state is None:
        col1, col2 = st.columns(2)

        with col1:
            first_time = st.button("はじめて")

        with col2:
            not_first_time = st.button("はじめてじゃない")

        if first_time:
            st.session_state.state = "init"
            st.rerun()
            
        elif not_first_time:
            st.session_state.state = "start"
            st.rerun()



    # --- initセッション ---
    if st.session_state.state == "init":
        img_placeholder.image(tranosuke_greeting)

        # 初期化
        initialize()

        # アクセストークンの取得
        st.markdown("""とらのすけを使ってくれてありがとう！このアプリでは話者分離にpyannoteを使っていて、そのためにHugging Faceのアクセストークンがいるよ！
1. [Hugging Face](https://huggingface.co)にアクセスしてアカウントを作成してね  
2. pyannoteの[利用規約](https://hf.co/pyannote/speaker-diarization-community-1)に同意してね  
3. アクセストークンを[発行](https://hf.co/settings/tokens)してね（一番上のToken typeでReadを選んでね）  
4. 作成したトークンを下に入力してね！""")
        HUGGINGFACE_ACCESS_TOKEN = st.text_input("アクセストークン")
        if st.button("保存"):
            if HUGGINGFACE_ACCESS_TOKEN:
                # config 読み込み
                if CONFIG_PATH.exists():
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f) or {}
                else:
                    config = {}

                # 更新
                config["HUGGINGFACE_ACCESS_TOKEN"] = HUGGINGFACE_ACCESS_TOKEN

                # 保存
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

                st.success("保存しました！")
                st.session_state.state = "start"
                st.rerun()



    # --- startセッション ---
    elif st.session_state.state == "start":
        img_placeholder.image(tranosuke)

        # --- 入力フォーム ---
        st.session_state.audio_path  = Path(st.text_input("音声ファイルはどこ？", "例. /Users/username/audio.wav"))
        st.session_state.quality = st.radio("どっちがいい？", ["スピード優先!", "クオリティ優先!"])

        # --- 解析ボタン ---
        if st.button("書き起こし実行！"):
            # 処理中画像に切り替え
            st.session_state.state = "processing"
            st.rerun()
    


    # --- processingセッション ---
    elif st.session_state.state == "processing":
        try:
            img_placeholder.image(tranosuke_transcribing)

            # 保存ディレクトリがなければ作成
            st.session_state.output_dir = st.session_state.audio_path.with_suffix("")
            if not os.path.exists(st.session_state.output_dir):
                os.makedirs(st.session_state.output_dir)

            

            # --- wavに変換 ---
            with st.spinner("wavファイルに変換してるよ🌊"):
                st.session_state.wav_path = st.session_state.output_dir / f"{st.session_state.audio_path.stem}_processed.wav"
                convert_to_mono_wav(
                    st.session_state.audio_path,
                    st.session_state.wav_path
                    )
                


            # --- 書き起こし ---
            with st.spinner("発話を書き起こしてるよ✏️"):
                if st.session_state.quality == "スピード優先":
                    model_name = "turbo"
                    beam_size = 5
                else:
                    model_name = "large-v3"
                    beam_size = 10

                df_utt = transcribe_utterance(
                    audio_path=st.session_state.wav_path,
                    model_name=model_name,
                    beam_size=beam_size
                    )
                
                if len(df_utt) == 0:
                    raise ValueError("発話が一つも検出できなかったよ！")
                


            # --- 形態素解析 ---
            with st.spinner("形態素解析をしてるよ📕"):
                df_morph = morph_analyze(df_utt)

                # ポーズ記号の行を削除
                df_morph_cleaned = df_morph.copy()
                df_morph_cleaned = df_morph_cleaned[df_morph_cleaned["orth"] != "¥"]



            # --- 強制アラインメント ---
            with st.spinner("強制アラインメントをしてるよ🔍"):
                df_phon = forced_align(st.session_state.wav_path, df_utt, df_morph, 3)
                df_phon.to_csv(st.session_state.output_dir / "phoneme.csv", encoding="utf-8_sig", index=None)



            # --- アラインメントの結果をもとに単語にセグメント情報を追加し発話の開始終了時間を修正 ---
            with st.spinner("単語にセグメント情報を追加してるよ⏰"):
                df_word = add_word_segment(df_morph_cleaned, df_phon)
                df_word.to_csv(st.session_state.output_dir / "word.csv", encoding="utf-8_sig", index=None)
            
                # アラインメントの結果をもとに発話の開始時間を修正
                df_utt_adj = adjust_utterance_time(df_utt, df_phon)
                df_utt_adj["utterance"] = df_utt_adj["utterance"].str.replace("¥", "")
                df_utt_adj.to_csv(st.session_state.output_dir / "utterance.csv", encoding="utf-8_sig", index=None)

            st.session_state.state = "done"
            st.rerun()
        
        # エラーが起きた場合
        except Exception as e:
            st.session_state.error_message = e
            st.session_state.state = "error"
            st.rerun()



    # --- エラー画面 ---
    elif st.session_state.state == "error":
        img_placeholder.image(tranosuke_confusing)
        if st.session_state.error_message:
            st.write(f"エラーが起きたよ！\n{st.session_state.error_message}")
        if st.button("最初に戻る！"):
                st.session_state.state = "start"
                st.rerun()

                

    # --- 完了画面 ---
    elif st.session_state.state == "done":
        img_placeholder.image(tranosuke_finish)
        st.success(f"終わったよ！ 結果を {st.session_state.output_dir} に保存したから確認してね！")

        if st.button("次のファイルを書き起こす!"):
            st.session_state.state = "start"
            st.rerun()



if __name__ == "__main__":
    main()