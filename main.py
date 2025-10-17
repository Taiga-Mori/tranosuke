import streamlit as st
from pathlib import Path
import os

from tranosuke.utils import *
from tranosuke.utterance import *
from tranosuke.morpheme import *
from tranosuke.phoneme import *
from tranosuke.word import *



def main():

    st.title("書き起こしアプリ とらのすけ")

    # --- 画像ファイルパス ---
    img1_path = resource_path("asset/tranosuke.png")  # 初期イメージ
    img2_path = resource_path("asset/tranosuke_transcribing.png")  # 処理中イメージ
    img3_path = resource_path("asset/tranosuke_finish.png")  # 完了イメージ



    # --- セッション状態の初期化 ---
    if "state" not in st.session_state:
        st.session_state.state = "start"          # start / processing / done

    def reset_state():
        st.session_state.state = "start"



    # --- 画像表示用のプレースホルダ ---
    img_placeholder = st.empty()

    # --- startセッション ---
    if st.session_state.state == "start":
        img_placeholder.image(img1_path)

        # --- 入力フォーム ---
        st.session_state.audio_path = audio_path = Path(st.text_input("音声ファイルはどこ？", "例. /Users/username/audio.wav"))
        st.session_state.output_dir = output_dir = Path(st.text_input("どこに保存する？", "例. /Users/username/audio"))
        st.session_state.quality = quality = st.radio("どっちがいい？", ["スピード優先!", "クオリティ優先!"])

        # --- 解析ボタン ---
        if st.button("書き起こし実行！"):
            if not audio_path.exists():
                st.error("音声ファイルが見つからないよ〜💦")
            else:
                # 処理中画像に切り替え
                st.session_state.state = "processing"
                st.rerun()  # 状態変更後に再レンダリング
        
    elif st.session_state.state == "processing":
        img_placeholder.image(img2_path)

        # 保存ディレクトリがなければ作成
        if not os.path.exists(st.session_state.output_dir):
            os.makedirs(st.session_state.output_dir)
            
        # --- 書き起こし ---
        with st.spinner("発話を書き起こしてるよ✏️"):
            if st.session_state.quality == "スピード優先":
                model_name = "turbo"
            else:
                model_name = "large-v3"

            df_utt = transcribe_utterance(
                audio_path=st.session_state.audio_path,
                model_name=model_name
                )
            


        # --- 形態素解析 ---
        with st.spinner("形態素解析をしてるよ📕"):
            df_morph = morph_analyze(df_utt)

            # ポーズ記号の行を削除
            df_morph_cleaned = df_morph.copy()
            df_morph_cleaned = df_morph_cleaned[df_morph_cleaned["orth"] != "¥"]



        # --- 強制アラインメント ---
        with st.spinner("強制アラインメントをしてるよ🔍"):
            if st.session_state.quality == "スピード優先":
                iterations = 3
            else:
                iterations = 5
            df_phon = forced_align(st.session_state.audio_path, df_utt, df_morph, iterations)
            df_phon.to_csv(st.session_state.output_dir / "phoneme.csv", encoding="utf-8_sig", index=None)



        # --- アラインメントの結果をもとに単語にセグメント情報を追加し発話の開始終了時間を修正 ---
        with st.spinner("単語にセグメント情報を追加してるよ⏰"):
            df_word = add_segment(df_utt, df_morph_cleaned, df_phon)
            df_word.to_csv(st.session_state.output_dir / "word.csv", encoding="utf-8_sig", index=None)
        
            # アラインメントの結果をもとに発話の開始時間を修正
            df_utt_adj = revise_utterance_time(df_utt, df_phon)
            df_utt_adj["utterance"] = df_utt_adj["utterance"].str.replace("¥", "(.)")
            df_utt_adj.to_csv(st.session_state.output_dir / "utterance.csv", encoding="utf-8_sig", index=None)



        st.session_state.state = "done"
        st.rerun()



    # --- 完了画面 ---
    elif st.session_state.state == "done":
        img_placeholder.image(img3_path)
        st.success(f"終わったよ！ 結果を {st.session_state.output_dir} に保存したから確認してね！")

        if st.button("次のファイルを書き起こす!"):
            reset_state()
            st.rerun()



if __name__ == "__main__":
    main()