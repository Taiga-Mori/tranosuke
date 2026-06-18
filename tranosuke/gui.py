from pathlib import Path

import streamlit as st

from tranosuke.alignment import align_phonemes_and_words
from tranosuke.config import initialize_app, read_user_config, save_huggingface_token
from tranosuke.corpus import build_corpus
from tranosuke.denoise import denoise_media
from tranosuke.luu import build_luus_from_word_csv
from tranosuke.media import convert_media_to_wavs
from tranosuke.morphology import analyze_ipu_csv
from tranosuke.transcription import transcribe_media_to_ipu_csv


QUALITY_OPTIONS = {
    "スピード優先": {"model_name": "turbo", "beam_size": 5},
    "クオリティ優先": {"model_name": "large-v3", "beam_size": 10},
}


def _ensure_startup_assets() -> None:
    if st.session_state.get("startup_assets_ready"):
        return

    with st.spinner("初回起動に必要な辞書とモデルを確認しています..."):
        initialize_app()
    st.session_state["startup_assets_ready"] = True


def _save_token_form() -> None:
    current_token = read_user_config().get("HUGGINGFACE_ACCESS_TOKEN", "")
    token = st.text_input("Hugging Face アクセストークン", value=current_token, type="password")
    if st.button("トークンを保存"):
        save_huggingface_token(token)
        st.success("保存しました。")


def _conversion_tab() -> None:
    st.subheader("1. メディアを wav に変換")
    input_path = st.text_input("入力ファイル", key="convert_input")
    split_channels = st.checkbox("チャンネルごとの wav も出力する", value=True)
    if st.button("変換する", key="convert_run"):
        result = convert_media_to_wavs(input_path, split_channels=split_channels)
        st.success(f"モノラル wav を作成しました: {result.mixed_mono_wav}")
        if result.channel_wavs:
            st.write("チャンネル別 wav:")
            for path in result.channel_wavs:
                st.write(str(path))


def _denoise_tab() -> None:
    st.subheader("ノイズ低減")
    st.caption("初回実行時は、公式の DeepFilterNet バイナリを自動ダウンロードします。")
    input_path = st.text_input("入力ファイル", key="denoise_input")
    if st.button("ノイズ低減する", key="denoise_run"):
        output_path = denoise_media(input_path)
        st.success(f"ノイズ低減後のファイル: {output_path}")


def _transcription_tab() -> None:
    st.subheader("話者分離と IPU 書き起こし")
    input_path = st.text_input("入力ファイル", key="transcribe_input")
    quality = st.radio("品質", list(QUALITY_OPTIONS), key="transcribe_quality")
    if st.button("書き起こす", key="transcribe_run"):
        options = QUALITY_OPTIONS[quality]
        csv_path, _ = transcribe_media_to_ipu_csv(
            input_path,
            model_name=options["model_name"],
            beam_size=options["beam_size"],
        )
        st.success(f"IPU 書き起こしを保存しました: {csv_path}")


def _morphology_tab() -> None:
    st.subheader("形態素解析")
    input_csv = st.text_input("ipu.csv のパス", key="morph_input")
    if st.button("形態素解析する", key="morph_run"):
        csv_path, _ = analyze_ipu_csv(input_csv)
        st.success(f"形態素解析結果を保存しました: {csv_path}")


def _alignment_tab() -> None:
    st.subheader("音素・単語アラインメント")
    audio_path = st.text_input("wav ファイル", key="align_audio")
    ipu_csv = st.text_input("ipu.csv", key="align_ipu")
    morph_csv = st.text_input("morpheme.csv", key="align_morph")
    if st.button("アラインメントする", key="align_run"):
        import pandas as pd

        df_ipu = pd.read_csv(ipu_csv)
        df_morph = pd.read_csv(morph_csv)
        result = align_phonemes_and_words(audio_path, df_ipu, df_morph)
        st.success(f"phoneme.csv: {result['phoneme_csv']}")
        st.success(f"word.csv: {result['word_csv']}")
        st.success(f"word2ipu.csv: {result['word2ipu_csv']}")
        st.success(f"ipu.csv: {result['ipu_csv']}")


def _luu_tab() -> None:
    st.subheader("LUU 作成")
    word_csv = st.text_input("word.csv", key="luu_word_csv")
    if st.button("LUU を作成する", key="luu_run"):
        result = build_luus_from_word_csv(word_csv)
        st.success(f"luu.csv: {result['luu_csv']}")
        st.success(f"word2luu.csv: {result['word2luu_csv']}")


def _corpus_tab() -> None:
    st.subheader("コーパスを一括作成")
    input_path = st.text_input("入力ファイル", key="corpus_input")
    quality = st.radio("品質", list(QUALITY_OPTIONS), key="corpus_quality")
    use_denoise = st.checkbox("先にノイズ低減を行う", value=False, key="corpus_denoise")
    if st.button("コーパスを作成", key="corpus_run"):
        options = QUALITY_OPTIONS[quality]
        result = build_corpus(
            input_path,
            use_denoise=use_denoise,
            model_name=options["model_name"],
            beam_size=options["beam_size"],
        )
        st.success(f"作成先: {result.media.mixed_mono_wav.parent}")
        st.write(f"ipu.csv: {result.ipu_csv}")
        st.write(f"morpheme.csv: {result.morpheme_csv}")
        st.write(f"word.csv: {result.word_csv}")
        st.write(f"word2ipu.csv: {result.word2ipu_csv}")
        st.write(f"luu.csv: {result.luu_csv}")
        st.write(f"word2luu.csv: {result.word2luu_csv}")
        st.write(f"phoneme.csv: {result.phoneme_csv}")


def main() -> None:
    st.set_page_config(page_title="とらのすけ", layout="wide")
    icon_path = Path(__file__).resolve().parent.parent / "asset" / "tranosuke.png"
    st.image(str(icon_path), width=180)
    st.title("とらのすけ")
    st.caption("メディア変換、ノイズ低減、IPU書き起こし、形態素解析、アラインメント、LUU作成、コーパス作成")

    try:
        _ensure_startup_assets()
        if not st.session_state.get("startup_status_shown"):
            st.success("起動に必要な辞書とモデルを確認しました。")
            st.session_state["startup_status_shown"] = True
    except Exception as error:
        st.error(f"初期化に失敗しました: {error}")
        st.stop()

    if st.button("辞書とモデルを再確認する"):
        initialize_app()
        st.success("再確認が完了しました。")

    with st.expander("アクセストークン設定"):
        _save_token_form()

    tabs = st.tabs(
        ["wav変換", "ノイズ低減", "IPU書き起こし", "形態素解析", "アラインメント", "LUU作成", "コーパス作成"]
    )
    with tabs[0]:
        _conversion_tab()
    with tabs[1]:
        _denoise_tab()
    with tabs[2]:
        _transcription_tab()
    with tabs[3]:
        _morphology_tab()
    with tabs[4]:
        _alignment_tab()
    with tabs[5]:
        _luu_tab()
    with tabs[6]:
        _corpus_tab()


if __name__ == "__main__":
    main()
