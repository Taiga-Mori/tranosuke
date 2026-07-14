import faulthandler
import sys
import traceback
from pathlib import Path

import streamlit as st

from tranosuke.alignment import align_phonemes_and_words
from tranosuke.config import initialize_app, list_cuda_devices, read_user_config, save_huggingface_token
from tranosuke.corpus import build_corpus
from tranosuke.denoise import denoise_media
from tranosuke.media import convert_media_to_wavs
from tranosuke.morphology import analyze_ipu_csv
from tranosuke.transcription import transcribe_media_to_ipu_csv


_FAULT_LOG_FILE = None


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


def _device_selector(key: str) -> tuple[str | None, int | None]:
    devices = list_cuda_devices()
    options = ["自動"]
    option_values: dict[str, tuple[str | None, int | None]] = {"自動": (None, None)}

    for device in devices:
        label = (
            f"GPU {device['index']}: {device['name']} "
            f"({device['free_gb']}GB / {device['total_gb']}GB 空き)"
        )
        options.append(label)
        option_values[label] = ("cuda", int(device["index"]))

    options.append("CPU")
    option_values["CPU"] = ("cpu", None)

    selected = st.selectbox("処理デバイス", options, key=key)
    return option_values[selected]


def _console_progress(value: float, message: str) -> None:
    width = 32
    normalized = max(0.0, min(1.0, value))
    filled = int(width * normalized)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(normalized * 100)
    sys.stderr.write(f"\r[{bar}] {percent:3d}% {message}")
    sys.stderr.flush()
    if value >= 1.0:
        sys.stderr.write("\n")


def _streamlit_progress() -> tuple[object, object, object]:
    progress_bar = st.progress(0)
    status = st.empty()

    def update(value: float, message: str) -> None:
        progress_bar.progress(max(0, min(100, int(value * 100))))
        status.write(message)
        _console_progress(value, message)

    return update, progress_bar, status


def _enable_fault_log() -> None:
    global _FAULT_LOG_FILE
    if _FAULT_LOG_FILE is not None:
        return
    log_path = Path("/tmp/tranosuke_streamlit_fault.log")
    _FAULT_LOG_FILE = log_path.open("a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_LOG_FILE, all_threads=True)


def _show_error(error: Exception) -> None:
    st.error(f"処理中にエラーが発生しました: {error}")
    st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))


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
    segment_buffer_s = st.number_input(
        "話者区間の前後バッファ 秒",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.1,
        key="transcribe_segment_buffer",
    )
    pause_threshold_s = st.number_input(
        "無音分割閾値 秒",
        min_value=0.01,
        max_value=5.0,
        value=0.2,
        step=0.01,
        key="transcribe_pause_threshold",
    )
    device, device_index = _device_selector("transcribe_device")
    if st.button("書き起こす", key="transcribe_run"):
        options = QUALITY_OPTIONS[quality]
        progress_callback, _, _ = _streamlit_progress()
        csv_path, _ = transcribe_media_to_ipu_csv(
            input_path,
            model_name=options["model_name"],
            beam_size=options["beam_size"],
            pause_threshold_ms=int(pause_threshold_s * 1000),
            device=device,
            device_index=device_index,
            segment_buffer_s=segment_buffer_s,
            progress_callback=progress_callback,
        )
        st.success(f"IPU書き起こしを保存しました: {csv_path}")


def _morphology_tab() -> None:
    st.subheader("形態素解析")
    input_csv = st.text_input("IPU.csv のパス", key="morph_input")
    if st.button("形態素解析する", key="morph_run"):
        csv_path, _ = analyze_ipu_csv(input_csv)
        st.success(f"形態素解析結果を保存しました: {csv_path}")


def _alignment_tab() -> None:
    st.subheader("音素・単語アラインメント")
    audio_path = st.text_input("wav ファイル", key="align_audio")
    ipu_csv = st.text_input("IPU.csv", key="align_ipu")
    morph_csv = st.text_input("morpheme.csv", key="align_morph")
    alignment_buffer_s = st.number_input(
        "アラインメント前後バッファ 秒",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.1,
        key="align_buffer",
    )
    if st.button("アラインメントする", key="align_run"):
        import pandas as pd

        df_ipu = pd.read_csv(ipu_csv)
        df_morph = pd.read_csv(morph_csv)
        result = align_phonemes_and_words(audio_path, df_ipu, df_morph, alignment_buffer_s=alignment_buffer_s)
        st.success(f"phoneme.csv: {result['phoneme_csv']}")
        st.success(f"word.csv: {result['word_csv']}")
        st.success(f"word2IPU.csv: {result['word2ipu_csv']}")
        st.success(f"phoneme2IPU.csv: {result['phoneme2ipu_csv']}")
        st.success(f"IPU.csv: {result['ipu_csv']}")


def _corpus_tab() -> None:
    st.subheader("コーパスを一括作成")
    input_path = st.text_input("入力ファイル", key="corpus_input")
    quality = st.radio("品質", list(QUALITY_OPTIONS), key="corpus_quality")
    use_denoise = st.checkbox("先にノイズ低減を行う", value=False, key="corpus_denoise")
    segment_buffer_s = st.number_input(
        "話者区間の前後バッファ 秒",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.1,
        key="corpus_segment_buffer",
    )
    pause_threshold_s = st.number_input(
        "無音分割閾値 秒",
        min_value=0.01,
        max_value=5.0,
        value=0.2,
        step=0.01,
        key="corpus_pause_threshold",
    )
    device, device_index = _device_selector("corpus_device")
    if st.button("コーパスを作成", key="corpus_run"):
        options = QUALITY_OPTIONS[quality]
        progress_callback, _, _ = _streamlit_progress()
        try:
            result = build_corpus(
                input_path,
                use_denoise=use_denoise,
                model_name=options["model_name"],
                beam_size=options["beam_size"],
                pause_threshold_ms=int(pause_threshold_s * 1000),
                device=device,
                device_index=device_index,
                segment_buffer_s=segment_buffer_s,
                progress_callback=progress_callback,
            )
        except Exception as error:
            _show_error(error)
            return
        st.success(f"作成先: {result.media.mixed_mono_wav.parent}")
        st.write(f"IPU.csv: {result.ipu_csv}")
        st.write(f"morpheme.csv: {result.morpheme_csv}")
        st.write(f"word.csv: {result.word_csv}")
        st.write(f"word2IPU.csv: {result.word2ipu_csv}")
        st.write(f"phoneme2IPU.csv: {result.phoneme2ipu_csv}")
        st.write(f"phoneme.csv: {result.phoneme_csv}")


def main() -> None:
    _enable_fault_log()
    st.set_page_config(page_title="とらのすけ", layout="wide")
    icon_path = Path(__file__).resolve().parent.parent / "asset" / "tranosuke.png"
    st.image(str(icon_path), width=180)
    st.title("とらのすけ")
    st.caption("メディア変換、ノイズ低減、IPU書き起こし、形態素解析、アラインメント、コーパス作成")

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
        ["wav変換", "ノイズ低減", "IPU書き起こし", "形態素解析", "アラインメント", "コーパス作成"]
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
        _corpus_tab()


if __name__ == "__main__":
    main()
