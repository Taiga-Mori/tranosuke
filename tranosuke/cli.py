import argparse
from pathlib import Path
import sys

from tranosuke.alignment import align_phonemes_and_words
from tranosuke.config import initialize_app, save_huggingface_token
from tranosuke.corpus import build_corpus
from tranosuke.denoise import denoise_media
from tranosuke.luu import build_luus_from_word_csv
from tranosuke.media import convert_media_to_wavs
from tranosuke.morphology import analyze_ipu_csv
from tranosuke.transcription import transcribe_media_to_ipu_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tranosuke")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    token_parser = subparsers.add_parser("token")
    token_parser.add_argument("value")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("input_path")
    convert_parser.add_argument("--output-dir")
    convert_parser.add_argument("--no-split-channels", action="store_true")

    denoise_parser = subparsers.add_parser("denoise")
    denoise_parser.add_argument("input_path")
    denoise_parser.add_argument("--output-dir")

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("input_path")
    transcribe_parser.add_argument("--output-dir")
    transcribe_parser.add_argument("--model-name", default="turbo")
    transcribe_parser.add_argument("--beam-size", type=int, default=5)

    morph_parser = subparsers.add_parser("morph")
    morph_parser.add_argument("input_csv")
    morph_parser.add_argument("--output-csv")

    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("audio_path")
    align_parser.add_argument("ipu_csv")
    align_parser.add_argument("morpheme_csv")
    align_parser.add_argument("--output-dir")

    luu_parser = subparsers.add_parser("luu")
    luu_parser.add_argument("word_csv")
    luu_parser.add_argument("--output-dir")

    corpus_parser = subparsers.add_parser("corpus")
    corpus_parser.add_argument("input_path")
    corpus_parser.add_argument("--output-dir")
    corpus_parser.add_argument("--model-name", default="turbo")
    corpus_parser.add_argument("--beam-size", type=int, default=5)
    corpus_parser.add_argument("--denoise", action="store_true")

    subparsers.add_parser("gui")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"morph", "align", "luu", "corpus", "gui"}:
        initialize_app(include_denoise=args.command == "corpus" and getattr(args, "denoise", False))

    if args.command == "init":
        paths = initialize_app()
        print(paths.cache_dir)
        return 0

    if args.command == "token":
        save_huggingface_token(args.value)
        print("saved")
        return 0

    if args.command == "convert":
        result = convert_media_to_wavs(
            args.input_path,
            output_dir=args.output_dir,
            split_channels=not args.no_split_channels,
        )
        print(result.mixed_mono_wav)
        for channel_wav in result.channel_wavs:
            print(channel_wav)
        return 0

    if args.command == "denoise":
        initialize_app(include_denoise=True)
        print(denoise_media(args.input_path, output_dir=args.output_dir))
        return 0

    if args.command == "transcribe":
        csv_path, _ = transcribe_media_to_ipu_csv(
            args.input_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            beam_size=args.beam_size,
        )
        print(csv_path)
        return 0

    if args.command == "morph":
        csv_path, _ = analyze_ipu_csv(args.input_csv, output_csv_path=args.output_csv)
        print(csv_path)
        return 0

    if args.command == "align":
        import pandas as pd

        df_ipu = pd.read_csv(args.ipu_csv)
        df_morph = pd.read_csv(args.morpheme_csv)
        result = align_phonemes_and_words(args.audio_path, df_ipu, df_morph, output_dir=args.output_dir)
        print(result["phoneme_csv"])
        print(result["word_csv"])
        print(result["word2ipu_csv"])
        print(result["ipu_csv"])
        return 0

    if args.command == "luu":
        result = build_luus_from_word_csv(args.word_csv, output_dir=args.output_dir)
        print(result["luu_csv"])
        print(result["word2luu_csv"])
        return 0

    if args.command == "corpus":
        result = build_corpus(
            args.input_path,
            output_dir=args.output_dir,
            use_denoise=args.denoise,
            model_name=args.model_name,
            beam_size=args.beam_size,
        )
        print(result.ipu_csv)
        print(result.morpheme_csv)
        print(result.word_csv)
        print(result.word2ipu_csv)
        print(result.luu_csv)
        print(result.word2luu_csv)
        print(result.phoneme_csv)
        return 0

    if args.command == "gui":
        import streamlit.web.cli as stcli

        gui_script = str(Path(__file__).with_name("gui.py"))
        sys.argv = ["streamlit", "run", gui_script, "--global.developmentMode=false"]
        return stcli.main()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
