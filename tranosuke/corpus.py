from dataclasses import dataclass
from pathlib import Path

from tranosuke.alignment import align_phonemes_and_words
from tranosuke.denoise import denoise_media
from tranosuke.luu import build_luus
from tranosuke.media import MediaConversionResult, convert_media_to_wavs
from tranosuke.morphology import analyze_ipus
from tranosuke.transcription import transcribe_ipus


@dataclass(frozen=True)
class CorpusBuildResult:
    media: MediaConversionResult
    working_wav: Path
    ipu_csv: Path
    morpheme_csv: Path
    word_csv: Path
    word2ipu_csv: Path
    luu_csv: Path
    word2luu_csv: Path
    phoneme_csv: Path


def build_corpus(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    use_denoise: bool = False,
    model_name: str = "turbo",
    beam_size: int = 5,
) -> CorpusBuildResult:
    """
    Run the end-to-end corpus pipeline from media input to CSV outputs.
    """
    media_result = convert_media_to_wavs(input_path, output_dir=output_dir, split_channels=True)
    working_wav = media_result.mixed_mono_wav

    if use_denoise:
        working_wav = denoise_media(working_wav)

    df_ipu = transcribe_ipus(working_wav, model_name=model_name, beam_size=beam_size)
    df_morph = analyze_ipus(df_ipu)

    target_dir = media_result.mixed_mono_wav.parent
    morpheme_csv = target_dir / "morpheme.csv"
    df_morph.to_csv(morpheme_csv, encoding="utf-8_sig", index=False)

    alignment_result = align_phonemes_and_words(working_wav, df_ipu, df_morph, output_dir=target_dir)
    luu_result = build_luus(alignment_result["word_df"])
    luu_csv = target_dir / "luu.csv"
    word2luu_csv = target_dir / "word2luu.csv"
    luu_result[0].to_csv(luu_csv, encoding="utf-8_sig", index=False)
    luu_result[1].to_csv(word2luu_csv, encoding="utf-8_sig", index=False)

    return CorpusBuildResult(
        media=media_result,
        working_wav=Path(working_wav),
        ipu_csv=Path(alignment_result["ipu_csv"]),
        morpheme_csv=morpheme_csv,
        word_csv=Path(alignment_result["word_csv"]),
        word2ipu_csv=Path(alignment_result["word2ipu_csv"]),
        luu_csv=luu_csv,
        word2luu_csv=word2luu_csv,
        phoneme_csv=Path(alignment_result["phoneme_csv"]),
    )
