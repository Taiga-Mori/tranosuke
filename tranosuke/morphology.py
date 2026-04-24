from pathlib import Path
import re

import MeCab
import numpy as np
import pandas as pd
import pykakasi

from tranosuke.config import get_app_paths


def kana_to_roman(kana: str, kakasi) -> str:
    return "".join(word["hepburn"] for word in kakasi.convert(kana))


def roman_to_phonemes(roman: str) -> str:
    text = roman.strip().lower()
    vowels = set("aiueo")
    two_onsets = {"ky", "gy", "sh", "ch", "ny", "hy", "by", "py", "my", "ry", "ts"}
    consonants = set("bcdfghjklmnpqrstvwxyz")

    tokens = []
    index = 0
    while index < len(text):
        if text[index] == "n":
            if index + 1 == len(text) or (text[index + 1] not in vowels and text[index + 1] != "y"):
                tokens.append("N")
                index += 1
                continue

        if (
            index + 1 < len(text)
            and text[index] == text[index + 1]
            and text[index] in consonants
            and text[index] != "n"
        ):
            tokens.append("cl")
            index += 1
            continue

        if index + 2 < len(text):
            onset = text[index:index + 2]
            vowel = text[index + 2]
            if onset in two_onsets and vowel in vowels:
                tokens.extend([onset, vowel])
                index += 3
                continue

        if index + 1 < len(text):
            consonant = text[index]
            vowel = text[index + 1]
            if consonant in consonants and vowel in vowels:
                tokens.extend([consonant, vowel])
                index += 2
                continue

        if text[index] in vowels:
            tokens.append(text[index])
            index += 1
            continue

        tokens.append(text[index])
        index += 1

    return " ".join(tokens)


def _parse_morphemes(text: str, tagger, kakasi) -> list:
    node = tagger.parseToNode(text)
    morphemes = []

    while node:
        surface = node.surface
        morpheme = node.feature.split(",")
        morpheme.insert(0, surface)

        if morpheme[1] != "BOS/EOS":
            if len(morpheme) >= 15:
                morpheme = [morpheme[i] for i in [0, 1, 2, 3, 4, 6, 5, 8, 7, 10]]
            else:
                morpheme = [morpheme[i] for i in [0, 1, 2, 3, 4]] + [np.nan for _ in range(5)]

            morpheme = [np.nan if value == "*" else value for value in morpheme]
            if morpheme[0] == "#":
                morpheme[-1] = "シャープ"
            if morpheme[0] == "%":
                morpheme[-1] = "パーセント"
            if pd.isna(morpheme[-1]):
                morpheme[-1] = "".join(item["kana"] for item in kakasi.convert(morpheme[0]))

            phonemes = roman_to_phonemes(kana_to_roman(morpheme[-1], kakasi))
            if re.search(r"[^A-Za-z ]", phonemes):
                phonemes = ""
            if morpheme[0] == "¥":
                phonemes = "pau"

            morpheme.append(phonemes)
            morphemes.append(morpheme)

        node = node.next

    length = len(morphemes)
    for nth, morpheme in enumerate(morphemes, start=1):
        morpheme.append(nth)
        morpheme.append(length)

    return morphemes


def analyze_ipus(df_ipu: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IPU rows into morpheme rows with phoneme strings.
    """
    kakasi = pykakasi.kakasi()
    paths = get_app_paths()
    dic_path = paths.cache_dir / "unidic-csj-202302"
    if paths.system == "Windows":
        dic_path = Path(str(dic_path).replace("\\", "/"))

    tagger = MeCab.Tagger(f"-d {dic_path}")
    tagger.parse("")

    rows = []
    for _, ipu_row in df_ipu.iterrows():
        if not pd.notna(ipu_row["ipu"]):
            continue
        morphemes = _parse_morphemes(ipu_row["ipu"], tagger, kakasi)
        rows.extend(
            [[ipu_row["filename"], ipu_row["speaker"], ipu_row["ipuID"]] + item for item in morphemes]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "speaker",
            "ipuID",
            "orth",
            "pos",
            "pos1",
            "pos2",
            "pos3",
            "cForm",
            "cType",
            "lemma",
            "ruby",
            "pron",
            "phonemes",
            "nth",
            "len",
        ],
    )


def analyze_ipu_csv(input_csv_path: str | Path, output_csv_path: str | Path | None = None) -> tuple[Path, pd.DataFrame]:
    source = Path(input_csv_path).expanduser().resolve()
    df_ipu = pd.read_csv(source)
    df_morph = analyze_ipus(df_ipu)
    target = Path(output_csv_path).expanduser().resolve() if output_csv_path else source.with_name("morpheme.csv")
    df_morph.to_csv(target, encoding="utf-8_sig", index=False)
    return target, df_morph
