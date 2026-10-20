from pathlib import Path
import re

import MeCab
import numpy as np
import pandas as pd
import pykakasi

from tranosuke.config import get_app_paths


def kana_to_roman(kana: str, kakasi) -> str:
    return "".join(word["hepburn"] for word in kakasi.convert(kana))


def kana_to_phonemes(kana: str, kakasi) -> str:
    # ローマ字では「ン＋ヤ行」と「ニャ行」が区別できない（キンヨー/kinyoo）ため、「ン」は仮名の段階で N にする
    # 「ンー」の長音は撥音を伸ばしたものとして N を重ねる
    kana = re.sub("[ンん]ー+", lambda match: "ン" * len(match.group(0)), kana)
    tokens = []
    for index, piece in enumerate(re.split("[ンん]", kana)):
        if index > 0:
            tokens.append("N")
        phonemes = _kana_piece_to_phonemes(piece, kakasi)
        if phonemes:
            tokens.append(phonemes)
    return " ".join(tokens)


def _kana_piece_to_phonemes(kana: str, kakasi) -> str:
    # pykakasi は語末の「ッ」を "tsu" と読むため、切り離して促音 cl として付け足す
    # pydomino は cl → cl の遷移を受け付けないので、連続していても cl は1つにする
    stripped = kana.rstrip("ッっ")
    trailing_cl = ["cl"] if len(stripped) < len(kana) else []
    phonemes = roman_to_phonemes(kana_to_roman(stripped, kakasi)) if stripped else ""
    return " ".join([item for item in [phonemes] if item] + trailing_cl)


def roman_to_phonemes(roman: str) -> str:
    text = roman.strip().lower()
    vowels = set("aiueo")
    two_onsets = {"ky", "gy", "sh", "ch", "ny", "hy", "by", "py", "my", "ry", "ts"}
    consonants = set("bcdfghjklmnpqrstvwxyz")

    tokens = []
    index = 0
    while index < len(text):
        # ヘボン式は「ン＋母音/ヤ行」を "gin'iro" のように区切る。n は直後の ' により撥音 N になるので ' 自体は捨てる
        if text[index] == "'":
            index += 1
            continue

        if text[index] == "n":
            if index + 1 == len(text) or (text[index + 1] not in vowels and text[index + 1] != "y"):
                tokens.append("N")
                index += 1
                continue

        # ヘボン式では「っち」が "tchi" になるため、"tch" の t も促音として扱う
        if (
            index + 1 < len(text)
            and (text[index] == text[index + 1] or text[index:index + 3] == "tch")
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


# _parse_morphemes が作る形態素リストの位置: 0 表層形(orth), 9 発音形(pron), 10 音素
# 8 の ruby は語彙素読みなので、引き延ばしでは変えない
SURFACE_INDEX, PRON_INDEX, PHONEMES_INDEX = 0, 9, 10


def _pron_to_phonemes(pron: str, kakasi) -> str:
    phonemes = kana_to_phonemes(pron, kakasi)
    if re.search(r"[^A-Za-z ]", phonemes):
        return ""
    return phonemes


def _append_prolongation(morpheme: list, prolongation: str, kakasi) -> None:
    for index in [SURFACE_INDEX, PRON_INDEX]:
        if isinstance(morpheme[index], str):
            morpheme[index] += prolongation
    morpheme[PHONEMES_INDEX] = _pron_to_phonemes(morpheme[PRON_INDEX], kakasi)


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
            if pd.isna(morpheme[-1]):
                morpheme[-1] = "".join(item["kana"] for item in kakasi.convert(morpheme[0]))

            if re.fullmatch("ー+", morpheme[0]):
                # 単独の伸ばし棒は直前の形態素の引き延ばしとして連結する（先頭にある場合は伸ばす対象がないので捨てる）
                if morphemes:
                    _append_prolongation(morphemes[-1], morpheme[0], kakasi)
                node = node.next
                continue

            morpheme.append(_pron_to_phonemes(morpheme[-1], kakasi))
            morphemes.append(morpheme)

        node = node.next

    length = len(morphemes)
    for nth, morpheme in enumerate(morphemes, start=1):
        morpheme.append(nth)
        morpheme.append(length)

    return morphemes


def _resolve_mecab_dictionary_path(dic_path: Path) -> Path:
    if (dic_path / "dicrc").exists():
        return dic_path

    for child in dic_path.iterdir():
        if child.is_dir() and (child / "dicrc").exists():
            return child

    return dic_path


def analyze_ipus(df_ipu: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IPU rows into morpheme rows with phoneme strings.
    """
    kakasi = pykakasi.kakasi()
    paths = get_app_paths()
    dic_path = _resolve_mecab_dictionary_path(paths.unidic_dir)
    if paths.system == "Windows":
        dic_path = Path(str(dic_path).replace("\\", "/"))

    tagger = MeCab.Tagger(f"-r /dev/null -d {dic_path}")
    tagger.parse("")

    rows = []
    for _, ipu_row in df_ipu.iterrows():
        if not pd.notna(ipu_row["IPU"]):
            continue
        morphemes = _parse_morphemes(ipu_row["IPU"], tagger, kakasi)
        rows.extend(
            [[ipu_row["filename"], ipu_row["speaker"], ipu_row["IPUID"]] + item for item in morphemes]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "speaker",
            "IPUID",
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
