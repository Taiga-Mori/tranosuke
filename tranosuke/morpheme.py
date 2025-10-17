import MeCab
import pykakasi
import numpy as np
import pandas as pd
import re
import platform

try:
    from tranosuke.utils import *
except:
    from utils import *



# 音素モデルのダウンロード
download(
    "unidic-csj-202302",
    "https://clrd.ninjal.ac.jp/unidic_archive/2302/unidic-csj-202302.zip"
)

# kakasiとtaggerの初期化
kks = pykakasi.kakasi()
dic_path = resource_path("unidic-csj-202302")
if platform.system() == "Windows":
    dic_path = dic_path.replace("\\", "/")
tagger = MeCab.Tagger(f"-d {dic_path}")
tagger.parse('')

def kana2roman(kana: str, kks=kks) -> str:
    """
    カナ文字列をローマ字列に変換する。
    """

    kks = pykakasi.kakasi()
    words = kks.convert(kana)
    result = []
    for w in words:
        result.append(w['hepburn'])
    roman = "".join(result)
    return roman

def roman2phonemes(roman: str) -> str:
    """
    ローマ字文字列を音素トークン列に変換する。

    - 出力は音素トークンのリスト（例: ['k','a'] や ['sh','i']、母音は 'a','i','u','e','o'）。
    - 特殊音: 'ん' -> 'N', 促音(っ) -> 'cl', 'ヴ' の子音 -> 'v'
    - 二重子音 (例: 'kk') は先頭に 'cl' を挿入して処理します。

    注意: 別に行われるforced alignmentのために音素トークン列を作成します。
    音素トークンの表記方法については以下を参照して下さい。
    https://github.com/DwangoMediaVillage/pydomino
    """

    s = roman.strip().lower()
    vowels = set("aiueo")

    # 2文字子音（パリャライズや特別なもの）を最初に扱う
    two_onsets = {"ky","gy","sh","ch","ny","hy","by","py","my","ry","ts"}
    # 単一子音（yもここで扱うが 'ya' は y + a）
    consonants = set(list("bcdfghjklmnpqrstvwxyz"))
    # 出力トークン
    toks = []
    i = 0
    n = len(s)

    while i < n:
        # 1) 末尾の 'n' -> ン (N)
        if s[i] == "n":
            # 次が存在しないか、次が母音や 'y' でない場合は N
            if i + 1 == n or (s[i+1] not in vowels and s[i+1] != "y"):
                toks.append("N")
                i += 1
                continue
            # 次が母音または y の場合は通常次の音節の一部になる -> fallthrough

        # 2) 促音（っ）相当: 二重子音の先頭。ルール: 現在と次の文字が同じ子音 (母音でない) なら促音
        if i + 1 < n and s[i] == s[i+1] and s[i] in consonants and s[i] != "n":
            toks.append("cl")
            # スキップして次の子音から音節化する（先頭子音は残すので i+=1）
            i += 1
            continue

        # 3) 3文字パターン： two_onset + vowel （ex: 'kya','shu','cha','tsu'）
        if i + 2 < n:
            onset2 = s[i:i+2]
            v = s[i+2]
            if onset2 in two_onsets and v in vowels:
                toks.append(onset2)
                toks.append(v)
                i += 3
                continue

        # 4) 2文字パターン： 単独子音 + 母音（ex: 'ka','ji','ba'）または 'ts'+'u' (handled by two_onsets)
        if i + 1 < n:
            c = s[i]
            v = s[i+1]
            if c in consonants and v in vowels:
                # 'j' は 'ji','ju','jo' などで単体子音 j として扱う
                toks.append(c)
                toks.append(v)
                i += 2
                continue

        # 5) 母音単独（あいうえお）
        if s[i] in vowels:
            toks.append(s[i])
            i += 1
            continue

        # 6) 未知・例外処理: 単文字を取り出して進める（安全策）
        toks.append(s[i])
        i += 1

    phonemes = " ".join(toks)
    return phonemes

def parse_morphemes(text: str, tagger=tagger, kks=kks) -> list:
    node = tagger.parseToNode(text)
    morphemes = []
    while node:
        # 表層系
        surface = node.surface
        # 形態素解析結果
        morpheme = node.feature.split(',')
        morpheme.insert(0, surface)
        # 文頭・文末記号ではないかつ補助記号ではないなら
        if morpheme[1] != u'BOS/EOS' and morpheme[1] != u'補助記号':
            # 未知語ではないなら(解析結果が30個なら)
            if len(morpheme) >= 15:
                # "orth", "pos", "pos1", "pos2", "pos3", "cForm", "cType", "lemma", "ruby", "pron"
                morpheme = [morpheme[i] for i in [0, 1, 2, 3, 4, 6, 5, 8, 7, 10]]
            else:
                morpheme =  [morpheme[i] for i in [0, 1, 2, 3, 4]] + [np.nan for i in range(0, 5)]
        
            morpheme = [np.nan if m == "*" else m for m in morpheme]

            # 未知語でpronがないならorthをカタカナに変換
            if pd.isna(morpheme[-1]):
                result = kks.convert(morpheme[0])
                kana = []
                for r in result:
                    kana.append(r["kana"])
                kana = "".join(kana)
                morpheme[-1] = kana

            # 音素追加
            roman = kana2roman(morpheme[-1])
            phonemes = roman2phonemes(roman)
            # もしphonemesがアルファベットと半角スペース以外なら
            if bool(re.search(r"[^A-Za-z ]", phonemes)):
                phonemes = ""
            # もしorthがマイクロポーズ記号ならphonemeはpauとする
            if morpheme[0] == "¥":
                phonemes = "pau"
            morpheme.append(phonemes)

            morphemes.append(morpheme)
        node = node.next
  
    # 位置と総数を追加
    length = len(morphemes)
    for nth, sublist in enumerate(morphemes, start=1):
        sublist.append(nth)
        sublist.append(length)
    return morphemes

def morph_analyze(df_utt: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: 発話の書き起こしのデータフレーム
    
    Returns:
        pandas.DataFrame で列 ["file", "utteranceID", "startTime", "endTime", "utterance"] （start,end は秒, float）
    """

    result = []
    for _, row in df_utt.iterrows():
        filename = row["filename"]
        utteranceID = row["utteranceID"]
        utterance = row["utterance"]

        if not pd.notna(utterance):
            continue

        morphemes = parse_morphemes(utterance)
        result = result  + [[filename, utteranceID] + sublist for sublist in morphemes]
    
    df_morph = pd.DataFrame(
        result,
        columns=["filename", "utteranceID", "orth", "pos", "pos1", "pos2", "pos3", "cForm", "cType", "lemma", "ruby", "pron", "phonemes", "nth", "len"]
        )
    
    return df_morph

if __name__ == '__main__':
    print(parse_morphemes(text="すもももももものうち", tagger=tagger, kks=kks))