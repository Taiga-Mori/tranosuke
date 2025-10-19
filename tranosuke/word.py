import pandas as pd

from utils import *



def calculate_word_times(
    df_morph: pd.DataFrame, 
    df_phon: pd.DataFrame
    ):
    """
    形態素解析結果と音素のforced alignment結果から
    各単語の開始・終了時間を求める。

    Args:
        df_morph: 形態素解析結果のDataFrame
            必須列: ['utteranceID', 'nth', 'phonemes']
        df_phon: 音素のforced alignment結果のDataFrame
            必須列: ['utteranceID', 'startTime', 'endTime', 'phoneme']

    Returns:
        List[List]: [[utteranceID, wordStartTime, wordEndTime, nth], ...]
    """

    result = []

    # UtteranceIDごとに処理
    for utteranceID, morph_group in df_morph.groupby("utteranceID"):
        # この発話の音素のみ抽出
        phonemes_group = df_phon[df_phon["utteranceID"] == utteranceID].copy()
        phonemes_group = phonemes_group.sort_values('startTime')
        phonemes_group = phonemes_group.reset_index(drop=True)
        
        # phonemeの列をリスト化
        phoneme_list = phonemes_group["phoneme"].tolist()
        phoneme_list = [s.lower() for s in phoneme_list]
        start_list = phonemes_group["startTime"].tolist()
        end_list = phonemes_group["endTime"].tolist()

        phoneme_idx = 0

        # 各単語に対して
        morph_group = morph_group.sort_values('nth')
        for _, row in morph_group.iterrows():
            word_phonemes = row['phonemes'].split()  # 例: "h a i" -> ["h","a","i"]
            word_phonemes = [s.lower() for s in word_phonemes]
            word_start = None
            word_end = None

            # 単語の音素とforced alignmentの音素を順番に照合
            match_idx = 0
            while phoneme_idx < len(phoneme_list) and match_idx < len(word_phonemes):
                if phoneme_list[phoneme_idx] == word_phonemes[match_idx]:
                    if match_idx == 0:
                        word_start = start_list[phoneme_idx]
                    word_end = end_list[phoneme_idx]
                    match_idx += 1
                phoneme_idx += 1

            # マッチしなかった場合はスキップ（安全策）
            if word_start is None or word_end is None:
                continue

            result.append([utteranceID, word_start, word_end, row["nth"]])

    return result



def add_word_segment(
        df_morph: pd.DataFrame,
        df_phon: pd.DataFrame,
        ) -> pd.DataFrame:

    data = calculate_word_times(
        df_morph,
        df_phon
    )

    df_temp = pd.DataFrame(
        data,
        columns=["utteranceID", "startTime", "endTime", "nth"]
        )

    df_word = pd.merge(df_morph, df_temp, on=["utteranceID", "nth"], how='left')

    df_word["timestamp"] = df_word["startTime"].apply(float_to_timecode)
    df_word["wordID"] = df_word["timestamp"].astype(str) + df_word["speaker"].astype(str)
    df_word["tier"] = "Word_" + df_word["speaker"].astype(str)
    df_word = df_word[[
        'filename', 'speaker', 'tier', 'utteranceID', 'wordID', 'startTime', 'endTime', 
        'orth', 'pos', 'pos1', 'pos2', 'pos3', 'cForm', 'cType', 'lemma', 'ruby', 'pron', 'phonemes', 'nth', 'len'
        ]]
    
    return df_word
