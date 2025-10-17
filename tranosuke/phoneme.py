import pydomino
import librosa
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
import soundfile as sf
import os

try:
    from tranosuke.utils import *
except:
    from utils import *



# 辞書のダウンロード
download(
    "phoneme_transition_model.onnx",
    "https://github.com/DwangoMediaVillage/pydomino/raw/main/onnx_model/phoneme_transition_model.onnx"
)

model_path = resource_path("phoneme_transition_model.onnx")
aligner = pydomino.Aligner(model_path)

def parse_phonemes(wav_path: str, phoneme_sequence: str, iterations: int, aligner=aligner):

    y, _ = librosa.load(wav_path, sr=16000, mono=True, dtype=np.float32)
    alignment = aligner.align(y, phoneme_sequence, iterations)
    phonemes = [x for x in alignment if x[-1] != "pau"]

    return phonemes

def forced_align(
        audio_path: PosixPath,
        df_utt: pd.DataFrame,
        df_morph: pd.DataFrame,
        iterations: int,
        ) -> pd.DataFrame:

    temp_path = "temp.wav"
    data = []

    for _, row in df_utt.iterrows():

        filename = row["filename"]
        utteranceID = row["utteranceID"]
        utt_start_time = row["startTime"]
        utt_end_time = row["endTime"]
        utterance = row["utterance"]

        if not pd.isna(utterance):

            phoneme_sequence = " ".join(df_morph[df_morph["utteranceID"] == utteranceID]["phonemes"].values)

            # 発話中の音声の切り出し
            y, sr = sf.read(audio_path)  # y.shape -> (サンプル数, チャンネル数)

            # Whisperは前にずれている傾向にあるので開始と終了を少し遅くする
            start_delay = 0.2
            end_buffa = 0.2

            start_sample = int((utt_start_time + start_delay) * sr)
            end_sample   = int((utt_end_time + end_buffa) * sr)

            cut_channel = y[start_sample:end_sample]

            sf.write(temp_path, cut_channel, sr)

            # Forced alignment
            try:
                phonemes = parse_phonemes(temp_path, f"pau {phoneme_sequence} pau", iterations)

                for (_start_time, _end_time, phoneme) in phonemes:

                    phon_start_time = float(round(_start_time + utt_start_time + start_delay, 4))
                    phon_end_time = float(round(_end_time + utt_start_time + start_delay, 4))

                    data.append([filename, utteranceID, phon_start_time, phon_end_time, phoneme])

            except Exception as e:
                print(e)

    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    df_phon = pd.DataFrame(
        data,
        columns=["filename", "utteranceID", "startTime", "endTime", "phoneme"]
        )

    df_phon["timestamp"] = df_phon["startTime"].apply(float_to_timecode)
    df_phon["phonemeID"] = df_phon["timestamp"].astype(str) + filename
    df_phon["tier"] = f"Phoneme_{filename}"
    df_phon = df_phon[["filename", "tier", "utteranceID", "phonemeID", "startTime", "endTime", "phoneme"]]

    return df_phon