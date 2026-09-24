"""
とらのすけのスモークテスト。

    pytest tests                         # 数秒で終わる基本テスト
    TRANOSUKE_RUN_SLOW=1 pytest tests    # sample.wav でコーパス一括作成まで実行（HFトークンが必要）
"""

import os
from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf

from tranosuke.config import get_app_paths, read_user_config
from tranosuke.media import convert_media_to_wavs


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WAV = REPO_ROOT / "sample" / "sample.wav"
SAMPLE_MP4 = REPO_ROOT / "sample" / "sample_mp4.mp4"


def test_base_dir_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_app_paths().base_dir == REPO_ROOT


def test_convert_stereo_wav(tmp_path):
    result = convert_media_to_wavs(SAMPLE_WAV, output_dir=tmp_path)

    mono_info = sf.info(str(result.mixed_mono_wav))
    assert mono_info.channels == 1
    assert mono_info.samplerate == 16000
    assert [path.name for path in result.channel_wavs] == ["sample_ch1.wav", "sample_ch2.wav"]
    for channel_wav in result.channel_wavs:
        assert sf.info(str(channel_wav)).channels == 1


def test_convert_video(tmp_path):
    result = convert_media_to_wavs(SAMPLE_MP4, output_dir=tmp_path, split_channels=False)
    assert sf.info(str(result.mixed_mono_wav)).duration > 0
    assert result.channel_wavs == []


def test_convert_rejects_file_without_audio(tmp_path):
    not_media = tmp_path / "not_media.mp4"
    not_media.write_text("this is not media")
    with pytest.raises(RuntimeError):
        convert_media_to_wavs(not_media, output_dir=tmp_path)


@pytest.mark.skipif(not get_app_paths().unidic_dir.exists(), reason="UniDic-CSJ が未ダウンロード")
def test_morphology():
    from tranosuke.morphology import analyze_ipus

    df_ipu = pd.DataFrame(
        [["sample", "SPEAKER_00", "IPU", "IPU_0001", 0.0, 1.0, "今日はいい天気です"]],
        columns=["filename", "speaker", "tier", "IPUID", "startTime", "endTime", "IPU"],
    )
    df_morph = analyze_ipus(df_ipu)

    assert len(df_morph) > 0
    assert "".join(df_morph["orth"]) == "今日はいい天気です"
    assert (df_morph["IPUID"] == "IPU_0001").all()
    assert df_morph["phonemes"].notna().all()


@pytest.mark.skipif(os.environ.get("TRANOSUKE_RUN_SLOW") != "1", reason="TRANOSUKE_RUN_SLOW=1 のときだけ実行")
@pytest.mark.skipif(not read_user_config().get("HUGGINGFACE_ACCESS_TOKEN"), reason="HFトークンが未設定")
def test_build_corpus(tmp_path):
    from tranosuke.config import initialize_app
    from tranosuke.corpus import build_corpus

    initialize_app()
    result = build_corpus(SAMPLE_WAV, output_dir=tmp_path, model_name="tiny")

    expected_columns = {
        result.ipu_csv: "filename,speaker,tier,IPUID,startTime,endTime,IPU",
        result.morpheme_csv: "filename,speaker,IPUID,orth,pos,pos1,pos2,pos3,cForm,cType,lemma,ruby,pron,phonemes,nth,len",
        result.word_csv: "filename,speaker,tier,wordID,startTime,endTime,orth,pos,pos1,pos2,pos3,cForm,cType,lemma,ruby,pron,phonemes",
        result.word2ipu_csv: "filename,wordID,IPUID,nth,len",
        result.phoneme_csv: "filename,speaker,tier,phonemeID,startTime,endTime,phoneme",
        result.phoneme2ipu_csv: "filename,phonemeID,IPUID,nth,len",
    }
    for csv_path, columns in expected_columns.items():
        df = pd.read_csv(csv_path)
        assert ",".join(df.columns) == columns, csv_path.name
        assert len(df) > 0, csv_path.name
