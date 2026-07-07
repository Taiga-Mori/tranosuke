import json
import sys
from pathlib import Path

import librosa
import pydomino
import soundfile as sf


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: alignment_worker.py request.json segment.wav output.json", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    wav_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    request = json.loads(request_path.read_text())

    audio_segment, sample_rate = sf.read(wav_path)
    if getattr(audio_segment, "ndim", 1) > 1:
        audio_segment = audio_segment.mean(axis=1)
    resampled = librosa.resample(audio_segment.astype("float32"), orig_sr=sample_rate, target_sr=16000)

    aligner = pydomino.Aligner(str(request["model_path"]))
    alignment = aligner.align(resampled, request["phoneme_sequence"], int(request["iterations"]))
    output_path.write_text(json.dumps(alignment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
