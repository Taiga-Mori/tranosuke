import json
import os
import sys
from pathlib import Path

import librosa
import pydomino
import soundfile as sf


def _load_segment(wav_path: Path):
    audio_segment, sample_rate = sf.read(wav_path)
    if getattr(audio_segment, "ndim", 1) > 1:
        audio_segment = audio_segment.mean(axis=1)
    return librosa.resample(audio_segment.astype("float32"), orig_sr=sample_rate, target_sr=16000)


def serve(model_path: str) -> int:
    """標準入力から1行1リクエストの JSON を受け取り、アラインメント結果を1行1レスポンスで返し続ける。

    pydomino の C++ 側は標準出力にログを出すので、応答用に元の標準出力を複製してから
    標準出力そのものは標準エラーへ向ける。セグメンテーション違反ではプロセスごと落ちるので、
    呼び出し側が検出して起動し直す。
    """
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    aligner = pydomino.Aligner(model_path)

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            alignment = aligner.align(
                _load_segment(Path(request["segment_path"])),
                request["phoneme_sequence"],
                int(request["min_aligned_timeframe"]),
            )
            response = {"ok": True, "alignment": alignment}
        except Exception as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        protocol.write(json.dumps(response) + "\n")
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--serve":
        print("usage: alignment_worker.py --serve phoneme_transition_model.onnx", file=sys.stderr)
        return 2
    return serve(sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
