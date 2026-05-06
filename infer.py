"""Run real-time inference and trigger skipping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from capture import AudioRingBuffer
from features import extract_audio_features, load_clip_model, extract_clip_features, run_asr, run_ocr, _keyword_features
from skipper import send_right_arrow

import mss
import sounddevice as sd
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time skip inference.")
    parser.add_argument("--model", type=str, required=True, help="Trained model path.")
    parser.add_argument("--features", type=str, required=True, help="Feature list JSON path.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Skip probability threshold.")
    parser.add_argument("--debounce", type=int, default=2, help="Consecutive windows required.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size.")
    parser.add_argument("--hop-seconds", type=float, default=1.0, help="Hop size.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index.")
    parser.add_argument("--device", type=str, default=None, help="Sounddevice input device.")
    parser.add_argument("--loopback", action="store_true", help="Enable WASAPI loopback.")
    parser.add_argument("--use-clip", action="store_true", help="Enable CLIP embeddings.")
    parser.add_argument("--use-asr", action="store_true", help="Enable Whisper transcripts.")
    parser.add_argument("--use-ocr", action="store_true", help="Enable OCR.")
    parser.add_argument("--torch-device", type=str, default="cpu", help="torch device for CLIP.")
    parser.add_argument("--dry-run", action="store_true", help="Do not press skip key.")
    return parser.parse_args()


def _audio_callback(buffer: AudioRingBuffer, indata: np.ndarray, _frames: int, _time, _status) -> None:
    buffer.append(indata.copy())


def main() -> None:
    args = _parse_args()

    import joblib

    model = joblib.load(args.model)
    feature_columns = json.loads(Path(args.features).read_text())

    window_samples = int(args.window_seconds * args.sample_rate)
    audio_buffer = AudioRingBuffer(max(window_samples * 3, window_samples + 1), args.channels)

    extra = sd.WasapiSettings(loopback=True) if args.loopback else None
    stream = sd.InputStream(
        samplerate=args.sample_rate,
        channels=args.channels,
        device=args.device,
        dtype="float32",
        callback=lambda indata, frames, time_info, status: _audio_callback(
            audio_buffer, indata, frames, time_info, status
        ),
        extra_settings=extra,
    )
    stream.start()

    clip_model = None
    clip_preprocess = None
    if args.use_clip:
        clip_model, clip_preprocess = load_clip_model(args.torch_device)

    consecutive = 0
    next_time = time.time() + args.hop_seconds

    print("Inference running. Ctrl+C to stop.")
    with mss.mss() as sct:
        monitor = sct.monitors[args.monitor]
        try:
            while True:
                now = time.time()
                if now < next_time:
                    time.sleep(0.01)
                    continue

                audio = audio_buffer.get_last(window_samples)
                if audio is None:
                    next_time += args.hop_seconds
                    continue

                frame = sct.grab(monitor)
                image = Image.frombytes("RGB", frame.size, frame.rgb)

                feature_map = {}
                feature_map.update(extract_audio_features(audio, args.sample_rate))

                if args.use_asr:
                    transcript = run_asr(audio, args.sample_rate)
                    feature_map.update(_keyword_features(transcript, "asr"))

                if args.use_clip:
                    clip_vec = extract_clip_features(image, clip_model, clip_preprocess, args.torch_device)
                    for idx, value in enumerate(clip_vec):
                        feature_map[f"clip_{idx:03d}"] = float(value)

                if args.use_ocr:
                    ocr_text = run_ocr(image)
                    feature_map.update(_keyword_features(ocr_text, "ocr"))

                vector = np.array([float(feature_map.get(col, 0.0)) for col in feature_columns], dtype=np.float32)
                proba = float(model.predict_proba(vector.reshape(1, -1))[0][1])

                if proba >= args.threshold:
                    consecutive += 1
                else:
                    consecutive = 0

                if consecutive >= args.debounce:
                    print(f"Skip triggered (p={proba:.2f}).")
                    if not args.dry_run:
                        send_right_arrow()
                    consecutive = 0

                next_time += args.hop_seconds
        except KeyboardInterrupt:
            print("\nInference stopped.")
        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    main()
