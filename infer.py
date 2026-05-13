"""Run real-time inference and trigger skipping."""

from __future__ import annotations

import argparse
import json
import time
import threading
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from capture import AudioRingBuffer
from features import (
    extract_audio_features,
    load_clip_model,
    extract_clip_features,
    run_asr,
    run_ocr,
    _keyword_features,
    set_asr_runtime,
)
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
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="sounddevice",
        choices=["sounddevice", "soundcard"],
        help="Audio backend: sounddevice (default) or soundcard for loopback.",
    )
    parser.add_argument("--use-clip", action="store_true", help="Enable CLIP embeddings.")
    parser.add_argument("--use-asr", action="store_true", help="Enable Whisper transcripts.")
    parser.add_argument("--use-ocr", action="store_true", help="Enable OCR.")
    parser.add_argument("--torch-device", type=str, default="cpu", help="torch device for CLIP.")
    parser.add_argument("--dry-run", action="store_true", help="Do not press skip key.")
    parser.add_argument(
        "--asr-device",
        type=str,
        default="auto",
        help="ASR device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--asr-compute-type",
        type=str,
        default="auto",
        help="ASR compute type: auto, int8, float16, or float32.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="model_cache",
        help="Directory to store downloaded model caches (CLIP/ASR).",
    )
    return parser.parse_args()


def _apply_model_cache_dir(cache_dir: str | None) -> None:
    if not cache_dir:
        return

    cache_path = Path(cache_dir).expanduser().resolve()
    os.makedirs(cache_path, exist_ok=True)
    cache_value = str(cache_path)
    os.environ["TORCH_HOME"] = cache_value
    os.environ["HF_HOME"] = cache_value
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_value
    os.environ["TRANSFORMERS_CACHE"] = cache_value


def _resolve_asr_runtime(device: str, compute_type: str) -> None:
    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"

    resolved_compute = compute_type
    if compute_type == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"

    set_asr_runtime(resolved_device, resolved_compute)


def _audio_callback(buffer: AudioRingBuffer, indata: np.ndarray, _frames: int, _time, _status) -> None:
    buffer.append(indata.copy())


def _get_wasapi_settings(loopback: bool) -> object | None:
    if not loopback:
        return None
    try:
        return sd.WasapiSettings(loopback=True)
    except TypeError:
        settings = sd.WasapiSettings()
        if hasattr(settings, "loopback"):
            settings.loopback = True
        return settings


def _select_loopback_device(device: str | None) -> str | int | None:
    if device is not None:
        return device

    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
    except Exception:
        return device

    wasapi_index = None
    for index, hostapi in enumerate(hostapis):
        if hostapi.get("name", "").lower().startswith("wasapi"):
            wasapi_index = index
            break

    if wasapi_index is None:
        return device

    output_device = hostapis[wasapi_index].get("default_output_device")
    if output_device is None or output_device < 0:
        return device

    return output_device


def _resolve_device(device: str | None) -> str | int | None:
    if device is None:
        return None

    value = device.strip()
    if value.isdigit():
        return int(value)

    try:
        devices = sd.query_devices()
    except Exception:
        return device

    for index, info in enumerate(devices):
        if info.get("name") == value:
            return index

    for index, info in enumerate(devices):
        if value.lower() in info.get("name", "").lower():
            return index

    return device


def _select_soundcard_loopback(device: str | None):
    import soundcard as sc

    if device:
        return sc.get_microphone(device, include_loopback=True)

    default_speaker = sc.default_speaker()
    return sc.get_microphone(default_speaker.name, include_loopback=True)


def _start_soundcard_recorder(
    buffer: AudioRingBuffer, sample_rate: int, channels: int, device: str | None
) -> tuple[threading.Event, threading.Thread]:
    import soundcard as sc

    stop_event = threading.Event()
    block_samples = max(1, int(sample_rate * 0.1))
    loopback = _select_soundcard_loopback(device)

    def _run() -> None:
        initialized = False
        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
                initialized = True
            except Exception:
                import ctypes

                if ctypes.windll.ole32.CoInitialize(None) == 0:
                    initialized = True

            with loopback.recorder(samplerate=sample_rate, channels=channels) as recorder:
                while not stop_event.is_set():
                    data = recorder.record(numframes=block_samples)
                    buffer.append(data.astype(np.float32))
        finally:
            if initialized:
                try:
                    import pythoncom

                    pythoncom.CoUninitialize()
                except Exception:
                    import ctypes

                    ctypes.windll.ole32.CoUninitialize()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop_event, thread


def main() -> None:
    args = _parse_args()

    _apply_model_cache_dir(args.model_cache_dir)
    _resolve_asr_runtime(args.asr_device, args.asr_compute_type)

    import joblib

    model = joblib.load(args.model)
    feature_columns = json.loads(Path(args.features).read_text())

    window_samples = int(args.window_seconds * args.sample_rate)
    audio_buffer = AudioRingBuffer(max(window_samples * 3, window_samples + 1), args.channels)

    stream = None
    stop_event = None
    record_thread = None
    if args.audio_backend == "soundcard":
        stop_event, record_thread = _start_soundcard_recorder(
            audio_buffer, args.sample_rate, args.channels, args.device
        )
    else:
        extra = _get_wasapi_settings(args.loopback)
        stream = sd.InputStream(
            samplerate=args.sample_rate,
            channels=args.channels,
            device=_select_loopback_device(args.device) if args.loopback else _resolve_device(args.device),
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
            if stream is not None:
                stream.stop()
                stream.close()
            if stop_event is not None:
                stop_event.set()
            if record_thread is not None:
                record_thread.join(timeout=1)


if __name__ == "__main__":
    main()
