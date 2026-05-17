"""Run real-time inference and trigger skipping."""

from __future__ import annotations

import argparse
import csv
import json
import time
import threading
import os
import traceback
import atexit
import signal
import sys
import faulthandler
import multiprocessing as mp
import queue
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "TRUE")
os.environ.setdefault("TORCH_HOME", "model_cache")
os.environ.setdefault("HF_HOME", "model_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "model_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "model_cache")

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


_DEBUG_LOG_FILE = None


def _init_debug_log(path: str | None) -> None:
    global _DEBUG_LOG_FILE
    if not path:
        return

    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _DEBUG_LOG_FILE = log_path.open("a", encoding="utf-8")
    _DEBUG_LOG_FILE.write("\n--- infer debug start ---\n")
    _DEBUG_LOG_FILE.flush()

    faulthandler.enable(file=_DEBUG_LOG_FILE)

    def _hook(exc_type, exc, tb) -> None:
        _DEBUG_LOG_FILE.write("\nUnhandled exception:\n")
        traceback.print_exception(exc_type, exc, tb, file=_DEBUG_LOG_FILE)
        _DEBUG_LOG_FILE.flush()

    def _thread_hook(args) -> None:
        _DEBUG_LOG_FILE.write("\nThread exception:\n")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_DEBUG_LOG_FILE)
        _DEBUG_LOG_FILE.flush()

    sys.excepthook = _hook
    try:
        threading.excepthook = _thread_hook
    except AttributeError:
        pass


def _install_exit_logging() -> None:
    def _on_exit() -> None:
        print("Inference exiting.")
        if _DEBUG_LOG_FILE is not None:
            _DEBUG_LOG_FILE.write("Inference exiting.\n")
            _DEBUG_LOG_FILE.flush()
            _DEBUG_LOG_FILE.close()

    def _on_signal(signum, _frame) -> None:
        print(f"Received signal {signum}.")

    atexit.register(_on_exit)
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _on_signal)
        except Exception:
            continue


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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio device name or index for capture (sounddevice/soundcard).",
    )
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
        "--asr-mode",
        type=str,
        default="subprocess",
        choices=["inprocess", "subprocess"],
        help="Run ASR in-process or in a separate subprocess.",
    )
    parser.add_argument(
        "--asr-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for ASR before skipping the transcript.",
    )
    parser.add_argument(
        "--asr-echo",
        action="store_true",
        help="Print ASR transcripts for validation.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="model_cache",
        help="Directory to store downloaded model caches (CLIP/ASR).",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default="infer_debug.log",
        help="Write debug logs to this file.",
    )
    parser.add_argument(
        "--ignore-skip-errors",
        action="store_true",
        help="Log SendInput failures instead of stopping.",
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


def _warm_asr(sample_rate: int) -> None:
    silence = np.zeros(sample_rate, dtype=np.float32)
    try:
        run_asr(silence, sample_rate)
    except Exception as exc:
        print(f"ASR warm-up failed: {exc}")
        traceback.print_exc()


def _asr_worker_main(in_q: mp.Queue, out_q: mp.Queue, device: str, compute_type: str, sample_rate: int) -> None:
    set_asr_runtime(device, compute_type)
    while True:
        item = in_q.get()
        if item is None:
            break
        try:
            transcript = run_asr(item, sample_rate)
            out_q.put(("ok", transcript))
        except Exception as exc:
            out_q.put(("err", repr(exc)))


class _AsrWorker:
    def __init__(self, device: str, compute_type: str, sample_rate: int, timeout: float) -> None:
        self._device = device
        self._compute_type = compute_type
        self._sample_rate = sample_rate
        self._timeout = timeout
        self._ctx = mp.get_context("spawn")
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None
        self._proc: mp.Process | None = None

    def start(self) -> None:
        self._in_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_asr_worker_main,
            args=(self._in_q, self._out_q, self._device, self._compute_type, self._sample_rate),
            daemon=True,
        )
        self._proc.start()

    def stop(self) -> None:
        if self._in_q is not None:
            self._in_q.put(None)
        if self._proc is not None:
            self._proc.join(timeout=1)

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self.start()

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_alive()
        if self._in_q is None or self._out_q is None:
            return ""
        self._in_q.put(audio)
        try:
            status, payload = self._out_q.get(timeout=self._timeout)
        except queue.Empty:
            return ""
        if status != "ok":
            print(f"ASR subprocess error: {payload}")
            return ""
        return payload


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


def _normalize_soundcard_device(device: str | None) -> str | None:
    if device is None:
        return None

    value = device.strip().lower()
    if value in {"auto", "cpu", "cuda"}:
        print("Note: --device is for audio devices; ignoring value.")
        return None

    return device


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
        except Exception as exc:
            print(f"Soundcard recorder error: {exc}")
            traceback.print_exc()
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
    _init_debug_log(args.debug_log)
    _install_exit_logging()

    _apply_model_cache_dir(args.model_cache_dir)
    _resolve_asr_runtime(args.asr_device, args.asr_compute_type)

    import joblib

    model = joblib.load(args.model)
    feature_columns = json.loads(Path(args.features).read_text())

    dry_run_dir = None
    dry_run_csv = None
    dry_run_writer = None
    if args.dry_run:
        dry_run_dir = Path("dry_run")
        dry_run_dir.mkdir(parents=True, exist_ok=True)
        dry_run_csv = (dry_run_dir / "skips.csv").open("a", newline="")
        dry_run_writer = csv.DictWriter(dry_run_csv, fieldnames=["timestamp", "proba", "image_path"])
        if dry_run_csv.tell() == 0:
            dry_run_writer.writeheader()

    clip_model = None
    clip_preprocess = None
    if args.use_clip:
        clip_model, clip_preprocess = load_clip_model(args.torch_device)

    if args.use_asr:
        if args.asr_mode == "inprocess":
            _warm_asr(args.sample_rate)
        else:
            print("ASR running in subprocess mode.")

    window_samples = int(args.window_seconds * args.sample_rate)
    audio_buffer = AudioRingBuffer(max(window_samples * 3, window_samples + 1), args.channels)

    stream = None
    stop_event = None
    record_thread = None
    asr_worker = None
    if args.use_asr and args.asr_mode == "subprocess":
        asr_worker = _AsrWorker(args.asr_device, args.asr_compute_type, args.sample_rate, args.asr_timeout)
        asr_worker.start()
    if args.audio_backend == "soundcard":
        soundcard_device = _normalize_soundcard_device(args.device)
        stop_event, record_thread = _start_soundcard_recorder(
            audio_buffer, args.sample_rate, args.channels, soundcard_device
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

    consecutive = 0
    next_time = time.time() + args.hop_seconds
    last_heartbeat = time.time()

    print("Inference running. Ctrl+C to stop.")
    with mss.mss() as sct:
        monitor = sct.monitors[args.monitor]
        try:
            while True:
                now = time.time()
                if now < next_time:
                    time.sleep(0.01)
                    continue

                if now - last_heartbeat >= 30.0:
                    print("Inference heartbeat: running.")
                    last_heartbeat = now

                if record_thread is not None and not record_thread.is_alive():
                    raise RuntimeError("Soundcard recorder thread stopped unexpectedly.")

                audio = audio_buffer.get_last(window_samples)
                if audio is None:
                    next_time += args.hop_seconds
                    continue

                frame = sct.grab(monitor)
                image = Image.frombytes("RGB", frame.size, frame.rgb)

                feature_map = {}
                feature_map.update(extract_audio_features(audio, args.sample_rate))

                if args.use_asr:
                    if asr_worker is not None:
                        transcript = asr_worker.transcribe(audio)
                    else:
                        transcript = run_asr(audio, args.sample_rate)
                    if args.asr_echo and transcript.strip():
                        print(f"ASR: {transcript.strip()}")
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
                    if args.dry_run:
                        print("Dry run: skip detected (no keypress sent).")
                        if dry_run_dir is not None and dry_run_writer is not None:
                            ts = time.time()
                            image_name = f"skip_{int(ts * 1000)}.jpg"
                            image_path = dry_run_dir / image_name
                            image.save(image_path, "JPEG", quality=90)
                            dry_run_writer.writerow(
                                {
                                    "timestamp": f"{ts:.3f}",
                                    "proba": f"{proba:.3f}",
                                    "image_path": str(image_path),
                                }
                            )
                            dry_run_csv.flush()
                    else:
                        try:
                            send_right_arrow()
                        except OSError as exc:
                            if args.ignore_skip_errors:
                                print(f"Skip key failed: {exc}")
                            else:
                                raise
                    consecutive = 0

                next_time += args.hop_seconds
        except KeyboardInterrupt:
            print("\nInference stopped.")
        except Exception as exc:
            print(f"Inference error: {exc}")
            traceback.print_exc()
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            if stop_event is not None:
                stop_event.set()
            if record_thread is not None:
                record_thread.join(timeout=1)
            if asr_worker is not None:
                asr_worker.stop()
            if dry_run_csv is not None:
                dry_run_csv.close()


if __name__ == "__main__":
    main()
