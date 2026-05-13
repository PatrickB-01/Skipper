"""Capture screen frames + audio windows into a session folder."""

from __future__ import annotations

import argparse
import csv
import json
import time
import threading
from pathlib import Path

import mss
import numpy as np
import sounddevice as sd
from PIL import Image


class AudioRingBuffer:
    def __init__(self, capacity_samples: int, channels: int) -> None:
        self.capacity = capacity_samples
        self.channels = channels
        self.buffer = np.zeros((capacity_samples, channels), dtype=np.float32)
        self.index = 0
        self.filled = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] != self.channels:
            raise ValueError("Audio channel mismatch.")

        n = samples.shape[0]
        if n >= self.capacity:
            samples = samples[-self.capacity :]
            n = samples.shape[0]

        end = self.index + n
        if end <= self.capacity:
            self.buffer[self.index : end] = samples
        else:
            first = self.capacity - self.index
            self.buffer[self.index :] = samples[:first]
            self.buffer[: end % self.capacity] = samples[first:]

        self.index = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def get_last(self, n_samples: int) -> np.ndarray | None:
        if self.filled < n_samples:
            return None
        start = (self.index - n_samples) % self.capacity
        if start < self.index:
            return self.buffer[start : self.index].copy()
        return np.vstack((self.buffer[start:], self.buffer[: self.index])).copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture screen + audio into a session folder.")
    parser.add_argument("--session-dir", type=str, required=True, help="Output session folder.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size in seconds.")
    parser.add_argument("--hop-seconds", type=float, default=1.0, help="Hop size in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for screen capture.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for saved frames (1-95).",
    )
    parser.add_argument("--device", type=str, default=None, help="Sounddevice input device name or index.")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="sounddevice",
        choices=["sounddevice", "soundcard"],
        help="Audio backend: sounddevice (default) or soundcard for loopback.",
    )
    parser.add_argument("--loopback", action="store_true", help="Enable WASAPI loopback on Windows.")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture.")
    parser.add_argument("--no-video", action="store_true", help="Disable video capture.")
    return parser.parse_args()


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
    if args.window_seconds <= 0 or args.hop_seconds <= 0:
        raise ValueError("--window-seconds and --hop-seconds must be > 0.")
    if not 1 <= args.jpeg_quality <= 95:
        raise ValueError("--jpeg-quality must be between 1 and 95.")

    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    audio_dir = session_dir / "audio"
    frames_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    meta = {
        "start_time": time.time(),
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "monitor": args.monitor,
        "device": args.device,
        "loopback": args.loopback,
        "no_audio": args.no_audio,
        "no_video": args.no_video,
    }
    (session_dir / "session.json").write_text(json.dumps(meta, indent=2))

    manifest_path = session_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["window_id", "timestamp", "frame_path", "audio_path"],
        )
        writer.writeheader()

        audio_buffer = None
        stream = None
        stop_event = None
        record_thread = None
        if not args.no_audio:
            window_samples = int(args.window_seconds * args.sample_rate)
            buffer_capacity = max(window_samples * 3, window_samples + 1)
            audio_buffer = AudioRingBuffer(buffer_capacity, args.channels)

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

        print("Capture running. Press Ctrl+C to stop.")
        next_time = time.time() + args.hop_seconds
        window_id = 0

        with mss.mss() as sct:
            monitor = sct.monitors[args.monitor]
            try:
                while True:
                    now = time.time()
                    if now < next_time:
                        time.sleep(0.01)
                        continue

                    window_id += 1
                    timestamp = now
                    frame_path = ""
                    audio_path = ""

                    if not args.no_video:
                        frame = sct.grab(monitor)
                        image = Image.frombytes("RGB", frame.size, frame.rgb)
                        frame_path = str(frames_dir / f"frame_{window_id:06d}.jpg")
                        image.save(frame_path, "JPEG", quality=args.jpeg_quality)

                    if not args.no_audio:
                        audio = audio_buffer.get_last(window_samples) if audio_buffer else None
                        if audio is None:
                            next_time += args.hop_seconds
                            continue
                        audio_path = str(audio_dir / f"audio_{window_id:06d}.npy")
                        np.save(audio_path, audio)

                    writer.writerow(
                        {
                            "window_id": window_id,
                            "timestamp": f"{timestamp:.3f}",
                            "frame_path": frame_path,
                            "audio_path": audio_path,
                        }
                    )
                    manifest_file.flush()
                    next_time += args.hop_seconds
            except KeyboardInterrupt:
                print("\nCapture stopped.")
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
