"""Play a captured .npy audio window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a recorded audio .npy file.")
    parser.add_argument("--path", type=str, required=True, help="Path to .npy audio file.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--device", type=str, default=None, help="Output device name or index.")
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio = np.load(path)
    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    print(f"Playing {path.name} ({audio.shape[0]} samples, {channels} channel(s))")
    sd.play(audio, samplerate=args.sample_rate, device=_resolve_device(args.device))
    sd.wait()


if __name__ == "__main__":
    main()
