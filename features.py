"""Extract audio/video/text features from a capture session."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import librosa
import numpy as np
from PIL import Image


KEYWORDS = [
    "intro",
    "opening",
    "previously",
    "recap",
    "credits",
    "ending",
    "next time",
    "promo",
    "sponsored",
]


def _keyword_features(text: str, prefix: str) -> dict[str, float]:
    lower = text.lower()
    features = {f"{prefix}_len": float(len(text))}
    for keyword in KEYWORDS:
        features[f"{prefix}_kw_{keyword.replace(' ', '_')}"] = float(lower.count(keyword))
    return features


def extract_audio_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    rms = librosa.feature.rms(y=audio)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    features: dict[str, float] = {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

    for idx in range(mfcc.shape[0]):
        features[f"mfcc_mean_{idx:02d}"] = float(np.mean(mfcc[idx]))
        features[f"mfcc_std_{idx:02d}"] = float(np.std(mfcc[idx]))

    return features


def load_clip_model(device: str = "cpu"):
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.to(device)
    model.eval()
    return model, preprocess


def extract_clip_features(image: Image.Image, model, preprocess, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0).to(device)
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().reshape(-1)


_ASR_MODEL = None
_ASR_DOWNLOAD_ROOT = None

def run_asr(audio: np.ndarray, sample_rate: int) -> str:
    from faster_whisper import WhisperModel

    global _ASR_MODEL
    if _ASR_MODEL is None:
        # "base" balances speed and accuracy; change to "tiny"/"small" as needed.
        _ASR_MODEL = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=_ASR_DOWNLOAD_ROOT,
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    segments, _info = _ASR_MODEL.transcribe(audio, language="en")
    return " ".join(segment.text for segment in segments)


def run_ocr(image: Image.Image) -> str:
    import pytesseract

    return pytesseract.image_to_string(image)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from a session.")
    parser.add_argument("--session-dir", type=str, required=True, help="Session folder.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--use-clip", action="store_true", help="Enable CLIP embeddings.")
    parser.add_argument("--use-asr", action="store_true", help="Enable Whisper transcript keywords.")
    parser.add_argument("--use-ocr", action="store_true", help="Enable OCR text keywords.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for CLIP.")
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Directory to store downloaded model caches (CLIP/ASR).",
    )
    return parser.parse_args()


def _apply_model_cache_dir(cache_dir: str | None) -> None:
    if not cache_dir:
        return

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir


def main() -> None:
    args = _parse_args()

    _apply_model_cache_dir(args.model_cache_dir)
    global _ASR_DOWNLOAD_ROOT
    _ASR_DOWNLOAD_ROOT = args.model_cache_dir
    session_dir = Path(args.session_dir)
    manifest_path = session_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.csv not found in session folder.")

    clip_model = None
    clip_preprocess = None
    if args.use_clip:
        clip_model, clip_preprocess = load_clip_model(args.device)

    features_path = session_dir / "features.csv"
    with manifest_path.open("r", newline="") as manifest_file, features_path.open(
        "w", newline=""
    ) as features_file:
        reader = csv.DictReader(manifest_file)
        fieldnames: list[str] = ["window_id", "timestamp"]
        writer = None

        for row in reader:
            window_id = row["window_id"]
            timestamp = float(row["timestamp"])
            frame_path = row["frame_path"]
            audio_path = row["audio_path"]

            features: dict[str, float] = {"window_id": float(window_id), "timestamp": timestamp}

            if audio_path:
                audio = np.load(audio_path)
                features.update(extract_audio_features(audio, args.sample_rate))

                if args.use_asr:
                    transcript = run_asr(audio, args.sample_rate)
                    features.update(_keyword_features(transcript, "asr"))

            if frame_path:
                image = Image.open(frame_path).convert("RGB")
                if args.use_clip:
                    clip_vec = extract_clip_features(image, clip_model, clip_preprocess, args.device)
                    for idx, value in enumerate(clip_vec):
                        features[f"clip_{idx:03d}"] = float(value)

                if args.use_ocr:
                    ocr_text = run_ocr(image)
                    features.update(_keyword_features(ocr_text, "ocr"))

            if writer is None:
                fieldnames = list(features.keys())
                writer = csv.DictWriter(features_file, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(features)

    print(f"Features written to {features_path}")


if __name__ == "__main__":
    main()
