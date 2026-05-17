# StreamSkipper

Skip boring segments by learning from your screen/audio in real time.

## Overview

This project captures screen frames and system audio, extracts multimodal features (audio, CLIP frame embeddings, ASR text, OCR text), labels segments with hotkeys, trains a RandomForest model, and runs real-time inference to trigger skipping.

## Quick Start (Windows)

1) Create a capture session

```bash
python capture.py --session-dir data/session_001 --window-seconds 5 --hop-seconds 1 --audio-backend soundcard --jpeg-quality 80
```

2) Label segments while the session is running

```bash
python labeler.py --session-dir data/session_001
```

3) Extract features

```bash
python features.py --session-dir data/session_001 --use-clip --use-asr --use-ocr
```

GPU (if available) for CLIP + ASR:

```bash
python features.py --session-dir data/session_001 --use-clip --use-asr --use-ocr --device cuda --asr-device auto --asr-compute-type auto
```

4) Train the model

```bash
python train.py --features data/session_001/features.csv --labels data/session_001/labels.csv --model-out model.joblib --feature-out features.json
```

Train across multiple sessions (same feature pipeline):

```bash
python train.py --features data/session_001/features.csv data/session_002/features.csv --labels data/session_001/labels.csv data/session_002/labels.csv --model-out model.joblib --feature-out features.json
```

5) Run real-time inference (dry run first)

```bash
python infer.py --model model.joblib --features features.json --use-clip --use-asr --use-ocr --audio-backend soundcard --dry-run
```

GPU (if available) for CLIP + ASR:

```bash
python infer.py --model model.joblib --features features.json --use-clip --use-asr --use-ocr --audio-backend soundcard --torch-device cuda --asr-device auto --asr-compute-type auto
```

```bash
python infer.py --model model.joblib --features features.json --use-clip --use-asr --use-ocr --audio-backend soundcard --asr-device cpu --asr-compute-type int8 --dry-run --asr-echo
```

Remove `--dry-run` to enable actual skipping.

## Labeling Hotkeys

- `Q` = toggle intro segment on/off
- `W` = toggle recap segment on/off
- `E` = toggle outro segment on/off
- `R` = toggle idle segment on/off (optional)
- `P` = quit

## Notes and Options

- `capture.py` args: `--audio-backend` (soundcard/sounddevice), `--loopback` (WASAPI), `--device` (audio device name/index), `--jpeg-quality` (frame size), `--no-audio`/`--no-video` (disable streams).
- `features.py` args: `--use-clip`/`--use-asr`/`--use-ocr` (feature toggles), `--device` (CLIP torch device), `--asr-device` + `--asr-compute-type` (ASR runtime), `--model-cache-dir` (model cache).
- `infer.py` args: `--threshold` (skip probability), `--debounce` (consecutive hits), `--window-seconds`/`--hop-seconds` (analysis cadence), `--dry-run` (no keypress), `--asr-mode` (subprocess/inprocess), `--asr-timeout` (skip slow ASR), `--asr-echo` (print transcripts), `--ignore-skip-errors` (log keypress failures), `--model-cache-dir` (model cache).
- Use `--audio-backend soundcard` for system audio capture on Windows. Use `--audio-backend sounddevice` with `--loopback` if preferred.
- For audio device selection, pass `--device "Headphones"` (partial match).
- If ASR or OCR are not installed, skip `--use-asr` or `--use-ocr`.
- RandomForest models are retrained from scratch when you add new sessions.
- If you want to avoid CLIP embeddings, omit `--use-clip` (features will be audio + text only).
- For talk-heavy or idle segments, label them consistently to avoid confusing the model.

## Listen to Captured Audio

Play a saved `.npy` audio window:

```bash
python play_audio.py --path data/session_001/audio/audio_000005.npy --sample-rate 16000
```

Select an output device by name (or index):

```bash
python play_audio.py --path data/session_001/audio/audio_000005.npy --sample-rate 16000 --device "Headphones"
```

## Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

Extra requirements:
- Tesseract OCR installed and on PATH for OCR.
- No extra tools required for `faster-whisper` when using the in-memory audio pipeline.

## Data Layout

Each session folder contains:
- `session.json`: capture settings + start time
- `manifest.csv`: window metadata and file paths
- `frames/`: JPEG frames
- `audio/`: NPY audio windows
- `labels.csv`: labeled segments
- `features.csv`: extracted feature rows
