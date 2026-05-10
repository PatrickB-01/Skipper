# StreamSkipper

Skip boring segments by learning from your screen/audio in real time.

## Overview

This project captures screen frames and system audio, extracts multimodal features (audio, CLIP frame embeddings, ASR text, OCR text), labels segments with hotkeys, trains a RandomForest model, and runs real-time inference to trigger skipping.

## Quick Start (Windows)

1) Create a capture session

```bash
python capture.py --session-dir data/session_001 --window-seconds 5 --hop-seconds 1 --audio-backend soundcard
```

2) Label segments while the session is running

```bash
python labeler.py --session-dir data/session_001
```

3) Extract features

```bash
python features.py --session-dir data/session_001 --use-clip --use-asr --use-ocr
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

Remove `--dry-run` to enable actual skipping.

## Labeling Hotkeys

- `Q` = intro
- `W` = recap
- `E` = outro
- `R` = idle (optional)
- `O` = toggle segment start/end
- `P` = quit

## Notes and Options

- Use `--audio-backend soundcard` for system audio capture on Windows.
- If you prefer `sounddevice`, use `--audio-backend sounddevice` with `--loopback`.
- If ASR or OCR are not installed, skip `--use-asr` or `--use-ocr`.
- You can pass `--device "Headphones"` (partial match) to pick a specific output.
- RandomForest models are retrained from scratch when you add new sessions.
- If you want to avoid CLIP embeddings, omit `--use-clip` (features will be audio + text only).
- For talk-heavy or idle segments, label them consistently to avoid confusing the model.

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
