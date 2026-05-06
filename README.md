# StreamSkipper

Skip boring segments by learning from your screen/audio in real time.

## Overview

This project captures screen frames and system audio, extracts multimodal features (audio, CLIP frame embeddings, ASR text, OCR text), labels segments with hotkeys, trains a RandomForest model, and runs real-time inference to trigger skipping.

## Quick Start (Windows)

1) Create a capture session

```bash
python capture.py --session-dir data/session_001 --window-seconds 5 --hop-seconds 1 --loopback
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

5) Run real-time inference (dry run first)

```bash
python infer.py --model model.joblib --features features.json --use-clip --use-asr --use-ocr --loopback --dry-run
```

Remove `--dry-run` to enable actual skipping.

## Labeling Hotkeys

- `1` = intro
- `2` = recap
- `3` = outro
- `4` = idle (optional)
- `F9` = toggle segment start/end
- `F10` = quit

## Notes and Options

- Use `--loopback` to capture system audio via WASAPI on Windows.
- If `Whisper` or `Tesseract` are not installed, skip `--use-asr` or `--use-ocr`.
- If you want to avoid CLIP embeddings, omit `--use-clip` (features will be audio + text only).
- For talk-heavy or idle segments, label them consistently to avoid confusing the model.

## Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

Extra requirements:
- Tesseract OCR installed and on PATH for OCR.
- FFmpeg installed and on PATH for Whisper.

## Data Layout

Each session folder contains:
- `session.json`: capture settings + start time
- `manifest.csv`: window metadata and file paths
- `frames/`: JPEG frames
- `audio/`: NPY audio windows
- `labels.csv`: labeled segments
- `features.csv`: extracted feature rows
