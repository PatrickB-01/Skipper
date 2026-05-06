"""Hotkey labeling tool for captured sessions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pynput import keyboard


LABEL_MAP = {
    "1": "intro",
    "2": "recap",
    "3": "outro",
    "4": "idle",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label skip segments using hotkeys.")
    parser.add_argument("--session-dir", type=str, required=True, help="Session folder.")
    parser.add_argument("--label-file", type=str, default=None, help="Output labels CSV file.")
    parser.add_argument("--toggle-key", type=str, default="F9", help="Toggle segment start/end.")
    parser.add_argument("--quit-key", type=str, default="F10", help="Quit labeling.")
    return parser.parse_args()


def _key_to_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char
    if isinstance(key, keyboard.Key):
        return key.name
    return None


def main() -> None:
    args = _parse_args()
    session_dir = Path(args.session_dir)
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise FileNotFoundError("session.json not found in session folder.")

    meta = json.loads(meta_path.read_text())
    session_start = meta.get("start_time", time.time())

    label_path = Path(args.label_file) if args.label_file else session_dir / "labels.csv"
    label_path.parent.mkdir(parents=True, exist_ok=True)

    current_label = "intro"
    segment_start = None

    print("Hotkeys:")
    print("  1-4: set label (1=intro, 2=recap, 3=outro, 4=idle)")
    print(f"  {args.toggle_key}: toggle segment start/end")
    print(f"  {args.quit_key}: quit")
    print("Tip: you can press the label key after starting a segment.")

    with label_path.open("w", newline="") as label_file:
        writer = csv.DictWriter(label_file, fieldnames=["start", "end", "label"])
        writer.writeheader()

        def on_press(key):
            nonlocal current_label, segment_start
            name = _key_to_name(key)
            if not name:
                return

            name_upper = name.upper() if isinstance(name, str) else name
            if name_upper == args.quit_key.upper():
                return False
            if name_upper == args.toggle_key.upper():
                now = time.time() - session_start
                if segment_start is None:
                    segment_start = now
                    print(f"Segment started ({current_label}) at {segment_start:.2f}s")
                else:
                    segment_end = now
                    writer.writerow(
                        {"start": f"{segment_start:.3f}", "end": f"{segment_end:.3f}", "label": current_label}
                    )
                    label_file.flush()
                    print(
                        f"Segment ended ({current_label}) at {segment_end:.2f}s"
                    )
                    segment_start = None
                return

            if isinstance(name, str) and name in LABEL_MAP:
                current_label = LABEL_MAP[name]
                if segment_start is None:
                    print(f"Label set to {current_label}")
                else:
                    print(f"Active segment label set to {current_label}")

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


if __name__ == "__main__":
    main()
