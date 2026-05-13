"""Hotkey labeling tool for captured sessions."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

from pynput import keyboard
import tkinter as tk


LABEL_MAP = {
    "Q": "intro",
    "W": "recap",
    "E": "outro",
    "R": "idle",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label skip segments using hotkeys.")
    parser.add_argument("--session-dir", type=str, required=True, help="Session folder.")
    parser.add_argument("--label-file", type=str, default=None, help="Output labels CSV file.")
    parser.add_argument("--quit-key", type=str, default="P", help="Quit labeling.")
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
    state_lock = threading.Lock()
    current_display = "NONE"

    print("Hotkeys:")
    print("  Q-W-E-R: toggle segment (Q=intro, W=recap, E=outro, R=idle)")
    print(f"  {args.quit_key}: quit")

    with label_path.open("w", newline="") as label_file:
        writer = csv.DictWriter(label_file, fieldnames=["start", "end", "label"])
        writer.writeheader()

        root = tk.Tk()
        root.title("StreamSkipper")
        root.attributes("-topmost", True)
        root.overrideredirect(True)
        root.geometry("+10+10")
        root.configure(bg="black")
        root.wm_attributes("-transparentcolor", "black")

        status_var = tk.StringVar(value="NONE")
        label = tk.Label(
            root,
            textvariable=status_var,
            fg="white",
            bg="black",
            font=("Segoe UI", 8, "bold"),
            padx=4,
            pady=2,
        )
        label.pack()

        def _refresh_indicator() -> None:
            with state_lock:
                display = current_display
            status_var.set(f"{display}")
            root.after(100, _refresh_indicator)

        def on_press(key):
            nonlocal current_label, segment_start, current_display
            name = _key_to_name(key)
            if not name:
                return

            name_upper = name.upper() if isinstance(name, str) else name
            if name_upper == args.quit_key.upper():
                root.after(0, root.destroy)
                return False
            if isinstance(name, str) and name_upper in LABEL_MAP:
                selected_label = LABEL_MAP[name_upper]
                now = time.time() - session_start
                if segment_start is None:
                    current_label = selected_label
                    segment_start = now
                    with state_lock:
                        current_display = current_label
                    print(f"Segment started ({current_label}) at {segment_start:.2f}s")
                elif selected_label == current_label:
                    segment_end = now
                    writer.writerow(
                        {"start": f"{segment_start:.3f}", "end": f"{segment_end:.3f}", "label": current_label}
                    )
                    label_file.flush()
                    print(f"Segment ended ({current_label}) at {segment_end:.2f}s")
                    segment_start = None
                    with state_lock:
                        current_display = "NONE"
                else:
                    segment_end = now
                    writer.writerow(
                        {"start": f"{segment_start:.3f}", "end": f"{segment_end:.3f}", "label": current_label}
                    )
                    label_file.flush()
                    print(f"Segment ended ({current_label}) at {segment_end:.2f}s")
                    current_label = selected_label
                    segment_start = now
                    with state_lock:
                        current_display = current_label
                    print(f"Segment started ({current_label}) at {segment_start:.2f}s")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        _refresh_indicator()
        root.mainloop()
        listener.stop()


if __name__ == "__main__":
    main()
