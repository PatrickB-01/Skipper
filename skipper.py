"""Press the Right Arrow key repeatedly at a configurable interval.

Usage:
	python skipper.py --seconds 5
"""

from __future__ import annotations

import argparse
import ctypes
import time
import winsound


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_RIGHT = 0x27


# Common virtual-key names for --toggle-key.
KEY_NAME_TO_VK = {
	"TAB": 0x09,
	"ENTER": 0x0D,
	"SHIFT": 0x10,
	"CTRL": 0x11,
	"ALT": 0x12,
	"PAUSE": 0x13,
	"CAPSLOCK": 0x14,
	"ESC": 0x1B,
	"SPACE": 0x20,
	"LEFT": 0x25,
	"UP": 0x26,
	"RIGHT": 0x27,
	"DOWN": 0x28,
	"INSERT": 0x2D,
	"DELETE": 0x2E,
	"HOME": 0x24,
	"END": 0x23,
	"PAGEUP": 0x21,
	"PAGEDOWN": 0x22,
	"BACKSPACE": 0x08,
}

# Add A-Z and 0-9.
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
	KEY_NAME_TO_VK[_ch] = ord(_ch)
for _ch in "0123456789":
	KEY_NAME_TO_VK[_ch] = ord(_ch)

# Add F1-F24.
for _i in range(1, 25):
	KEY_NAME_TO_VK[f"F{_i}"] = 0x6F + _i


class KEYBDINPUT(ctypes.Structure):
	_fields_ = [
		("wVk", ctypes.c_ushort),
		("wScan", ctypes.c_ushort),
		("dwFlags", ctypes.c_ulong),
		("time", ctypes.c_ulong),
		("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
	]


class INPUTUNION(ctypes.Union):
	_fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
	_fields_ = [("type", ctypes.c_ulong), ("union", INPUTUNION)]


def _send_right_arrow() -> None:
	"""Send a Right Arrow key down + up event using Win32 SendInput."""
	extra = ctypes.c_ulong(0)

	key_down = INPUT(
		type=INPUT_KEYBOARD,
		union=INPUTUNION(
			ki=KEYBDINPUT(
				wVk=VK_RIGHT,
				wScan=0,
				dwFlags=0,
				time=0,
				dwExtraInfo=ctypes.pointer(extra),
			)
		),
	)

	key_up = INPUT(
		type=INPUT_KEYBOARD,
		union=INPUTUNION(
			ki=KEYBDINPUT(
				wVk=VK_RIGHT,
				wScan=0,
				dwFlags=KEYEVENTF_KEYUP,
				time=0,
				dwExtraInfo=ctypes.pointer(extra),
			)
		),
	)

	inputs = (INPUT * 2)(key_down, key_up)
	sent = ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
	if sent != 2:
		raise OSError("SendInput failed to send Right Arrow key events.")


def _parse_toggle_vk(key_text: str) -> int:
	"""Parse a key name like F8/A/SPACE or a hex VK like 0x77."""
	value = key_text.strip().upper()
	if not value:
		raise ValueError("--toggle-key cannot be empty.")

	if value.startswith("0X"):
		vk = int(value, 16)
		if not 0 <= vk <= 0xFF:
			raise ValueError("Hex virtual-key code must be between 0x00 and 0xFF.")
		return vk

	try:
		return KEY_NAME_TO_VK[value]
	except KeyError as exc:
		raise ValueError(
			f"Unsupported --toggle-key '{key_text}'. Use names like F8, A, SPACE, ESC, or hex like 0x77."
		) from exc


def _is_key_down(vk_code: int) -> bool:
	"""Return True while the key is physically pressed."""
	return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def _play_toggle_sound(paused: bool) -> None:
	"""Play a short sound when toggling pause/resume."""
	if paused:
		winsound.Beep(500, 140)
		winsound.Beep(380, 140)
	else:
		winsound.Beep(650, 140)
		winsound.Beep(800, 140)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Press the Right Arrow key every few seconds until stopped."
	)
	parser.add_argument(
		"--seconds",
		type=float,
		default=5.0,
		help="Interval between key presses in seconds (default: 5).",
	)
	parser.add_argument(
		"--toggle-key",
		type=str,
		default="F8",
		help="Key used to toggle pause/resume (default: F8).",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	if args.seconds <= 0:
		raise ValueError("--seconds must be greater than 0.")
	if args.toggle_key.upper() == "RIGHT":
		raise ValueError("--toggle-key cannot be RIGHT because RIGHT is used for skipping.")

	toggle_vk = _parse_toggle_vk(args.toggle_key)
	paused = False
	was_toggle_down = False
	next_skip_time = time.monotonic()

	print(
		f"Skipping is ACTIVE: pressing Right Arrow every {args.seconds:g} second(s). "
		f"Press {args.toggle_key.upper()} to toggle. Ctrl+C to stop."
	)
	while True:
		now = time.monotonic()
		toggle_down = _is_key_down(toggle_vk)

		# Trigger toggle only on key down edge to avoid repeated flips while held.
		if toggle_down and not was_toggle_down:
			paused = not paused
			_play_toggle_sound(paused)
			if paused:
				print("Toggle ON: skipping PAUSED.")
			else:
				print("Toggle OFF: skipping ACTIVE.")
				next_skip_time = now

		was_toggle_down = toggle_down

		if not paused and now >= next_skip_time:
			_send_right_arrow()
			next_skip_time = now + args.seconds

		# Small sleep keeps CPU usage low while still feeling responsive.
		time.sleep(0.02)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nStopped.")
