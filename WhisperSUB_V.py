import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = (APP_DIR / "models" / "ggml-large-v3.bin") if (APP_DIR / "models" / "ggml-large-v3.bin").exists() else (APP_DIR / "model" / "ggml-large-v3.bin")
DEFAULT_THREADS = max(4, (os.cpu_count() or 8) // 2)
DEFAULT_MAX_SUBTITLE_LEN = 55
SAVE_DIR = APP_DIR / "save_files"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
APP_VERSION = "1.0.0"
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOGGER = logging.getLogger("WhisperSUB_V")



def subprocess_no_window_kwargs() -> dict:
    if os.name != "nt":
        return {}

    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startup,
    }
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    LOGGER.info("=== WhisperSUB_V start v%s ===", APP_VERSION)

LANG_OPTIONS = ["auto", "en", "vi", "ja", "ko", "zh", "fr", "de", "es"]
DEFAULT_DOWNLOAD_URLS = {
    "model": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
    "whisper_cli": "https://github.com/ggml-org/whisper.cpp/releases",
    "ffmpeg": "https://www.ffmpeg.org/download.html",
    "mpv": "https://github.com/mpv-player/mpv/releases",
    "vad": "https://huggingface.co/ggml-org/whisper-vad/resolve/main/silero_v5.1.2.bin?download=true",
}
ADVANCED_PRESETS = {
    "auto": {
        "temperature": 0.2,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.6,
        "entropy_thold": 2.4,
        "logprob_thold": -1.0,
        "use_vad": True,
        "vad_threshold": 0.5,
        "source_lang": "auto",
    },
    "en (English)": {
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.7,
        "entropy_thold": 2.4,
        "logprob_thold": -1.0,
        "use_vad": True,
        "vad_threshold": 0.5,
        "source_lang": "en",
    },
    "vi (Vietnamese)": {
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.4,
        "entropy_thold": 3.0,
        "logprob_thold": -1.0,
        "use_vad": False,
        "vad_threshold": 0.5,
        "source_lang": "vi",
    },
    "ja (Japanese)": {
        "temperature": 0.0,
        "beam_size": 6,
        "best_of": 5,
        "no_speech_thold": 0.4,
        "entropy_thold": 3.0,
        "logprob_thold": -1.0,
        "use_vad": False,
        "vad_threshold": 0.5,
        "source_lang": "ja",
    },
    "ko (Korean)": {
        "temperature": 0.0,
        "beam_size": 6,
        "best_of": 5,
        "no_speech_thold": 0.4,
        "entropy_thold": 3.0,
        "logprob_thold": -1.0,
        "use_vad": False,
        "vad_threshold": 0.5,
        "source_lang": "ko",
    },
    "zh (Chinese)": {
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.3,
        "entropy_thold": 3.0,
        "logprob_thold": -1.0,
        "use_vad": False,
        "vad_threshold": 0.5,
        "source_lang": "zh",
    },
    "fr (French)": {
        "temperature": 0.1,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.5,
        "entropy_thold": 2.4,
        "logprob_thold": -1.0,
        "use_vad": True,
        "vad_threshold": 0.5,
        "source_lang": "fr",
    },
    "de (German)": {
        "temperature": 0.1,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.6,
        "entropy_thold": 2.4,
        "logprob_thold": -1.0,
        "use_vad": True,
        "vad_threshold": 0.5,
        "source_lang": "de",
    },
    "es (Spanish)": {
        "temperature": 0.1,
        "beam_size": 5,
        "best_of": 5,
        "no_speech_thold": 0.5,
        "entropy_thold": 2.4,
        "logprob_thold": -1.0,
        "use_vad": True,
        "vad_threshold": 0.5,
        "source_lang": "es",
    },
}

@dataclass
class SubtitleEntry:
    start_ms: int
    end_ms: int
    text: str


def guess_whisper_cli() -> Path | None:
    candidates = [
        APP_DIR / "Release" / "whisper-cli.exe",
        APP_DIR / "Release" / "main.exe",
        APP_DIR / "whisper-cli.exe",
        APP_DIR / "main.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def to_windows_short_path(path: Path) -> str:
    value = str(path)
    if not path.exists() or not hasattr(ctypes, "windll"):
        return value

    get_short_path = ctypes.windll.kernel32.GetShortPathNameW
    needed = get_short_path(value, None, 0)
    if needed == 0:
        return value

    buffer = ctypes.create_unicode_buffer(needed)
    result = get_short_path(value, buffer, needed)
    return buffer.value if result != 0 else value


def find_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    local = APP_DIR / "Release" / "ffmpeg.exe"
    return str(local) if local.exists() else None


def find_mpv() -> str | None:
    system = shutil.which("mpv")
    if system:
        return system
    local = APP_DIR / "Release" / "mpv.exe"
    return str(local) if local.exists() else None



def required_components() -> list[dict]:
    return [
        {
            "key": "model",
            "label": "Model ggml-large-v3.bin",
            "path": APP_DIR / "models" / "ggml-large-v3.bin",
            "aliases": ["ggml-large-v3.bin"],
        },
        {
            "key": "whisper_cli",
            "label": "whisper-cli.exe",
            "path": APP_DIR / "Release" / "whisper-cli.exe",
            "aliases": ["whisper-cli.exe", "main.exe"],
        },
        {
            "key": "ffmpeg",
            "label": "ffmpeg.exe",
            "path": APP_DIR / "Release" / "ffmpeg.exe",
            "aliases": ["ffmpeg.exe"],
        },
        {
            "key": "mpv",
            "label": "mpv.exe",
            "path": APP_DIR / "Release" / "mpv.exe",
            "aliases": ["mpv.exe"],
        },
        {
            "key": "vad",
            "label": "VAD model (silero)",
            "path": APP_DIR / "Release" / "silero_vad.bin",
            "aliases": ["silero_vad.bin", "silero_v5.1.2.bin", "silero_vad.onnx", "silero-vad.onnx"],
        },
    ]


def normalize_download_url(url: str) -> str:
    u = url.strip()
    if not u:
        return u
    u = u.replace("/blob/", "/resolve/") if "huggingface.co/" in u and "/blob/" in u else u
    u = u.replace("/blob/", "/raw/") if "github.com/" in u and "/blob/" in u else u
    return u


def _download_to_file(url: str, out_file: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "WhisperSUB_V/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp, out_file.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _find_first_matching_file(root_dir: Path, candidate_names: list[str]) -> Path | None:
    names = {n.lower() for n in candidate_names}
    for p in root_dir.rglob("*"):
        if p.is_file() and p.name.lower() in names:
            return p
    return None


def _save_component_from_url(url: str, target: Path, aliases: list[str]) -> None:
    fixed_url = normalize_download_url(url)
    if not fixed_url:
        raise RuntimeError("URL trong")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="whispersubv_dl_"))
    try:
        parsed = urllib.parse.urlparse(fixed_url)
        guessed = Path(parsed.path).name or target.name
        tmp_file = tmp_dir / guessed
        _download_to_file(fixed_url, tmp_file)

        if zipfile.is_zipfile(tmp_file):
            extract_dir = tmp_dir / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_file, "r") as zf:
                zf.extractall(extract_dir)
            match = _find_first_matching_file(extract_dir, aliases)
            if not match:
                raise RuntimeError("Zip khong chua file can thiet")
            shutil.copyfile(match, target)
            return

        shutil.copyfile(tmp_file, target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def find_vad_model_path() -> Path | None:
    candidates = [
        APP_DIR / "Release" / "silero_vad.bin",
        APP_DIR / "Release" / "silero_v5.1.2.bin",
        APP_DIR / "Release" / "silero_vad.onnx",
        APP_DIR / "silero_vad.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
def prepare_audio_for_whisper(input_path: Path) -> tuple[Path, Path | None, float]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Khong tim thay ffmpeg. Can ffmpeg de tach audio tu video.")

    temp_dir = Path(tempfile.mkdtemp(prefix="dichthuat_"))
    # Luon dung ten ASCII cho file tam de tranh loi unicode/ky tu dac biet
    # voi mot so ban build whisper-cli tren Windows.
    wav_path = temp_dir / "audio_input.wav"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **subprocess_no_window_kwargs())
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0 or not wav_path.exists():
        raise RuntimeError("Khong the trich audio tam bang ffmpeg.")

    return wav_path, temp_dir, elapsed


def profile_params(profile_name: str) -> tuple[int, int, int]:
    if profile_name == "Nhanh":
        return 5, 5, max(4, DEFAULT_THREADS)
    if profile_name == "Chinh xac":
        return 5, 5, max(2, DEFAULT_THREADS - 2)
    return 5, 5, max(4, DEFAULT_THREADS)


def is_vad_model_error(stderr_text: str) -> bool:
    msg = stderr_text.lower()
    return (
        "invalid model data (bad magic)" in msg
        or "failed to initialize vad context" in msg
        or "failed to compute vad" in msg
    )


def parse_srt_timecode(value: str) -> int:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def parse_time_input_to_ms(value: str) -> int:
    v = value.strip()
    if not v:
        return 0
    if ":" in v and "," in v:
        return parse_srt_timecode(v)
    return int(float(v) * 1000)


def format_srt_timecode(ms: int) -> str:
    ms = max(0, ms)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt(path: Path) -> list[SubtitleEntry]:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    entries: list[SubtitleEntry] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        left, right = [part.strip() for part in lines[1].split("-->", 1)]
        try:
            start_ms = parse_srt_timecode(left)
            end_ms = parse_srt_timecode(right)
        except Exception:
            continue
        text = " ".join(lines[2:]).strip()
        entries.append(SubtitleEntry(start_ms=start_ms, end_ms=end_ms, text=text))

    return entries


def write_srt(path: Path, entries: list[SubtitleEntry]) -> None:
    chunks: list[str] = []
    for i, e in enumerate(entries, start=1):
        chunks.append(str(i))
        chunks.append(f"{format_srt_timecode(e.start_ms)} --> {format_srt_timecode(e.end_ms)}")
        chunks.append(e.text)
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


def entries_to_json(entries: list[SubtitleEntry]) -> list[dict]:
    return [
        {"start_ms": int(e.start_ms), "end_ms": int(e.end_ms), "text": e.text}
        for e in entries
    ]


def entries_from_json(items: list[dict]) -> list[SubtitleEntry]:
    result: list[SubtitleEntry] = []
    for it in items:
        try:
            result.append(
                SubtitleEntry(
                    start_ms=int(it.get("start_ms", 0)),
                    end_ms=int(it.get("end_ms", 0)),
                    text=str(it.get("text", "")).strip(),
                )
            )
        except Exception:
            continue
    return result


def save_progress_json(progress_path: Path, media_path: Path, srt_path: Path, meta: dict | None = None) -> None:
    payload = {
        "version": 2,
        "media_path": str(media_path),
        "srt_path": str(srt_path),
        "meta": meta or {},
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_progress_json(progress_path: Path) -> tuple[Path, Path, dict, list[SubtitleEntry]]:
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    media = Path(payload.get("media_path", ""))
    srt = Path(payload.get("srt_path", ""))
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}

    # Backward compatibility: old progress files may still carry embedded entries.
    legacy_entries = entries_from_json(payload.get("entries", [])) if isinstance(payload.get("entries", []), list) else []
    return media, srt, meta, legacy_entries


def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def collapse_repeated_chunks(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return text

    tokens = t.split(" ")
    collapsed: list[str] = []
    for token in tokens:
        if len(collapsed) >= 2 and collapsed[-1] == token and collapsed[-2] == token:
            continue
        collapsed.append(token)

    candidate = " ".join(collapsed)

    for n in range(8, 1, -1):
        parts = candidate.split(" ")
        if len(parts) < n * 2:
            continue
        a = parts[-n:]
        b = parts[-2 * n : -n]
        if a == b:
            candidate = " ".join(parts[:-n])

    return candidate.strip()


def postprocess_srt(entries: list[SubtitleEntry], clean_repetitions: bool) -> tuple[list[SubtitleEntry], int]:
    cleaned: list[SubtitleEntry] = []
    changed = 0

    for e in entries:
        txt = e.text
        if clean_repetitions:
            txt = collapse_repeated_chunks(txt)
            if txt != normalize_text(e.text):
                changed += 1

        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue

        cleaned.append(SubtitleEntry(start_ms=e.start_ms, end_ms=e.end_ms, text=txt))

    return cleaned, changed


def detect_speech_bounds(media_path: Path, start_ms: int, end_ms: int) -> tuple[int, int] | None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None

    clip_start = max(0.0, start_ms / 1000.0 - 1.0)
    clip_end = end_ms / 1000.0 + 1.0
    clip_dur = max(0.1, clip_end - clip_start)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{clip_start:.3f}",
        "-to",
        f"{clip_end:.3f}",
        "-i",
        str(media_path),
        "-af",
        "silencedetect=noise=-32dB:d=0.10",
        "-f",
        "null",
        "NUL",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **subprocess_no_window_kwargs())
    text = proc.stderr

    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", text)]

    silence_ranges: list[tuple[float, float]] = []
    for i in range(min(len(starts), len(ends))):
        if ends[i] > starts[i]:
            silence_ranges.append((starts[i], ends[i]))

    speech_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for s0, s1 in silence_ranges:
        if s0 > cursor:
            speech_ranges.append((cursor, s0))
        cursor = max(cursor, s1)
    if cursor < clip_dur:
        speech_ranges.append((cursor, clip_dur))

    if not speech_ranges:
        return None

    target_start = start_ms / 1000.0 - clip_start
    target_end = end_ms / 1000.0 - clip_start

    best = None
    best_score = -1.0
    for a, b in speech_ranges:
        overlap = max(0.0, min(b, target_end) - max(a, target_start))
        dur = max(0.01, b - a)
        score = overlap + min(dur, 1.0) * 0.1
        if score > best_score:
            best_score = score
            best = (a, b)

    if not best:
        return None

    new_start = int((clip_start + best[0]) * 1000)
    new_end = int((clip_start + best[1]) * 1000)
    if new_end <= new_start:
        new_end = new_start + 100
    return new_start, new_end


class SubtitleEditorWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, media_path: Path, srt_path: Path):
        super().__init__(master)
        self.title(f"SRT Editor - {srt_path.name}")
        self.geometry("1200x760")

        self.media_path = media_path
        self.srt_path = srt_path
        self.entries = parse_srt(srt_path)

        self.selected_index: int | None = None

        self.ffplay_proc = None
        self.ffplay_hwnd = 0
        self.ffplay_title = f"WhisperSUBV_Editor_{os.getpid()}_{id(self)}"
        self.mpv_pipe_path = ""
        self.mpv_pipe = None
        self.mpv_lock = threading.RLock()

        self.video_playing = False
        self.video_len_sec = self._probe_duration()
        self.current_sec = 0.0
        self.play_start_wall = 0.0
        self.play_start_sec = 0.0
        self.play_tick_after = None
        self.ffplay_started_wall = 0.0
        self.last_tick_wall = 0.0
        self._updating_seek = False

        self.overlay_dir = Path(tempfile.mkdtemp(prefix="whispersubv_overlay_"))
        self.overlay_srt = self.overlay_dir / "overlay.srt"
        write_srt(self.overlay_srt, self.entries)

        self._build_ui()
        self._load_tree()
        self._update_time_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)

        self.video_host = tk.Frame(left, bg="black", width=640, height=360)
        self.video_host.pack(fill="both", expand=True)
        self.video_host.pack_propagate(False)
        self.video_host.bind("<Configure>", self._on_video_host_resize)

        controls = ttk.Frame(left)
        controls.pack(fill="x", pady=(6, 0))

        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(controls, text="-1s", command=lambda: self.seek_relative(-1.0)).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="+1s", command=lambda: self.seek_relative(1.0)).pack(side="left", padx=(6, 0))

        self.time_var = tk.StringVar(value="00:00.000")
        ttk.Label(controls, textvariable=self.time_var).pack(side="left", padx=(10, 0))

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(left, from_=0.0, to=max(1.0, self.video_len_sec), orient="horizontal", variable=self.seek_var, command=self.on_seek)
        self.seek_scale.pack(fill="x", pady=(4, 0))

        right = ttk.Frame(root, width=540)
        right.pack(side="left", fill="both", expand=False, padx=(10, 0))

        cols = ("idx", "start", "end", "text")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=20)
        self.tree.heading("idx", text="#")
        self.tree.heading("start", text="Start")
        self.tree.heading("end", text="End")
        self.tree.heading("text", text="Text")
        self.tree.column("idx", width=40, anchor="center")
        self.tree.column("start", width=95)
        self.tree.column("end", width=95)
        self.tree.column("text", width=300)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click_force)
        self.tree.bind("<Delete>", self.delete_selected_sub)

        edit = ttk.LabelFrame(right, text="Edit subtitle", padding=8)
        edit.pack(fill="x", pady=(8, 0))

        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.text_var = tk.StringVar()

        ttk.Label(edit, text="Start").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.start_var, width=18).grid(row=0, column=1, sticky="we", padx=(6, 8))
        ttk.Label(edit, text="End").grid(row=0, column=2, sticky="w")
        ttk.Entry(edit, textvariable=self.end_var, width=18).grid(row=0, column=3, sticky="we", padx=(6, 0))

        ttk.Label(edit, text="Text").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(edit, textvariable=self.text_var).grid(row=1, column=1, columnspan=3, sticky="we", padx=(6, 0), pady=(6, 0))

        row2 = ttk.Frame(edit)
        row2.grid(row=2, column=0, columnspan=4, sticky="we", pady=(8, 0))
        ttk.Button(row2, text="Apply", command=self.apply_edit).pack(side="left")
        ttk.Button(row2, text="Add Below", command=self.add_sub_below).pack(side="left", padx=(6, 0))
        ttk.Button(row2, text="-100ms", command=lambda: self.shift_selected(-100)).pack(side="left", padx=(6, 0))
        ttk.Button(row2, text="+100ms", command=lambda: self.shift_selected(100)).pack(side="left", padx=(6, 0))
        ttk.Button(row2, text="Set Start=Now", command=self.set_start_to_current).pack(side="left", padx=(6, 0))
        ttk.Button(row2, text="Set End=Now", command=self.set_end_to_current).pack(side="left", padx=(6, 0))
        ttk.Button(row2, text="Do gi?ng noi", command=self.snap_selected_to_speech).pack(side="left", padx=(6, 0))

        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Save SRT", command=self.save_srt).pack(side="left")
        ttk.Button(bottom, text="Save Progress", command=self.save_progress).pack(side="left", padx=(6, 0))
        ttk.Button(bottom, text="Reload", command=self.reload_srt).pack(side="left", padx=(6, 0))

        edit.columnconfigure(1, weight=1)
        edit.columnconfigure(3, weight=1)

    def _probe_duration(self) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0.0
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(self.media_path),
        ]
        try:
            out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace").strip()
            return max(0.0, float(out))
        except Exception:
            return 0.0
    def _is_ffplay_alive(self) -> bool:
        return bool(self.ffplay_proc and self.ffplay_proc.poll() is None)

    def _new_mpv_pipe_path(self) -> str:
        return rf"\\.\pipe\whispersubv_mpv_{os.getpid()}_{id(self)}_{int(time.time() * 1000)}"

    def _connect_mpv_ipc(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.perf_counter() + max(0.2, timeout_sec)
        while time.perf_counter() < deadline:
            try:
                self.mpv_pipe = open(self.mpv_pipe_path, "r+b", buffering=0)
                return True
            except Exception:
                time.sleep(0.05)
        self.mpv_pipe = None
        return False

    def _mpv_send(self, payload: dict) -> bool:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self.mpv_lock:
            if not self.mpv_pipe:
                return False
            try:
                self.mpv_pipe.write(data)
                return True
            except Exception:
                return False

    def _mpv_command(self, *args) -> bool:
        return self._mpv_send({"command": list(args)})

    def _on_video_host_resize(self, _evt=None) -> None:
        return

    def _refresh_overlay_srt(self) -> None:
        write_srt(self.overlay_srt, self.entries)
        if self._is_ffplay_alive():
            self._mpv_command("sub-reload")

    def _start_ffplay(self, start_sec: float) -> bool:
        # Chi khoi dong mpv mot lan cho moi cua so editor.
        if self._is_ffplay_alive():
            self._mpv_command("seek", round(max(0.0, start_sec), 3), "absolute")
            self.current_sec = max(0.0, start_sec)
            return True

        mpv = find_mpv()
        if not mpv:
            messagebox.showerror("Loi", "Khong tim thay mpv.exe. Hay cai mpv hoac dat vao Release/mpv.exe", parent=self)
            return False

        self._refresh_overlay_srt()

        self.mpv_pipe_path = self._new_mpv_pipe_path()
        wid = int(self.video_host.winfo_id())
        cmd = [
            mpv,
            "--no-config",
            "--quiet",
            "--no-terminal",
            "--force-window=yes",
            "--idle=yes",
            "--keep-open=yes",
            "--pause=yes",
            f"--wid={wid}",
            f"--input-ipc-server={self.mpv_pipe_path}",
            f"--sub-file={self.overlay_srt}",
            "--sub-auto=no",
            "--osd-level=0",
            "--audio-display=no",
            str(self.media_path),
        ]
        try:
            self.ffplay_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **subprocess_no_window_kwargs())
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong mo duoc mpv: {exc}", parent=self)
            self.ffplay_proc = None
            return False

        if not self._connect_mpv_ipc(timeout_sec=5.0):
            self._stop_ffplay()
            messagebox.showerror("Loi", "Khong ket noi duoc mpv IPC.", parent=self)
            return False

        self._mpv_command("seek", round(max(0.0, start_sec), 3), "absolute")
        self.current_sec = max(0.0, start_sec)
        self.ffplay_started_wall = time.perf_counter()
        return True

    def _stop_ffplay(self) -> None:
        if self.mpv_pipe:
            try:
                self.mpv_pipe.close()
            except Exception:
                pass
        self.mpv_pipe = None

        if self.ffplay_proc and self.ffplay_proc.poll() is None:
            self.ffplay_proc.terminate()
            try:
                self.ffplay_proc.wait(timeout=1.0)
            except Exception:
                pass
        self.ffplay_proc = None
        self.ffplay_hwnd = 0
        self.mpv_pipe_path = ""

    def _load_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, e in enumerate(self.entries, start=1):
            self.tree.insert("", "end", iid=str(i - 1), values=(i, format_srt_timecode(e.start_ms), format_srt_timecode(e.end_ms), e.text))

    def _update_time_ui(self) -> None:
        t = max(0.0, self.current_sec)
        self._updating_seek = True
        try:
            self.seek_var.set(t)
        finally:
            self._updating_seek = False
        self.time_var.set(f"{int(t//60):02}:{t%60:06.3f}")

    def _tick_playback(self):
        if not self.video_playing:
            return

        now = time.perf_counter()

        if self.last_tick_wall == 0.0:
            self.last_tick_wall = now

        delta = now - self.last_tick_wall
        self.last_tick_wall = now

        self.current_sec += delta

        if self.video_len_sec > 0:
            self.current_sec = min(self.current_sec, self.video_len_sec)

        self._update_time_ui()
        self.play_tick_after = self.after(33, self._tick_playback)

    def toggle_play(self) -> None:
        if self.video_playing:
            if self._is_ffplay_alive():
                self._mpv_command("set_property", "pause", True)

            self.video_playing = False
            self.play_btn.configure(text="Play")

            if self.play_tick_after:
                self.after_cancel(self.play_tick_after)
                self.play_tick_after = None

            self.last_tick_wall = 0.0
            return

        if self._is_ffplay_alive():
            self._mpv_command("set_property", "pause", False)
            self.video_playing = True
            self.last_tick_wall = time.perf_counter()
            self.play_btn.configure(text="Pause")
            self._tick_playback()
            return

        if self._start_ffplay(self.current_sec):
            self._mpv_command("seek", round(self.current_sec, 3), "absolute")
            self._mpv_command("set_property", "pause", False)
            self.video_playing = True
            self.last_tick_wall = time.perf_counter()
            self.play_btn.configure(text="Pause")
            self._tick_playback()

    def on_seek(self, _evt=None) -> None:
        if self._updating_seek:
            return

        sec = float(self.seek_var.get())
        self.current_sec = max(0.0, sec)

        if self._is_ffplay_alive():
            self._mpv_command("set_property", "pause", True)
            self._mpv_command("seek", round(self.current_sec, 3), "absolute")
            if self.video_playing:
                self._mpv_command("set_property", "pause", False)
        self._update_time_ui()

    def seek_relative(self, delta_sec: float) -> None:
        new_t = self.current_sec + delta_sec
        if self.video_len_sec > 0:
            new_t = min(max(0.0, new_t), self.video_len_sec)
        else:
            new_t = max(0.0, new_t)
        self.current_sec = new_t

        # Tranh trigger on_seek 2 lan (do seek_var callback + goi truc tiep).
        self._updating_seek = True
        try:
            self.seek_var.set(new_t)
        finally:
            self._updating_seek = False

        self.on_seek()

    def current_ms(self) -> int:
        return int(self.current_sec * 1000)

    def _jump_to_entry(self, idx: int, stop_if_playing: bool = True) -> None:
        e = self.entries[idx]
        self.selected_index = idx
        self.start_var.set(format_srt_timecode(e.start_ms))
        self.end_var.set(format_srt_timecode(e.end_ms))
        self.text_var.set(e.text)

        self.current_sec = e.start_ms / 1000.0
        self._update_time_ui()

        if stop_if_playing:
            if self.video_playing:
                self.video_playing = False

            if self.play_tick_after:
                self.after_cancel(self.play_tick_after)
                self.play_tick_after = None

            self.last_tick_wall = 0.0
            self.play_btn.configure(text="Play")

        if self._is_ffplay_alive():
            self._mpv_command("set_property", "pause", True)
            self._mpv_command("seek", round(self.current_sec, 3), "absolute")

    def on_tree_click_force(self, event) -> None:
        row = self.tree.identify_row(event.y)
        if not row:
            return
        idx = int(row)
        self.tree.selection_set(row)
        self._jump_to_entry(idx, stop_if_playing=True)

    def on_tree_select(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._jump_to_entry(idx, stop_if_playing=True)

    def _refresh_row(self, idx: int) -> None:
        e = self.entries[idx]
        self.tree.item(str(idx), values=(idx + 1, format_srt_timecode(e.start_ms), format_srt_timecode(e.end_ms), e.text))

    def apply_edit(self) -> None:
        if self.selected_index is None:
            return
        idx = self.selected_index
        try:
            start_ms = parse_time_input_to_ms(self.start_var.get())
            end_ms = parse_time_input_to_ms(self.end_var.get())
        except Exception:
            messagebox.showerror("Loi", "Sai dinh dang time. Dung HH:MM:SS,mmm hoac giay.", parent=self)
            return

        if end_ms <= start_ms:
            end_ms = start_ms + 100

        self.entries[idx] = SubtitleEntry(start_ms=start_ms, end_ms=end_ms, text=self.text_var.get().strip())
        self._refresh_row(idx)
        self._refresh_overlay_srt()
        self.current_sec = start_ms / 1000.0
        self._update_time_ui()

    def add_sub_below(self) -> None:
        if self.selected_index is None:
            return

        idx = self.selected_index
        cur = self.entries[idx]
        next_start = self.video_len_sec * 1000 if self.video_len_sec > 0 else (cur.end_ms + 4000)
        if idx + 1 < len(self.entries):
            next_start = self.entries[idx + 1].start_ms

        new_start = cur.end_ms
        new_end = min(new_start + 1200, max(new_start + 120, int(next_start)))
        if new_end <= new_start:
            new_end = new_start + 120

        new_entry = SubtitleEntry(start_ms=int(new_start), end_ms=int(new_end), text="")
        insert_idx = idx + 1
        self.entries.insert(insert_idx, new_entry)

        self._load_tree()
        self._refresh_overlay_srt()

        row_id = str(insert_idx)
        self.tree.selection_set(row_id)
        self.tree.focus(row_id)
        self.tree.see(row_id)
        self._jump_to_entry(insert_idx, stop_if_playing=True)

    def delete_selected_sub(self, _evt=None):
        if self.selected_index is None or not self.entries:
            return "break"

        idx = self.selected_index
        if idx < 0 or idx >= len(self.entries):
            return "break"

        del self.entries[idx]
        self._load_tree()
        self._refresh_overlay_srt()

        if not self.entries:
            self.selected_index = None
            self.start_var.set("")
            self.end_var.set("")
            self.text_var.set("")
            return "break"

        new_idx = min(idx, len(self.entries) - 1)
        row_id = str(new_idx)
        self.tree.selection_set(row_id)
        self.tree.focus(row_id)
        self.tree.see(row_id)
        self._jump_to_entry(new_idx, stop_if_playing=True)
        return "break"

    def shift_selected(self, delta_ms: int) -> None:
        if self.selected_index is None:
            return
        idx = self.selected_index
        e = self.entries[idx]
        start_ms = max(0, e.start_ms + delta_ms)
        end_ms = max(start_ms + 80, e.end_ms + delta_ms)
        self.entries[idx] = SubtitleEntry(start_ms=start_ms, end_ms=end_ms, text=e.text)
        self._refresh_row(idx)
        self._refresh_overlay_srt()
        self.start_var.set(format_srt_timecode(start_ms))
        self.end_var.set(format_srt_timecode(end_ms))

    def set_start_to_current(self) -> None:
        if self.selected_index is None:
            return
        self.start_var.set(format_srt_timecode(self.current_ms()))
        self.apply_edit()

    def set_end_to_current(self) -> None:
        if self.selected_index is None:
            return
        self.end_var.set(format_srt_timecode(self.current_ms()))
        self.apply_edit()

    def snap_selected_to_speech(self) -> None:
        if self.selected_index is None:
            return
        idx = self.selected_index
        e = self.entries[idx]
        result = detect_speech_bounds(self.media_path, e.start_ms, e.end_ms)
        if not result:
            messagebox.showwarning("Canh bao", "Khong tim duoc ranh gioi giong noi cho doan nay.", parent=self)
            return

        s0, s1 = result
        self.entries[idx] = SubtitleEntry(start_ms=s0, end_ms=s1, text=e.text)
        self._refresh_row(idx)
        self._refresh_overlay_srt()
        self.start_var.set(format_srt_timecode(s0))
        self.end_var.set(format_srt_timecode(s1))

    def save_srt(self) -> None:
        write_srt(self.srt_path, self.entries)
        messagebox.showinfo("OK", f"Da luu: {self.srt_path}", parent=self)

    def save_progress(self) -> None:
        default_name = f"{self.srt_path.stem}.json"
        path = filedialog.asksaveasfilename(
            title="Luu tien trinh",
            initialdir=str(SAVE_DIR),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("Progress", "*.json")],
            parent=self,
        )
        if not path:
            return

        progress_path = Path(path)
        meta = {"source": "editor"}
        save_progress_json(progress_path, self.media_path, self.srt_path, meta)
        messagebox.showinfo("OK", f"Da luu tien trinh: {progress_path}", parent=self)

    def reload_srt(self) -> None:
        self.entries = parse_srt(self.srt_path)
        self._load_tree()
        self._refresh_overlay_srt()

    def _on_close(self) -> None:
        if self.play_tick_after:
            self.after_cancel(self.play_tick_after)
            self.play_tick_after = None
        self._stop_ffplay()
        self.last_tick_wall = 0.0
        shutil.rmtree(self.overlay_dir, ignore_errors=True)
        self.destroy()


class AdvancedSettingsWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, app: "TranslatorApp"):
        super().__init__(master)
        self.app = app
        self.title("Che do nang cao")
        self.geometry("760x520")
        self.minsize(700, 480)
        self.transient(master)

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Preset:").grid(row=0, column=0, sticky="w")
        preset_values = ["Tuy chinh"] + list(ADVANCED_PRESETS.keys())
        preset_combo = ttk.Combobox(frame, textvariable=self.app.advanced_preset, state="readonly", values=preset_values, width=20)
        preset_combo.grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Button(frame, text="Ap dung preset", command=self._apply_preset).grid(row=0, column=2, sticky="w")

        ttk.Label(frame, text="Nhom 1: Sampling / Decoding").grid(row=1, column=0, columnspan=4, sticky="w", pady=(14, 4))
        ttk.Label(frame, text="Temperature (-tp)").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.app.temperature, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(frame, text="Temperature-inc (-tpi)").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.app.temperature_inc, width=8).grid(row=2, column=3, sticky="w")
        ttk.Label(frame, text="Best-of (-bo)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=10, increment=1, textvariable=self.app.best_of, width=8).grid(row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Label(frame, text="Beam-size (-bs)").grid(row=3, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=12, increment=1, textvariable=self.app.beam_size, width=8).grid(row=3, column=3, sticky="w", pady=(6, 0))

        ttk.Label(frame, text="Nhom 2: Drop doan / thieu cau").grid(row=4, column=0, columnspan=4, sticky="w", pady=(14, 4))
        ttk.Label(frame, text="No-speech-thold (-nth)").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.app.no_speech_thold, width=8).grid(row=5, column=1, sticky="w")
        ttk.Label(frame, text="Entropy-thold (-et)").grid(row=5, column=2, sticky="w")
        ttk.Spinbox(frame, from_=0.0, to=10.0, increment=0.1, textvariable=self.app.entropy_thold, width=8).grid(row=5, column=3, sticky="w")
        ttk.Label(frame, text="Logprob-thold (-lpt)").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=-5.0, to=0.0, increment=0.1, textvariable=self.app.logprob_thold, width=8).grid(row=6, column=1, sticky="w", pady=(6, 0))

        ttk.Label(frame, text="Nhom 3: VAD").grid(row=7, column=0, columnspan=4, sticky="w", pady=(14, 4))
        ttk.Checkbutton(frame, text="Bat VAD (--vad)", variable=self.app.use_vad).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="VAD threshold (-vt)").grid(row=8, column=2, sticky="w")
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.app.vad_threshold, width=8).grid(row=8, column=3, sticky="w")

        ttk.Label(frame, text="Nhom 4: Context").grid(row=9, column=0, columnspan=4, sticky="w", pady=(14, 4))
        ttk.Label(frame, text="Max-context (-mc)").grid(row=10, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=2048, increment=64, textvariable=self.app.max_context, width=8).grid(row=10, column=1, sticky="w")
        ttk.Checkbutton(frame, text="No fallback (-nf)", variable=self.app.no_fallback).grid(row=10, column=2, columnspan=2, sticky="w")

        ttk.Label(frame, text="Nhom 5: Language").grid(row=11, column=0, columnspan=4, sticky="w", pady=(14, 4))
        ttk.Label(frame, text="Language (-l)").grid(row=12, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.app.source_lang, state="readonly", values=LANG_OPTIONS, width=10).grid(row=12, column=1, sticky="w")

        ttk.Label(
            frame,
            text="Goi y: temp 0.2-0.3 can bang, beam 5-8 chinh xac hon, nth 0.2-0.4 giu duoc nhieu cau.",
        ).grid(row=13, column=0, columnspan=4, sticky="w", pady=(14, 8))

        btns = ttk.Frame(frame)
        btns.grid(row=14, column=0, columnspan=4, sticky="e")
        ttk.Button(btns, text="Dong", command=self.destroy).pack(side="right")

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    def _apply_preset(self) -> None:
        name = self.app.advanced_preset.get().strip()
        self.app.apply_advanced_preset(name)


class MissingComponentsWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, app: "TranslatorApp", missing_specs: list[dict]):
        super().__init__(master)
        self.app = app
        self.missing_specs = missing_specs
        self.url_vars: dict[str, tk.StringVar] = {}

        self.title("Tai file con thieu")
        self.geometry("860x360")
        self.transient(master)

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Cac thanh phan con thieu - dien URL truc tiep (file hoac zip):").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        for idx, spec in enumerate(self.missing_specs, start=1):
            ttk.Label(frame, text=f"{spec['label']} -> {spec['path']}").grid(row=idx * 2 - 1, column=0, columnspan=3, sticky="w")
            default_url = DEFAULT_DOWNLOAD_URLS.get(spec["key"], "")
            var = tk.StringVar(value=default_url)
            self.url_vars[spec["key"]] = var
            ttk.Entry(frame, textvariable=var).grid(row=idx * 2, column=0, columnspan=2, sticky="we", pady=(2, 8))
            ttk.Button(frame, text="Paste", command=lambda v=var: self._paste(v)).grid(row=idx * 2, column=2, padx=(8, 0), sticky="e")

        btns = ttk.Frame(frame)
        btns.grid(row=999, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Tai ngay", command=self._start_download).pack(side="right")
        ttk.Button(btns, text="Dong", command=self.destroy).pack(side="right", padx=(0, 8))

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _paste(self, var: tk.StringVar) -> None:
        try:
            var.set(self.clipboard_get().strip())
        except Exception:
            pass

    def _start_download(self) -> None:
        url_by_key = {k: v.get().strip() for k, v in self.url_vars.items()}
        self.app.start_download_missing(self.missing_specs, url_by_key)
        self.destroy()
class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"WhisperSUB_V v{APP_VERSION} - Subtitle Translator (Whisper)")
        self.root.geometry("900x640")

        self.audio_path = tk.StringVar()
        self.whisper_cli_path = tk.StringVar(value=str(guess_whisper_cli() or ""))
        self.source_lang = tk.StringVar(value="auto")
        self.profile = tk.StringVar(value="Can bang")
        self.max_subtitle_len = tk.IntVar(value=DEFAULT_MAX_SUBTITLE_LEN)
        self.use_gpu = tk.BooleanVar(value=True)
        self.use_vad = tk.BooleanVar(value=True)
        self.translate_to_english = tk.BooleanVar(value=False)
        self.clean_repetitions = tk.BooleanVar(value=True)

        self.temperature = tk.DoubleVar(value=0.25)
        self.temperature_inc = tk.DoubleVar(value=0.20)
        self.best_of = tk.IntVar(value=5)
        self.beam_size = tk.IntVar(value=5)
        self.no_speech_thold = tk.DoubleVar(value=0.72)
        self.entropy_thold = tk.DoubleVar(value=2.4)
        self.logprob_thold = tk.DoubleVar(value=-1.0)
        self.vad_threshold = tk.DoubleVar(value=0.50)
        self.max_context = tk.IntVar(value=0)
        self.no_fallback = tk.BooleanVar(value=False)
        self.advanced_preset = tk.StringVar(value="Tuy chinh")

        self.is_running = False
        self.advanced_window: AdvancedSettingsWindow | None = None
        self.download_window: MissingComponentsWindow | None = None
        self.last_media: Path | None = None
        self.last_srt: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=str(MODEL_PATH)).grid(row=0, column=1, columnspan=4, sticky="we", pady=(0, 8))

        ttk.Label(frame, text="Whisper CLI:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.whisper_cli_path).grid(row=1, column=1, columnspan=3, sticky="we", padx=(8, 8))
        ttk.Button(frame, text="Chon", command=self.choose_cli).grid(row=1, column=4, sticky="e")

        ttk.Label(frame, text="File video/audio:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.audio_path).grid(row=2, column=1, columnspan=3, sticky="we", padx=(8, 8))
        ttk.Button(frame, text="Chon", command=self.choose_audio).grid(row=2, column=4, sticky="e")

        ttk.Label(frame, text="Ngon ngu nguon:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(frame, textvariable=self.source_lang, state="readonly", values=LANG_OPTIONS, width=10).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Profile:").grid(row=3, column=2, sticky="e", pady=(8, 0))
        ttk.Combobox(frame, textvariable=self.profile, state="readonly", values=["Nhanh", "Can bang", "Chinh xac"], width=12).grid(row=3, column=3, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Max ky tu/sub:").grid(row=3, column=4, sticky="w", pady=(8, 0))
        ttk.Spinbox(frame, from_=30, to=120, increment=5, textvariable=self.max_subtitle_len, width=6).grid(row=3, column=4, sticky="e", pady=(8, 0))

        ttk.Checkbutton(frame, text="Dung GPU neu backend ho tro", variable=self.use_gpu).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(frame, text="Bat VAD (giam lap cau)", variable=self.use_vad).grid(row=4, column=2, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(frame, text="Lam sach lap cau", variable=self.clean_repetitions).grid(row=4, column=4, sticky="e", pady=(8, 0))

        ttk.Checkbutton(frame, text="Dich sang tieng Anh (-tr)", variable=self.translate_to_english).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

        btns = ttk.Frame(frame)
        btns.grid(row=6, column=0, columnspan=5, sticky="we", pady=(12, 8))
        self.run_btn = ttk.Button(btns, text="Xuat SRT", command=self.start_translate)
        self.run_btn.pack(side="left")
        ttk.Button(btns, text="Xoa ket qua", command=self.clear_output).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Save Progress", command=self.save_progress_main).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Load Progress", command=self.load_progress_main).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Che do nang cao", command=self.open_advanced_mode).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Tai file thieu", command=self.open_missing_components).pack(side="left", padx=(8, 0))
        self.open_editor_btn = ttk.Button(btns, text="Mo SRT Editor", command=self.open_editor, state="disabled")
        self.open_editor_btn.pack(side="left", padx=(8, 0))

        self.status_label = ttk.Label(frame, text="San sang")
        self.status_label.grid(row=7, column=0, columnspan=5, sticky="w")

        self.output = tk.Text(frame, wrap="word", height=24)
        self.output.grid(row=8, column=0, columnspan=5, sticky="nsew", pady=(8, 0))

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        frame.columnconfigure(4, weight=1)
        frame.rowconfigure(8, weight=1)

    def _get_missing_components(self) -> list[dict]:
        missing: list[dict] = []
        for spec in required_components():
            p: Path = spec["path"]
            if spec["key"] == "whisper_cli":
                selected = self.whisper_cli_path.get().strip()
                if selected and Path(selected).exists():
                    continue
            if not p.exists():
                missing.append(spec)
        return missing

    def open_missing_components(self) -> None:
        missing = self._get_missing_components()
        if not missing:
            messagebox.showinfo("Thong bao", "Khong co thanh phan nao bi thieu.")
            return
        if self.download_window and self.download_window.winfo_exists():
            self.download_window.lift()
            self.download_window.focus_force()
            return
        self.download_window = MissingComponentsWindow(self.root, self, missing)

    def start_download_missing(self, specs: list[dict], url_by_key: dict[str, str]) -> None:
        threading.Thread(target=self._download_missing_worker, args=(specs, url_by_key), daemon=True).start()

    def _download_missing_worker(self, specs: list[dict], url_by_key: dict[str, str]) -> None:
        self.root.after(0, lambda: self.set_status("Dang tai file con thieu..."))
        self.root.after(0, lambda: self.append_output("[Setup] Bat dau tai file con thieu...\n"))

        ok_count = 0
        fail_count = 0
        for spec in specs:
            key = spec["key"]
            label = spec["label"]
            target: Path = spec["path"]
            aliases: list[str] = spec["aliases"]
            url = (url_by_key.get(key, "") or "").strip()

            if not url:
                fail_count += 1
                self.root.after(0, lambda l=label: self.append_output(f"[Setup] Thieu URL cho {l}\n"))
                continue

            try:
                self.root.after(0, lambda l=label: self.append_output(f"[Setup] Dang tai {l} ...\n"))
                _save_component_from_url(url, target, aliases)
                if not target.exists():
                    raise RuntimeError("Khong tao duoc file dich")

                if key == "whisper_cli":
                    self.root.after(0, lambda p=str(target): self.whisper_cli_path.set(p))

                ok_count += 1
                self.root.after(0, lambda p=target: self.append_output(f"[Setup] OK: {p}\n"))
            except Exception as exc:
                fail_count += 1
                self.root.after(0, lambda l=label, e=str(exc): self.append_output(f"[Setup] Loi {l}: {e}\n"))

        def finish() -> None:
            self.append_output(f"[Setup] Hoan tat. Thanh cong: {ok_count}, That bai: {fail_count}\n")
            self.set_status("San sang")
            if fail_count == 0:
                messagebox.showinfo("Thong bao", "Da tai xong cac thanh phan bi thieu.")
            else:
                messagebox.showwarning("Thong bao", "Da tai xong mot phan. Kiem tra log de bo sung URL con thieu.")

        self.root.after(0, finish)
    def choose_cli(self) -> None:
        path = filedialog.askopenfilename(title="Chon whisper-cli.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.whisper_cli_path.set(path)

    def choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Chon file video/audio",
            filetypes=[("Media", "*.wav *.mp3 *.m4a *.mp4 *.flac *.ogg *.webm *.mkv *.avi"), ("All files", "*.*")],
        )
        if path:
            self.audio_path.set(path)

    def clear_output(self) -> None:
        self.output.delete("1.0", "end")

    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def append_output(self, text: str) -> None:
        self.output.insert("end", text)
        self.output.see("end")
        msg = text.strip()
        if msg:
            LOGGER.info(msg)

    def open_editor(self) -> None:
        if not self.last_media or not self.last_srt or not self.last_srt.exists():
            messagebox.showwarning("Canh bao", "Chua co ket qua SRT de mo editor.")
            return
        SubtitleEditorWindow(self.root, self.last_media, self.last_srt)
    def open_advanced_mode(self) -> None:
        if self.advanced_window and self.advanced_window.winfo_exists():
            self.advanced_window.lift()
            self.advanced_window.focus_force()
            return
        self.advanced_window = AdvancedSettingsWindow(self.root, self)

    def apply_advanced_preset(self, preset_name: str) -> None:
        preset = ADVANCED_PRESETS.get(preset_name)
        if not preset:
            return
        self.temperature.set(float(preset.get("temperature", self.temperature.get())))
        self.beam_size.set(int(preset.get("beam_size", self.beam_size.get())))
        self.best_of.set(int(preset.get("best_of", self.best_of.get())))
        self.no_speech_thold.set(float(preset.get("no_speech_thold", self.no_speech_thold.get())))
        self.entropy_thold.set(float(preset.get("entropy_thold", self.entropy_thold.get())))
        self.logprob_thold.set(float(preset.get("logprob_thold", self.logprob_thold.get())))
        self.use_vad.set(bool(preset.get("use_vad", self.use_vad.get())))
        self.vad_threshold.set(float(preset.get("vad_threshold", self.vad_threshold.get())))
        lang = str(preset.get("source_lang", self.source_lang.get()))
        if lang in LANG_OPTIONS:
            self.source_lang.set(lang)
    def save_progress_main(self) -> None:
        media: Path | None = None
        if self.audio_path.get().strip():
            media = Path(self.audio_path.get().strip())
        elif self.last_media:
            media = self.last_media

        if not media or not media.exists():
            messagebox.showwarning("Canh bao", "Chua co file media hop le de luu tien trinh.")
            return

        srt = media.with_suffix(".srt")
        if self.last_srt and self.last_srt.exists():
            srt = self.last_srt

        if not srt.exists():
            messagebox.showwarning("Canh bao", "Chua co file SRT. Hay xuat SRT hoac mo editor truoc.")
            return

        default_name = f"{srt.stem}.json"
        path = filedialog.asksaveasfilename(
            title="Luu tien trinh",
            initialdir=str(SAVE_DIR),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("Progress", "*.json")],
        )
        if not path:
            return

        meta = {
            "source": "main",
            "profile": self.profile.get(),
            "source_lang": self.source_lang.get(),
            "temperature": float(self.temperature.get()),
            "temperature_inc": float(self.temperature_inc.get()),
            "best_of": int(self.best_of.get()),
            "beam_size": int(self.beam_size.get()),
            "no_speech_thold": float(self.no_speech_thold.get()),
            "entropy_thold": float(self.entropy_thold.get()),
            "logprob_thold": float(self.logprob_thold.get()),
            "vad_threshold": float(self.vad_threshold.get()),
            "max_context": int(self.max_context.get()),
            "no_fallback": bool(self.no_fallback.get()),
            "advanced_preset": self.advanced_preset.get(),
            "translate_to_english": bool(self.translate_to_english.get()),
            "clean_repetitions": bool(self.clean_repetitions.get()),
        }
        progress_path = Path(path)
        save_progress_json(progress_path, media, srt, meta)
        self.append_output(f"[Progress] Da luu: {progress_path}\n")

    def load_progress_main(self) -> None:
        path = filedialog.askopenfilename(
            title="Mo tien trinh",
            initialdir=str(SAVE_DIR),
            filetypes=[("Progress", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            media, srt, meta, legacy_entries = load_progress_json(Path(path))
        except Exception as exc:
            messagebox.showerror("Loi", f"Doc file tien trinh that bai: {exc}")
            return

        if not media.exists():
            messagebox.showerror("Loi", f"Khong tim thay media: {media}")
            return

        if not srt.exists():
            if legacy_entries:
                srt.parent.mkdir(parents=True, exist_ok=True)
                write_srt(srt, legacy_entries)
            else:
                messagebox.showerror("Loi", f"Khong tim thay SRT: {srt}")
                return

        self.audio_path.set(str(media))
        self.last_media = media
        self.last_srt = srt
        self.open_editor_btn.config(state="normal")

        if isinstance(meta, dict):
            if meta.get("profile") in ["Nhanh", "Can bang", "Chinh xac"]:
                self.profile.set(meta.get("profile"))
            if meta.get("source_lang") in LANG_OPTIONS:
                self.source_lang.set(meta.get("source_lang"))
            self.translate_to_english.set(bool(meta.get("translate_to_english", self.translate_to_english.get())))
            self.temperature.set(float(meta.get("temperature", self.temperature.get())))
            self.temperature_inc.set(float(meta.get("temperature_inc", self.temperature_inc.get())))
            self.best_of.set(int(meta.get("best_of", self.best_of.get())))
            self.beam_size.set(int(meta.get("beam_size", self.beam_size.get())))
            self.no_speech_thold.set(float(meta.get("no_speech_thold", self.no_speech_thold.get())))
            self.entropy_thold.set(float(meta.get("entropy_thold", self.entropy_thold.get())))
            self.logprob_thold.set(float(meta.get("logprob_thold", self.logprob_thold.get())))
            self.vad_threshold.set(float(meta.get("vad_threshold", self.vad_threshold.get())))
            self.max_context.set(int(meta.get("max_context", self.max_context.get())))
            self.no_fallback.set(bool(meta.get("no_fallback", self.no_fallback.get())))
            self.advanced_preset.set(str(meta.get("advanced_preset", self.advanced_preset.get())))
            self.clean_repetitions.set(bool(meta.get("clean_repetitions", self.clean_repetitions.get())))

        self.append_output(f"[Progress] Da load: {path}\n")
        self.append_output(f"[Progress] Media: {media}\n")
        self.append_output(f"[Progress] SRT: {srt}\n")
        self.set_status("Da load tien trinh")

    def start_translate(self) -> None:
        if self.is_running:
            return

        cli = Path(self.whisper_cli_path.get().strip()) if self.whisper_cli_path.get().strip() else None
        media = Path(self.audio_path.get().strip()) if self.audio_path.get().strip() else None

        if not MODEL_PATH.exists():
            messagebox.showerror("Loi", f"Khong tim thay model: {MODEL_PATH}")
            return
        if not cli or not cli.exists():
            messagebox.showerror("Loi", "Khong tim thay whisper-cli.exe. Hay chon file CLI.")
            return
        if not media or not media.exists():
            messagebox.showerror("Loi", "Khong tim thay file video/audio.")
            return

        self.is_running = True
        self.run_btn.config(state="disabled")
        self.open_editor_btn.config(state="disabled")
        self.clear_output()
        self.set_status("Dang xu ly...")

        threading.Thread(target=self.run_translate, args=(cli, media), daemon=True).start()

    def run_translate(self, cli: Path, media: Path) -> None:
        srt_output = media.with_suffix(".srt")
        temp_dir: Path | None = None
        total_t0 = time.perf_counter()

        try:
            wav_for_whisper, temp_dir, convert_elapsed = prepare_audio_for_whisper(media)
            _, _, threads = profile_params(self.profile.get())
            max_len = max(30, int(self.max_subtitle_len.get()))

            best_of = max(1, int(self.best_of.get()))
            beam_size = max(1, int(self.beam_size.get()))
            temperature = min(1.0, max(0.0, float(self.temperature.get())))
            temperature_inc = min(1.0, max(0.0, float(self.temperature_inc.get())))
            no_speech_thold = min(1.0, max(0.0, float(self.no_speech_thold.get())))
            entropy_thold = max(0.0, float(self.entropy_thold.get()))
            logprob_thold = min(0.0, float(self.logprob_thold.get()))
            max_context = max(0, int(self.max_context.get()))
            vad_threshold = min(1.0, max(0.0, float(self.vad_threshold.get())))
            vad_model = find_vad_model_path()
            use_vad = (vad_model is not None) and self.use_vad.get()


            output_prefix = (temp_dir / "result") if temp_dir else media.with_suffix("")

            base_cmd = [
                to_windows_short_path(cli),
                "-m",
                to_windows_short_path(MODEL_PATH),
                "-f",
                to_windows_short_path(wav_for_whisper),
                "-l",
                self.source_lang.get(),
                "-osrt",
                "-of",
                to_windows_short_path(output_prefix),
                "-sow",
                "-ml",
                str(max_len),
                "-bo",
                str(best_of),
                "-bs",
                str(beam_size),
                "-t",
                str(threads),
                "-nth",
                str(no_speech_thold),
                "-et",
                str(entropy_thold),
                "-lpt",
                str(logprob_thold),
                "-mc",
                str(max_context),
                "-tp",
                str(temperature),
                "-tpi",
                str(temperature_inc),
            ]

            if self.translate_to_english.get():
                base_cmd.append("-tr")
            if not self.use_gpu.get():
                base_cmd.append("-ng")
            if self.no_fallback.get():
                base_cmd.append("-nf")

            pass1_t0 = time.perf_counter()
            cmd = list(base_cmd)
            if use_vad:
                cmd.extend(["--vad", "-vm", to_windows_short_path(vad_model), "-vt", str(vad_threshold)])

            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **subprocess_no_window_kwargs())
            err = proc.stderr.strip()
            vad_fallback_used = False

            if proc.returncode != 0 and use_vad and is_vad_model_error(err):
                vad_fallback_used = True
                proc = subprocess.run(base_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **subprocess_no_window_kwargs())

            pass1_elapsed = time.perf_counter() - pass1_t0
            err = proc.stderr.strip()

            generated_srt = output_prefix.with_suffix(".srt")
            if generated_srt.exists():
                shutil.copyfile(generated_srt, srt_output)

            post_elapsed = 0.0
            changed_count = 0
            if proc.returncode == 0 and srt_output.exists():
                post_t0 = time.perf_counter()
                entries = parse_srt(srt_output)
                entries, changed_count = postprocess_srt(
                    entries,
                    self.clean_repetitions.get(),
                )
                write_srt(srt_output, entries)
                post_elapsed = time.perf_counter() - post_t0

            total_elapsed = time.perf_counter() - total_t0

            def finish() -> None:
                if proc.returncode != 0:
                    self.append_output("Chay that bai.\n\n")
                    if err:
                        self.append_output(err + "\n")
                    self.set_status("Loi khi xu ly")
                else:
                    self.last_media = media
                    self.last_srt = srt_output
                    self.open_editor_btn.config(state="normal")

                    self.append_output(f"[Da xuat SRT] {srt_output}\n")
                    self.append_output("\n[Timing]\n")
                    self.append_output(f"- ffmpeg trich audio: {convert_elapsed:.2f}s\n")
                    self.append_output(f"- whisper pass chinh: {pass1_elapsed:.2f}s\n")
                    self.append_output(f"- hau xu ly SRT: {post_elapsed:.2f}s (da sua {changed_count} doan)\n")
                    self.append_output(f"- tong thoi gian: {total_elapsed:.2f}s\n")

                    if use_vad and not vad_fallback_used:
                        self.append_output("\n[VAD] Dang bat VAD model de giam lap/hallucination.\n")
                    if vad_fallback_used:
                        self.append_output("\n[VAD] Model VAD khong tuong thich voi whisper-cli hien tai, da tu dong chay lai khong VAD.\n")

                    if "using CUDA0 backend" in err:
                        self.append_output("[GPU] Dang dung CUDA0 backend.\n")
                    elif "no GPU found" in err and self.use_gpu.get():
                        self.append_output("[Canh bao GPU] Dang roi ve CPU.\n")

                    if err:
                        self.append_output("\n[Log]\n" + err + "\n")

                    self.set_status("Hoan tat")

                self.is_running = False
                self.run_btn.config(state="normal")

            self.root.after(0, finish)

        except Exception as exc:
            def fail() -> None:
                self.append_output(f"Loi: {exc}\n")
                self.set_status("Loi he thong")
                self.is_running = False
                self.run_btn.config(state="normal")

            self.root.after(0, fail)
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    setup_logging()
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()







































































































