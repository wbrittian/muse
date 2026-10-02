"""Melody sources beyond BiMMuDa. Each is fetched into data/raw/ by
scripts/download_data.sh and loaded into the same record shape as
collect_features: the 11 features plus "source", "song" and "notes"."""
import gzip
import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pandas import read_csv
from pretty_midi import PrettyMIDI

from pysrc.data_client.collect_features import melody_features
from pysrc.data_client.tokenizer import QDIV, Notes, clean_notes, quantize_notes

PREP_HINT = "./scripts/download_data.sh"
SEGMENT_BARS = 8


@dataclass
class Source:
    available: Callable[[Path], bool]
    load: Callable[[Path], list[dict[str, Any]]]
    prep_hint: str = PREP_HINT


def segment(notes: Notes, bar_divs: int, bars: int = SEGMENT_BARS) -> list[tuple[Notes, int]]:
    """Cut a whole song into non-overlapping `bars`-bar windows on bar lines.

    Times are made relative to each window's first bar line, so a leading
    rest still says where in the bar the phrase starts. Returns (notes, bars)
    pairs; the last window may be shorter.
    """
    win = bar_divs * bars
    first_bar = notes[0][0] // bar_divs
    windows: dict[int, Notes] = {}
    for start, pitch, dur in notes:
        w = (start - first_bar * bar_divs) // win
        origin = first_bar * bar_divs + w * win
        windows.setdefault(w, []).append((start - origin, pitch, min(dur, origin + win - start)))

    out = []
    for w in sorted(windows):
        seg = windows[w]
        end = max(s + d for s, _, d in seg)
        out.append((seg, max(1, ceil(end / bar_divs))))
    return out


def _mode(intervals: list[int]) -> str:
    # third above the tonic: 4 semitones = major family, 3 = minor family
    return "major" if sum(intervals[:2]) == 4 else "minor"


# ---------------------------------------------------------------- POP909

KEY_MODES = {"maj": "major", "min": "minor"}


def _pop909_available(base: Path) -> bool:
    return (base / "raw/POP909/001/001.mid").exists()


def _pop909_load(base: Path) -> list[dict[str, Any]]:
    """909 Chinese-pop songs with a human-transcribed MELODY track.

    The beat grid comes from beat_midi.txt (column 3 marks 4/4 downbeats); the
    key from key_audio.txt. Genre is Pop by construction; era is unknown.
    """
    melodies = []
    for song_dir in sorted((base / "raw/POP909").iterdir()):
        if not song_dir.is_dir():
            continue
        sid = song_dir.name
        pm = PrettyMIDI(str(song_dir / f"{sid}.mid"))
        melody_track = [i for i in pm.instruments if i.name == "MELODY"]
        beats = np.loadtxt(song_dir / "beat_midi.txt")
        keys = read_csv(song_dir / "key_audio.txt", sep="\t", header=None)
        if not melody_track or len(keys) != 1 or len(beats) < 8:
            continue   # skip songs that change key

        # make bar 0 start on the first annotated 4/4 downbeat
        downbeats = np.flatnonzero(beats[:, 2] == 1)
        beat_times = beats[downbeats[0]:, 0]
        bpm = 60.0 / float(np.median(np.diff(beat_times)))
        notes = quantize_notes(pm, beat_times=beat_times, instruments=melody_track)
        notes = clean_notes([n for n in notes if n[0] >= 0])
        if len(notes) < 4:
            continue
        mode = KEY_MODES.get(str(keys.iloc[0, 2]).split(":")[-1], "major")

        for seg, bars in segment(notes, 4 * QDIV):
            if len(seg) < 4:
                continue
            row = melody_features(seg, bpm, "4/4", bars, mode, "Pop", "Unknown")
            row.update({"source": "pop909", "song": sid, "notes": seg})
            melodies.append(row)
    return melodies


# ------------------------------------------------------------ HookTheory

METERS = {(4, 4): "4/4", (3, 4): "3/4"}
SKIP_TAGS = {"TEMPO_CHANGES", "KEY_CHANGES", "METER_CHANGES", "SWING_CHANGES"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _bimmuda_songs(base: Path) -> set[str]:
    songs = read_csv(base / "metadata/bimmuda_per_song_metadata.csv")
    return {_slug(t) for t in songs["Title"].astype(str)}


def _hooktheory_available(base: Path) -> bool:
    return (base / "raw/Hooktheory.json.gz").exists()


def _hooktheory_load(base: Path) -> list[dict[str, Any]]:
    """~24k pop/rock section excerpts annotated on HookTheory (Sheet Sage release).

    Pitches are octave-relative, so each melody is placed with its median
    pitch in C4-B4. Tempo comes from the beat-to-audio alignment. Genre and
    era are not annotated. Songs whose title matches a BiMMuDa song are
    dropped so held-out BiMMuDa songs cannot leak into training.
    """
    with gzip.open(base / "raw/Hooktheory.json.gz") as f:
        data = json.load(f)
    bimmuda = _bimmuda_songs(base)

    melodies = []
    for clip_id, entry in data.items():
        tags = set(entry["tags"])
        ann = entry["annotations"]
        if "MELODY" not in tags or "NO_SWING" not in tags or tags & SKIP_TAGS or not ann["melody"]:
            continue
        if _slug(entry["hooktheory"]["song"]) in bimmuda:
            continue
        meter = ann["meters"][0]
        ts = METERS.get((meter["beats_per_bar"], meter["beat_unit"]))
        align = entry["alignment"]["refined"] or entry["alignment"]["user"]
        if ts is None or not align or len(align["beats"]) < 2:
            continue

        beats, times = align["beats"], align["times"]
        bpm = 60.0 * (beats[-1] - beats[0]) / (times[-1] - times[0])
        if not 40 <= bpm <= 240:
            continue

        bar_divs = meter["beats_per_bar"] * QDIV
        raw = [
            (round(n["onset"] * QDIV), 60 + 12 * n["octave"] + n["pitch_class"],
             max(1, round((n["offset"] - n["onset"]) * QDIV)))
            for n in ann["melody"]
        ]
        shift = (min(s for s, _, _ in raw) // bar_divs) * bar_divs
        notes = clean_notes([(s - shift, p, d) for s, p, d in raw])
        octave = (int(np.median([p for _, p, _ in notes])) - 60) // 12
        notes = [(s, p - 12 * octave, d) for s, p, d in notes]
        if len(notes) < 4:
            continue

        bars = ceil(max(s + d for s, _, d in notes) / bar_divs)
        mode = _mode(ann["keys"][0]["scale_degree_intervals"])
        row = melody_features(notes, bpm, ts, bars, mode, "Unknown", "Unknown")
        song = f"{entry['hooktheory']['artist']}/{entry['hooktheory']['song']}"
        row.update({"source": "hooktheory", "song": song, "notes": notes})
        melodies.append(row)
    return melodies


EXTRA_SOURCES: dict[str, Source] = {
    "pop909": Source(_pop909_available, _pop909_load),
    "hooktheory": Source(_hooktheory_available, _hooktheory_load),
}
