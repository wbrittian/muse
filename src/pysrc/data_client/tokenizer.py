from typing import Any
from pretty_midi import PrettyMIDI

import numpy as np

QDIV = 12          # divisions per beat
MAX_NOTE = 48      # longest NOTE token, in divisions
MAX_REST = 12      # longest REST token, in divisions
MIN_PITCH, MAX_PITCH = -29, 28   # PITCH token range, relative to the first note

FEATURES = ["BPM", "TS", "BARS", "FIRST", "LAST", "MODE", "GENRE", "ERA", "CONTOUR", "DENSITY", "RANGE"]

Notes = list[tuple[int, int, int]]   # (start_div, midi_pitch, dur_divs)


def bar_divs(ts: str) -> int:
    """Divisions per bar; x/8 meters are compound, with dotted-quarter beats (as pretty_midi counts them)."""
    num, den = map(int, ts.split("/"))
    return num * QDIV if den == 4 else num // 3 * QDIV


def feature_to_token(key: str, val: Any, tok2id: dict[str, int]) -> int:
        raw_token = f"<{key}_{val}>"
        if raw_token not in tok2id:
            print("no token match")

        return tok2id[raw_token]


def clean_notes(notes: Notes) -> Notes:
    """Make a note list strictly monophonic and representable by the vocab.

    Simultaneous onsets keep the highest pitch, an overlapping note is cut at
    the next onset, and durations are capped at MAX_NOTE divisions.
    """
    by_start: dict[int, tuple[int, int, int]] = {}
    for start, pitch, dur in notes:
        if start not in by_start or pitch > by_start[start][1]:
            by_start[start] = (start, pitch, dur)
    ordered = [by_start[s] for s in sorted(by_start)]

    cleaned = []
    for i, (start, pitch, dur) in enumerate(ordered):
        if i + 1 < len(ordered):
            dur = min(dur, ordered[i + 1][0] - start)
        cleaned.append((start, pitch, max(1, min(dur, MAX_NOTE))))
    return cleaned


def quantize_notes(
        pm: PrettyMIDI, qdiv: int = QDIV,
        beat_times: np.ndarray | None = None, instruments: list | None = None
) -> Notes:
    """Snap notes to qdiv divisions per beat. Beats come from the MIDI's tempo
    map unless an annotated beat grid is given."""
    if beat_times is None:
        beat_times = pm.get_beats()
    end_time   = pm.get_end_time()

    beat_times = np.append(beat_times, max(end_time, beat_times[-1] + 1e-3))

    def time_to_division(t):
        i = np.searchsorted(beat_times, t, side='right') - 1
        i = min(max(i, 0), len(beat_times) - 2)

        sec_per_beat = beat_times[i+1] - beat_times[i]
        frac = (t - beat_times[i]) / sec_per_beat
        return int(round(i * qdiv + frac * qdiv))

    notes = []
    for inst in (pm.instruments if instruments is None else instruments):
        for n in inst.notes:
            start_div = time_to_division(n.start)
            end_div   = time_to_division(n.end)
            dur_divs  = max(1, end_div - start_div)
            notes.append((start_div, n.pitch, dur_divs))
    return clean_notes(notes)


def in_vocab_range(notes: Notes) -> bool:
    base = notes[0][1]
    return all(MIN_PITCH <= p - base <= MAX_PITCH for _, p, _ in notes)


def notes_to_tokens(notes: Notes) -> list[str]:
    """Encode clean notes as REST/NOTE/PITCH tokens, pitches relative to the first note."""
    base_pitch = notes[0][1]

    tokens = []
    prev_end_div = 0
    for start_div, pitch, dur_divs in notes:
        offset_div = start_div - prev_end_div
        while offset_div > MAX_REST:
            tokens.append(f"<REST_{MAX_REST}>")
            offset_div -= MAX_REST
        if offset_div > 0:
            tokens.append(f"<REST_{offset_div}>")

        tokens.extend([f"<NOTE_{dur_divs}>", f"<PITCH_{pitch - base_pitch:+d}>"])
        prev_end_div = start_div + dur_divs

    return tokens


def tokens_to_notes(tokens: list[str], base_pitch: int) -> Notes:
    """Inverse of notes_to_tokens; ignores anything that is not REST/NOTE/PITCH."""
    notes = []
    time = 0
    dur = None
    for tok in tokens:
        kind, _, val = tok[1:-1].partition("_")
        if kind == "REST":
            time += int(val)
        elif kind == "NOTE":
            dur = int(val)
        elif kind == "PITCH" and dur is not None:
            notes.append((time, base_pitch + int(val), dur))
            time += dur
            dur = None
    return notes


class Tokenizer:
    def __init__(self, tok2id: dict[str, int]):
        self._tok2id = tok2id

    def melody_to_tokens(self, melody: dict[str, Any]) -> list[int]:
        tokens = [0] + [feature_to_token(f, melody[f], self._tok2id) for f in FEATURES]
        return tokens + [self._tok2id[t] for t in notes_to_tokens(melody["notes"])]
