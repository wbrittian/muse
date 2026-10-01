from pathlib import Path
from typing import Any
from pretty_midi import PrettyMIDI, key_number_to_key_name
from typing import Optional

from pysrc.data_client.tokenizer import Notes, quantize_notes


def bpm_bin(bpm: float) -> str:
    if bpm < 70:
        return "60"
    elif bpm < 180:
        return str(int((bpm // 10) * 10))
    else:
        return "180"


def melody_features(
        notes: Notes, bpm: float, ts: str, num_bars: int,
        mode: str, genre: str, era: str
) -> dict[str, Any]:
    """The 11 control features for one clean, monophonic melody.

    genre/era may be "Unknown" for sources that do not label them.
    """
    pitches = [p for _, p, _ in notes]
    first_note = pitches[0]
    last_note  = pitches[-1]

    mid_third = pitches[len(pitches)//3 : 2*len(pitches)//3]
    max_mid = max(mid_third) if mid_third else first_note
    min_mid = min(mid_third) if mid_third else first_note
    if max_mid > max(first_note, last_note) + 3:
        contour = "arch"
    elif min_mid < min(first_note, last_note) - 3:
        contour = "valley"
    elif last_note > first_note + 3:
        contour = "ascending"
    elif last_note < first_note - 3:
        contour = "descending"
    else:
        contour = "arch"

    notes_per_bar = len(notes) / int(num_bars)
    if notes_per_bar < 4:
        density = "sparse"
    elif notes_per_bar < 8:
        density = "moderate"
    else:
        density = "dense"

    pitch_span = max(pitches) - min(pitches)
    if pitch_span < 12:
        note_range = "narrow"
    elif pitch_span < 24:
        note_range = "moderate"
    else:
        note_range = "wide"

    return {
        "BPM": bpm_bin(bpm),
        "TS": ts,
        "BARS": int(num_bars),
        "FIRST": first_note,
        "LAST": last_note,
        "MODE": mode,
        "GENRE": genre,
        "ERA": era,
        "CONTOUR": contour,
        "DENSITY": density,
        "RANGE": note_range,
    }


def collect_features(
        path: Path,
        melody_metadata: dict[str, list[Any]],
        song_metadata: dict[str, list[Any]],
) -> Optional[dict[str, Any]]:
    """Features and notes for one BiMMuDa melody file."""
    key = path.stem
    midi_data = PrettyMIDI(str(path))

    if key.endswith("misc"):
        return None

    bpm = midi_data.get_tempo_changes()[1][0]

    if key in melody_metadata:
        melody = melody_metadata[key]

        ts = melody["Time Signature"]
        num_bars = melody["Number of Bars"]
    else:
        signature_changes = midi_data.time_signature_changes
        num = signature_changes[0].numerator
        denom = signature_changes[0].denominator
        ts = str(num) + "/" + str(denom)

        num_bars = len(midi_data.get_downbeats())

    key = key[:-2]
    song = song_metadata[key][0]

    key_changes = midi_data.key_signature_changes
    key_name = key_number_to_key_name(key_changes[0].key_number)
    mode = "minor" if "minor" in key_name else "major"

    if "Genre (Broad 1)" in song:
        genre = song["Genre (Broad 1)"]
    else:
        genre = "Other"

    genre = genre.rstrip()
    if genre in {"Country", "Folk", "EDM/Dance", "Jazz", "Reggae", "Latin"}:
        genre = "Other"

    era = str((int(key[:4]) // 10) * 10) + "s"

    notes = quantize_notes(midi_data)
    row = melody_features(notes, bpm, ts, num_bars, mode, genre, era)
    row.update({"source": "bimmuda", "song": key, "notes": notes})
    return row
