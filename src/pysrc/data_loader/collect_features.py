from pathlib import Path
from typing import Any
from pretty_midi import PrettyMIDI, key_number_to_key_name
from typing import Optional

def collect_features(
        path: Path, 
        melody_metadata: dict[str, list[Any]], 
        song_metadata: dict[str, list[Any]],
) -> Optional[list[str, Any]]:
    key = path.stem
    midi_data = PrettyMIDI(str(path))

    if key.endswith("misc"):
        return None

    if key in melody_metadata:
        melody = melody_metadata[key]

        ts = melody["Time Signature"]
        num_bars = melody["Number of Bars"]
    else:
        signature_changes = midi_data.time_signature_changes
        num = signature_changes[0].numerator
        denom = signature_changes[0].denominator
        ts = str(num) + "/" + str(denom)

        num_bars = len(midi_data.get_beats())

    key = key[:-2]
    song = song_metadata[key][0]

    all_notes = [n for inst in midi_data.instruments for n in inst.notes]
    all_notes.sort(key=lambda n: n.start)

    first_note = all_notes[0].pitch
    last_note  = all_notes[-1].pitch

    key_changes = midi_data.key_signature_changes
    song_key = key_number_to_key_name(key_changes[0].key_number)

    if "Genre (Broad 1)" in song:
        genre = song["Genre (Broad 1)"]
    else:
        genre = "Other"

    if genre in {"Country", "Folk", "EDM/Dance", "Jazz", "Reggae", "Latin"}:
        genre = "Other"

    era = str((int(key[:4]) // 10) * 10) + "s"

    return {
        "BPM": midi_data.estimate_tempo(),
        "TS": ts,
        "BARS": num_bars,
        "FIRST": first_note,
        "LAST": last_note,
        "KEY": song_key,
        "GENRE": genre,
        "ERA": era,
        "midi": midi_data
    }