from typing import Any
from json import dump

def generate_tokens(melody_data: list[dict[str, Any]]) -> dict[int, str]:
    # collect unique types
    time_signatures = set()
    keys = set()
    eras = set()
    for melody in melody_data:
        ts = melody["TS"]
        if ts not in time_signatures:
            time_signatures.add(ts)

        key = melody["KEY"]
        if key not in keys:
            keys.add(key)

        era = melody["ERA"]
        if era not in eras:
            eras.add(era)
    
    keys = sorted(keys, key=lambda x: x[0])
    eras = sorted(eras, key=lambda x: int(x[:4]))

    # tokens
    id2tok = {}

    id2tok[0] = "<SOS>"
    id2tok[1] = "<EOS>"
    id2tok[2] = "<PAD>"
    i = 3

    # BPM
    for bpm in range(60, 181, 10):
        id2tok[i] = f"<BPM_{bpm}>"
        i += 1
    
    # TS
    for ts in time_signatures:
        id2tok[i] = f"<TS_{ts}>"
        i += 1

    # BARS
    for bars in range(2, 81):
        id2tok[i] = f"<BARS_{bars}>"
        i += 1

    # FIRST, LAST
    for note in range(128):
        id2tok[i] = "<FIRST_" + str(note) + ">"
        id2tok[128+i] = "<LAST_" + str(note) + ">"
        i += 1
    i += 128

    # KEY
    for key in keys:
        id2tok[i] = f"<KEY_{key}>"
        i += 1

    # GENRE
    genres = ["Pop", "Rock", "Funk/Soul", "R&B", "Hip-hop", "Other"]
    for genre in genres:
        id2tok[i] = f"<GENRE_{genre}>"
        i += 1

    # ERA
    for era in eras:
        id2tok[i] = f"<ERA_{era}>"
        i += 1

    # _NOTE
    for note_length in range(1, 49):
        id2tok[i] = f"<NOTE_{note_length}>"
        i += 1
    
    # PITCH
    for pitch in range(-29, 29):
        id2tok[i] = f"<PITCH_{pitch:+d}>"
        i += 1

    # REST
    for beats in range(1, 13):
        id2tok[i] = f"<REST_{beats}>"
        i += 1

    id2tok = dict(sorted(id2tok.items()))
    for idx, tok in id2tok.items():
        print(f"{idx}: {tok}")

    with open("model/tokens.json", "w") as f:
        dump(id2tok, f, indent=4)

    return id2tok