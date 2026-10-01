from json import dump

# The vocabulary is static so token ids do not depend on which melodies were
# loaded. Ids 0-419 are the original BiMMuDa vocabulary; tokens added later go
# at the END so existing ids never move.
TIME_SIGNATURES = ["12/8", "3/4", "4/4", "6/8", "9/8"]
ERAS = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
GENRES = ["Pop", "Rock", "Funk/Soul", "R&B", "Hip-hop", "Other"]

# labels for sources that do not annotate genre or era
APPENDED = ["<GENRE_Unknown>", "<ERA_Unknown>"]


def generate_tokens() -> dict[int, str]:
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
    for ts in TIME_SIGNATURES:
        id2tok[i] = f"<TS_{ts}>"
        i += 1

    # BARS
    for bars in range(2, 81):
        id2tok[i] = f"<BARS_{bars}>"
        i += 1

    # FIRST, LAST
    for note in range(21, 109):
        id2tok[i] = "<FIRST_" + str(note) + ">"
        id2tok[88+i] = "<LAST_" + str(note) + ">"
        i += 1
    i += 88

    # MODE
    for mode in ["major", "minor"]:
        id2tok[i] = f"<MODE_{mode}>"
        i += 1

    # GENRE
    for genre in GENRES:
        id2tok[i] = f"<GENRE_{genre}>"
        i += 1

    # ERA
    for era in ERAS:
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

    # CONTOUR
    for contour in ["ascending", "descending", "arch", "valley"]:
        id2tok[i] = f"<CONTOUR_{contour}>"
        i += 1

    # DENSITY
    for density in ["sparse", "moderate", "dense"]:
        id2tok[i] = f"<DENSITY_{density}>"
        i += 1

    # RANGE
    for note_range in ["narrow", "moderate", "wide"]:
        id2tok[i] = f"<RANGE_{note_range}>"
        i += 1

    for tok in APPENDED:
        id2tok[i] = tok
        i += 1

    id2tok = dict(sorted(id2tok.items()))

    with open("model/tokens.json", "w") as f:
        dump(id2tok, f, indent=4)

    return id2tok
