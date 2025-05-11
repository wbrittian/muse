from typing import Any
from pretty_midi import PrettyMIDI, Instrument, Note
from datetime import datetime

def get_input(prompt: str, default: Any, vals: list[Any]) -> Any:
    while True:
        inp = input(prompt)

        if inp in vals:
            return inp
        elif inp == "":
            return default
        else:
            print("incorrect format, please try again")

def tokens_to_events(tokens: list[str]) -> list[tuple[str, Any]]:
    events = []
    for tok in tokens:
        if tok == "<PAD>":
            break

        tok = tok[1:-1]
        event = tok.split("_")
        events.append((event[0], int(event[1])))

    return events

def tokens_to_midi(tokens: list[str], tempo: float, base_pitch: int) -> PrettyMIDI:
    pm = PrettyMIDI(initial_tempo=tempo)
    inst = Instrument(program=0)

    events = tokens_to_events(tokens)
    time = 0.0
    sec_per_beat = 60.0 / tempo

    next_dur = None
    result = []
    for event, outcome in events:
        if event == "REST":
            time += outcome * sec_per_beat
        elif event == "NOTE":
            next_dur = outcome * sec_per_beat
        elif event == "PITCH":
            if next_dur is not None:
                pitch = base_pitch + outcome
                start = time
                end = time + next_dur

                result.append((pitch, start, end))

                time += next_dur
                next_dur = None
            else:
                print(f"no note length for note {outcome}")

    for pitch, start, end in result:
        inst.notes.append(Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end
        ))

    pm.instruments.append(inst)
    pm.remove_invalid_notes()

    time = datetime.now().strftime("%m_%d-%H_$M")
    pm.write(f"output/muse_{time}.mid")

    return pm