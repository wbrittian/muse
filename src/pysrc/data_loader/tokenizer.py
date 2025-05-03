from typing import Any
from pretty_midi import PrettyMIDI, note_number_to_name

import numpy as np

class Tokenizer:
    def __init__(self, tok2id: list[str, int], melody_data: list[dict[str, Any]]):
        self._tok2id = tok2id
        self._melody_data = melody_data

    def _feature_to_token(self, key: str, val: Any) -> int:
        raw_token = "<" + key + "_" + val + ">"
        ###
        if raw_token not in self._tok2id:
            print("no token match")
        ###
        return self._tok2id[raw_token]

    # note: validate this is a good format
    def _midi_to_tokens(self, pm: PrettyMIDI) -> list[int]:
        beat_times = pm.get_beats()
        sec_per_div = np.diff(beat_times).mean() / 12

        notes = []
        for inst in pm.instruments:
            for n in inst.notes:
                q_start = int(np.round(n.start / sec_per_div))
                q_end   = int(np.round(n.end   / sec_per_div))
                dur_divs = max(1, q_end - q_start)

                dur_beats = dur_divs / 12

                notes.append((q_start, n.pitch, dur_beats))

        notes.sort(key=lambda x: (x[0], x[1]))

        base_pitch = notes[0][1]
        tokens = []
        prev_end = 0
        seq_end = 0
        seq_dur = 0
        for start, pitch, dur in notes:
            pitch_diff = pitch - base_pitch

            raw_tokens = [
                "<NOTE_" + round(dur, 2) + ">",
                "<PITCH_" + str(pitch_diff) + ">"
            ]

            offset = start - prev_end
            if offset != 0:
                raw_tokens.insert(0, "<REST_" + offset + ">")

            tokens = tokens + raw_tokens
            seq_end = start + dur
            seq_dur += offset + dur
            
        diff = seq_dur - seq_end
        if diff > 0:
            tokens.append("<REST_" + diff + ">")

        converted_tokens = [self._tok2id[tok] for tok in tokens]
        return converted_tokens


    def convert_to_tokens(self) -> list[list[int]]:
        tokenized_data = []

        for melody in self._melody_data:
            tokens = [0]

            for feature, value in melody.items():
                if feature != "midi":
                    tokens.append(self._feature_to_token(feature, value))

            stream = self._midi_to_tokens(melody["midi"])
            tokens = tokens + stream + [1]

            tokenized_data.append(tokens)

        return tokenized_data