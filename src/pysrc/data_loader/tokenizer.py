from typing import Any
from pretty_midi import PrettyMIDI, note_number_to_name

import numpy as np

class Tokenizer:
    def __init__(self, tok2id: list[str, int], melody_data: list[dict[str, Any]]):
        self._tok2id = tok2id
        self._melody_data = melody_data

    def _feature_to_token(self, key: str, val: Any) -> int:
        raw_token = f"<{key}_{val}>"
        ###
        if raw_token not in self._tok2id:
            print("no token match")
        ###
        return self._tok2id[raw_token]

    def _quantize_notes(self, pm: PrettyMIDI, qdiv: int):
        beat_times = pm.get_beats() 
        end_time   = pm.get_end_time()

        beat_times = np.append(beat_times, end_time)

        def time_to_division(t):
            i = np.searchsorted(beat_times, t, side='right') - 1
            i = min(max(i, 0), len(beat_times) - 2)

            sec_per_beat = beat_times[i+1] - beat_times[i]
            frac = (t - beat_times[i]) / sec_per_beat
            return int(round(i * qdiv + frac * qdiv))

        notes = []
        for inst in pm.instruments:
            for n in inst.notes:
                start_div = time_to_division(n.start)
                end_div   = time_to_division(n.end)
                dur_divs  = max(1, end_div - start_div)
                notes.append((start_div, n.pitch, dur_divs))
        return sorted(notes, key=lambda x: (x[0], x[1]))

    def _midi_to_tokens(self, pm: PrettyMIDI) -> list[int]:
        qdiv = 12
        notes = self._quantize_notes(pm, qdiv)
        base_pitch = notes[0][1]

        tokens = []
        prev_end_div  = 0
        for start_div, pitch, dur_divs in notes:
            raw_tokens = []

            pitch_diff = pitch - base_pitch
            while dur_divs > 48:
                raw_tokens.extend(["<NOTE_48>", f"<PITCH_{pitch_diff:+d}>"])
                dur_divs -= 48
            raw_tokens.extend([f"<NOTE_{dur_divs}>", f"<PITCH_{pitch_diff:+d}>"])
            
            # compute rest in divisions
            offset_div = start_div - prev_end_div
            rest_tokens = []
            if offset_div > 0:
                while offset_div > 12:
                    rest_tokens.append("<REST_12>")
                    offset_div -= 12
                rest_tokens.append(f"<REST_{offset_div}>")
            raw_tokens = rest_tokens + raw_tokens

            tokens.extend(raw_tokens)
            prev_end_div = start_div + dur_divs

        return [self._tok2id[t] for t in tokens]


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