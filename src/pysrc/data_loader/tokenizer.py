from typing import Any
from pretty_midi import PrettyMIDI

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

    def _midi_to_tokens(self, pm: PrettyMIDI) -> list[int]:
        pass

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