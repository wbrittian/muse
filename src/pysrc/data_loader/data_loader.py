from os import path
from json import load
from pandas import read_csv
from pathlib import Path

from pysrc.data_loader.generate_tokens import generate_tokens
from pysrc.data_loader.generate_melody import generate_melody
from pysrc.data_loader.melody import Melody

class DataLoader:
    def __init__(self) -> None:
        self.melody_names = []
        self.melody_data: list[Melody] = []
        self.token_data: list[list[int]] = []

        self._id2tok: dict = {}
        self._tok2id: dict = {}

    def _load_data(self, base_path: str) -> None:
        melody_raw = read_csv(base_path + "metadata/bimmuda_per_melody_metadata.csv")
        song_raw = read_csv(base_path + "metadata/bimmuda_per_song_metadata.csv")

        melody_metadata = melody_raw.set_index(melody_raw.columns[0]).to_dict(orient="index")
        song_metadata = song_raw.set_index(song_raw.columns[0]).to_dict(orient="index")

        root = Path(base_path + "bimmuda_dataset")
        for midfile in root.rglob('*.mid'):
            if 'full' not in midfile.stem:
                self.melody_data.append(generate_melody(midfile, melody_metadata, song_metadata))
        

    def _load_tokens(self, path: str) -> None:
        if path.exists(path):
            # load token data from JSON
            with open(path) as f:
                self._id2tok = load(f)
            self._tok2id = {v:k for k,v in self._id2tok.items()}
        else:
            # generate from loaded data
            tokens = generate_tokens()

    def _tokenize(self) -> None:
        pass