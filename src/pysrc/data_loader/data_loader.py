from json import load
from pandas import read_csv
from pathlib import Path
from typing import Any

from pysrc.data_loader.generate_tokens import generate_tokens
from pysrc.data_loader.tokenizer import Tokenizer
from pysrc.data_loader.collect_features import collect_features

class DataLoader:
    def __init__(self) -> None:
        self.melody_data: list[dict[str, Any]] = []
        self.tokenized_data: list[list[int]] = None

        self._id2tok: dict = None
        self._tok2id: dict = None

    def _load_data(self, base_path: Path) -> None:
        melody_raw = read_csv(base_path / "metadata/bimmuda_per_melody_metadata.csv")
        song_raw = read_csv(base_path / "metadata/bimmuda_per_song_metadata.csv")

        melody_metadata = melody_raw.set_index(melody_raw.columns[0]).to_dict(orient="index")

        song_raw["id"] = song_raw["Year"].astype(str) + "_0" + song_raw["Position"].astype(str)
        song_metadata = (
            song_raw
            .groupby("id")
            .apply(lambda g: g.to_dict(orient="records"))
            .to_dict()
        )

        root = Path(base_path + "bimmuda_dataset")
        for midfile in root.rglob('*.mid'):
            if 'full' not in midfile.stem:
                row = collect_features(midfile, melody_metadata, song_metadata)
                if row is not None:
                    self.melody_data.append(row)

        

    def _load_tokens(self, path: Path) -> None:
        if Path.exists(path):
            # load token data from JSON
            with open(path) as f:
                tokens = load(f)
        else:
            self._load_data("data/")
            tokens = generate_tokens(self.melody_data)

        self._id2tok = tokens
        self._tok2id = {v: k for k,v in self._id2tok.items()}


    def _get_data(self, path: Path) -> None:
        if Path.exists(path / "processed_data.json"):
            with open(path / "processed_data.json") as f:
                self.tokenized_data = load(f)
        else:
            if self.melody_data is None:
                self._load_data(path)
            tokenizer = Tokenizer(self._tok2id, self.melody_data)
            self.tokenized_data = tokenizer.convert_to_tokens()
            #print(self.tokenized_data)


    def load(self) -> None:
        self._load_tokens(Path("model/tokens.json"))
        self._get_data(Path("data/"))