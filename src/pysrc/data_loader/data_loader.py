from os import path
from json import load

from pysrc.data_loader.generate_tokens import generate_tokens
from pysrc.data_loader.melody import Melody

import pandas as pd

class DataLoader:
    def __init__(self) -> None:
        self.raw_data: list[Melody] = []

        self._id2tok: dict = {}
        self._tok2id: dict = {}

    def load_data(self) -> None:
        pass

    def load_tokens(self, path: str) -> None:
        if path.exists(path):
            # load token data from JSON
            with open(path) as f:
                self._id2tok = load(f)
            self._tok2id = {v:k for k,v in self._id2tok.items()}
        else:
            # generate from loaded data
            tokens = generate_tokens()

    def tokenize(self) -> None:
        pass