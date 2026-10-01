from torch.utils.data import Dataset
from torch import LongTensor, tensor, long
from torch.nn.utils.rnn import pad_sequence
from json import load, dump
from pandas import read_csv
from pathlib import Path
from random import Random
from typing import Any

from pysrc.data_client.generate_tokens import generate_tokens
from pysrc.data_client.tokenizer import Tokenizer, in_vocab_range
from pysrc.data_client.collect_features import collect_features
from pysrc.data_client.sources import EXTRA_SOURCES

MAX_SEQ_LEN = 512   # SOS + prefix + melody + EOS
CACHE_VERSION = 2


def load_bimmuda(base_path: Path) -> list[dict[str, Any]]:
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

    melodies = []
    root = Path(base_path / "bimmuda_dataset")
    for midfile in sorted(root.rglob('*.mid')):
        if 'full' not in midfile.stem:
            row = collect_features(midfile, melody_metadata, song_metadata)
            if row is not None:
                melodies.append(row)
    return melodies


class DataClient(Dataset):
    """Tokenized melodies from BiMMuDa plus any extra sources that are prepared on disk.

    Each record is {"source", "song", "tokens"}; tokens are unpadded and end
    with EOS. Use `collate` as the DataLoader collate_fn.
    """

    def __init__(self, sources: list[str] | None = None) -> None:
        # None = BiMMuDa plus every extra source whose data is present
        self.sources = sources
        self.melody_data: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = None
        self.indices: list[int] = None   # the subset this Dataset serves

        self._id2tok: dict = None
        self._tok2id: dict = None

    def _wanted_sources(self, base_path: Path) -> list[str]:
        if self.sources is not None:
            return list(self.sources)
        return ["bimmuda"] + [name for name, src in EXTRA_SOURCES.items() if src.available(base_path)]

    def _load_data(self, base_path: Path, sources: list[str]) -> None:
        for name in sources:
            if name == "bimmuda":
                melodies = load_bimmuda(base_path)
            else:
                src = EXTRA_SOURCES[name]
                if not src.available(base_path):
                    raise FileNotFoundError(f"{name} data missing; run: {src.prep_hint}")
                melodies = src.load(base_path)
            print(f"  {name}: {len(melodies)} melodies")
            self.melody_data.extend(melodies)

    def _load_tokens(self, path: Path) -> None:
        if Path.exists(path):
            # load token data from JSON
            with open(path) as f:
                tokens = load(f)
            tokens = {int(k): v for k, v in tokens.items()}
        else:
            print("generating tokens...")
            tokens = generate_tokens()

        self._id2tok = tokens
        self._tok2id = {v: k for k,v in self._id2tok.items()}

    def _keep(self, melody: dict[str, Any]) -> bool:
        return (
            len(melody["notes"]) >= 4
            and in_vocab_range(melody["notes"])
            and 2 <= melody["BARS"] <= 80
            and f"<TS_{melody['TS']}>" in self._tok2id
        )

    def _get_data(self, path: Path) -> None:
        sources = self._wanted_sources(path)
        # one cache per source set, so runs on different sets do not clobber each other
        cache_path = path / ("tokenized_data.json" if sources == ["bimmuda"] else f"tokenized_{'+'.join(sources)}.json")
        if Path.exists(cache_path):
            with open(cache_path) as f:
                cache = load(f)
            if isinstance(cache, dict) and cache.get("version") == CACHE_VERSION and cache["sources"] == sources:
                self.records = cache["records"]
                return

        self._load_data(path, sources)
        tokenizer = Tokenizer(self._tok2id)
        self.records = []
        dropped = 0
        for melody in self.melody_data:
            if not self._keep(melody):
                dropped += 1
                continue
            tokens = tokenizer.melody_to_tokens(melody) + [1]
            if len(tokens) > MAX_SEQ_LEN:
                dropped += 1
                continue
            self.records.append({"source": melody["source"], "song": melody["song"], "tokens": tokens})
        print(f"tokenized {len(self.records)} melodies ({dropped} dropped)")

        with open(cache_path, "w") as f:
            dump({"version": CACHE_VERSION, "sources": sources, "records": self.records}, f)

    def load_vocab(self) -> None:
        self._load_tokens(Path("model/tokens.json"))

    def load(self) -> None:
        print("loading data...")
        self.load_vocab()
        self._get_data(Path("data/"))
        self.indices = list(range(len(self.records)))

    def split(self, val_frac: float = 0.1, seed: int = 0) -> tuple["DataClient", "DataClient"]:
        """Hold out whole BiMMuDa songs (all their sections) for validation.

        Validation is BiMMuDa only, so scores stay comparable whatever extra
        sources are trained on.
        """
        songs = sorted({r["song"] for r in self.records if r["source"] == "bimmuda"})
        Random(seed).shuffle(songs)
        held_out = set(songs[:round(len(songs) * val_frac)])

        train, val = self._view([]), self._view([])
        for i, r in enumerate(self.records):
            if r["source"] == "bimmuda" and r["song"] in held_out:
                val.indices.append(i)
            else:
                train.indices.append(i)
        return train, val

    def _view(self, indices: list[int]) -> "DataClient":
        view = DataClient(self.sources)
        view.records, view.indices = self.records, indices
        view._id2tok, view._tok2id = self._id2tok, self._tok2id
        return view

    def vocab_size(self) -> int:
        return len(self._id2tok.keys())

    def max_seq_len(self) -> int:
        return MAX_SEQ_LEN

    def get_dict(self, reverse=False) -> dict[str, int]:
        if reverse:
            return self._id2tok
        else:
            return self._tok2id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int)-> tuple[LongTensor, LongTensor]:
        seq = self.records[self.indices[i]]["tokens"]
        inp = tensor(seq[:-1]).to(dtype=long)
        tgt = tensor(seq[1:]).to(dtype=long)
        return inp, tgt

    @staticmethod
    def collate(batch: list[tuple[LongTensor, LongTensor]]) -> tuple[LongTensor, LongTensor]:
        inputs, targets = zip(*batch)
        return (
            pad_sequence(inputs, batch_first=True, padding_value=2),
            pad_sequence(targets, batch_first=True, padding_value=2),
        )
