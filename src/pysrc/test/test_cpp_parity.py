import json
import unittest
from pathlib import Path

import numpy as np
import torch

from pysrc.exec.params import load_params
from pysrc.model.pytorch_model import PytorchModel
from pysrc.museformer import Museformer

MODEL_DIR = Path("model")
PREFIX = [
    "<SOS>", "<BPM_120>", "<TS_4/4>", "<BARS_16>", "<FIRST_60>", "<LAST_60>", "<MODE_major>",
    "<GENRE_Pop>", "<ERA_2000s>", "<CONTOUR_arch>", "<DENSITY_moderate>", "<RANGE_moderate>",
]


def load_models() -> tuple[PytorchModel, Museformer]:
    params = load_params(str(MODEL_DIR / "config.json"))
    vocab_size = len(json.loads((MODEL_DIR / "tokens.json").read_text()))
    state = torch.load(MODEL_DIR / "museformer.pt", map_location="cpu")
    max_seq_len = state["pos_embed"].shape[1]

    torch_model = PytorchModel(vocab_size, max_seq_len, params)
    torch_model.load_state_dict(state)
    torch_model.eval()

    cpp_model = Museformer(
        vocab_size, max_seq_len, params["d_model"], params["num_heads"], params["num_layers"], params["dim_ff"]
    )
    cpp_model.load(str(MODEL_DIR / "museformer.bin"))
    return torch_model, cpp_model


class CppParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.torch_model, cls.cpp_model = load_models()
        cls.vocab_size = cls.torch_model.output_proj.out_features
        cls.max_seq_len = cls.torch_model.pos_embed.shape[1]
        cls.id2tok = {int(i): tok for i, tok in json.loads((MODEL_DIR / "tokens.json").read_text()).items()}
        tok2id = {tok: i for i, tok in cls.id2tok.items()}
        cls.prefix = [tok2id[tok] for tok in PREFIX]

    def torch_logits(self, tokens: list[int]) -> np.ndarray:
        with torch.no_grad():
            return self.torch_model(torch.LongTensor([tokens]))[0].numpy()

    def assert_logits_match(self, tokens: list[int]) -> None:
        expected = self.torch_logits(tokens)
        actual = self.cpp_model.forward(tokens)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_random_sequences(self) -> None:
        rng = np.random.default_rng(0)
        for length in (1, 2, 12, 100, self.max_seq_len):
            self.assert_logits_match([0] + rng.integers(0, self.vocab_size, length - 1).tolist())

    def test_sampled_melody(self) -> None:
        allowed = [i for i, tok in self.id2tok.items() if tok.startswith(("<NOTE_", "<PITCH_", "<REST_"))]
        melody = self.cpp_model.generate(self.prefix, self.max_seq_len, allowed_tokens=allowed, seed=0)
        self.assertEqual(len(melody), self.max_seq_len)
        self.assert_logits_match(melody)

    def test_greedy_generation_matches(self) -> None:
        output = list(self.prefix)
        with torch.no_grad():
            while len(output) < self.max_seq_len:
                next_id = int(self.torch_model(torch.LongTensor([output]))[0, -1].argmax())
                if next_id == 1:
                    break
                output.append(next_id)

        self.assertEqual(self.cpp_model.generate(self.prefix, self.max_seq_len, top_k=1), output)


if __name__ == "__main__":
    unittest.main()
