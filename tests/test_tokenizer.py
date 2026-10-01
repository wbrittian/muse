"""Run from the repo root: poetry run python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pysrc.data_client.tokenizer import clean_notes, notes_to_tokens, tokens_to_notes


class TestRoundTrip(unittest.TestCase):
    def assertRoundTrip(self, notes):
        notes = clean_notes(notes)
        self.assertEqual(tokens_to_notes(notes_to_tokens(notes), notes[0][1]), notes)

    def test_simple(self):
        self.assertRoundTrip([(0, 60, 12), (12, 62, 12), (24, 64, 24)])

    def test_pickup_and_long_rests(self):
        self.assertRoundTrip([(30, 67, 6), (36, 69, 6), (90, 72, 3), (200, 60, 1)])

    def test_long_note_is_capped_not_split(self):
        notes = clean_notes([(0, 60, 100), (200, 62, 12)])
        self.assertEqual(notes, [(0, 60, 48), (200, 62, 12)])
        self.assertNotIn("<REST_", "".join(notes_to_tokens(notes)[:2]))
        self.assertRoundTrip([(0, 60, 100), (200, 62, 12)])

    def test_polyphony_is_reduced(self):
        notes = clean_notes([(0, 60, 24), (0, 64, 24), (12, 67, 12)])
        self.assertEqual(notes, [(0, 64, 12), (12, 67, 12)])

    def test_bimmuda_round_trip(self):
        root = Path("data/bimmuda_dataset")
        if not root.exists():
            self.skipTest("BiMMuDa not present")
        from pretty_midi import PrettyMIDI
        from pysrc.data_client.tokenizer import quantize_notes
        for path in sorted(root.rglob("*.mid"))[::25]:
            if "full" not in path.stem:
                self.assertRoundTrip(quantize_notes(PrettyMIDI(str(path))))


if __name__ == "__main__":
    unittest.main()
