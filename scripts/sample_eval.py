"""Generate from the held-out BiMMuDa control prefixes and score the melodies.

    poetry run python scripts/sample_eval.py baseline all_small --wav 6 [--bar_guard]

For each experiment: one melody per held-out prefix, scored on key fit
(share of notes in the best-fitting diatonic scale), distinct pitches, pitch
span, repetition (share of 4-note interval patterns seen earlier in the
melody) and bar count against the requested BARS. The real held-out melodies
are scored the same way as a reference. Writes experiments/<name>/samples/.
"""
import argparse
import json
import sys
import wave
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from pretty_midi import PrettyMIDI, Instrument, Note

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pysrc.data_client.data_client import DataClient
from pysrc.data_client.tokenizer import QDIV, bar_divs, tokens_to_notes
from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.sample import sample_tokens

MAJOR = {0, 2, 4, 5, 7, 9, 11}


def key_fit(pitches: list[int]) -> float:
    return max(mean((p - t) % 12 in MAJOR for p in pitches) for t in range(12))


def repetition(pitches: list[int], n: int = 4) -> float:
    """Share of n-note interval patterns that already occurred earlier in the melody."""
    steps = [b - a for a, b in zip(pitches, pitches[1:])]
    grams = [tuple(steps[i:i + n - 1]) for i in range(len(steps) - n + 2)]
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams) if grams else 0.0


def score(prefix: list[str], notes: list[tuple[int, int, int]]) -> dict[str, float]:
    feats = {t[1:-1].split("_", 1)[0]: t[1:-1].split("_", 1)[1] for t in prefix[1:12]}
    pitches = [p for _, p, _ in notes]
    end = max(s + d for s, _, d in notes)
    bars = int(np.ceil(end / bar_divs(feats["TS"])))
    return {
        "key_fit": key_fit(pitches),
        "distinct": len(set(pitches)),
        "repetition": repetition(pitches),
        "span": max(pitches) - min(pitches),
        "bars_err": abs(bars - int(feats["BARS"])),
        "bars_ok": abs(bars - int(feats["BARS"])) <= 1,
        "notes": len(notes),
    }


def to_midi(notes, bpm: float) -> PrettyMIDI:
    pm = PrettyMIDI(initial_tempo=bpm)
    inst = Instrument(program=0)
    spd = 60.0 / bpm / QDIV
    inst.notes = [Note(velocity=100, pitch=p, start=s * spd, end=(s + d) * spd) for s, p, d in notes]
    pm.instruments.append(inst)
    return pm


def write_wav(pm: PrettyMIDI, path: Path, fs: int = 22050) -> None:
    audio = pm.synthesize(fs=fs)
    audio = (audio / max(1e-9, np.abs(audio).max()) * 0.8 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(audio.tobytes())


def summarize(rows: list[dict]) -> dict[str, float]:
    return {
        "key_fit": round(mean(r["key_fit"] for r in rows), 3),
        "distinct_mean": round(mean(r["distinct"] for r in rows), 2),
        "span_mean": round(mean(r["span"] for r in rows), 2),
        "repetition": round(mean(r["repetition"] for r in rows), 3),
        "bars_within_1": round(mean(r["bars_ok"] for r in rows), 3),
        "bars_err_median": median(r["bars_err"] for r in rows),
        "notes_median": median(r["notes"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--wav", type=int, default=0, help="also render this many samples as MIDI+WAV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--bar_guard", action="store_true", help="make the length follow BARS while sampling")
    args = ap.parse_args()

    system = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data = DataClient(["bimmuda"])
    data.load()
    _, val = data.split(seed=0)
    id2tok = data.get_dict(reverse=True)
    prefixes = [[id2tok[t] for t in data.records[i]["tokens"]] for i in val.indices]

    def bpm_of(prefix):
        return float(prefix[1][5:-1])

    results = {"reference": summarize([
        score(p, tokens_to_notes(p[12:], int(p[4][7:-1]))) for p in prefixes
    ])}

    for name in args.names:
        exp = Path("experiments") / name
        with open(exp / "config.json") as f:
            params = json.load(f)
        model = PytorchModel(data.vocab_size(), data.max_seq_len(), params)
        model.load_state(str(exp / "museformer.pt"), system)
        out = exp / ("samples_guard" if args.bar_guard else "samples")
        out.mkdir(exist_ok=True)

        torch.manual_seed(args.seed)
        rows = []
        for k, prefix in enumerate(prefixes):
            guard = {"bar_divs": bar_divs(prefix[2][4:-1]), "bars": int(prefix[3][6:-1])} if args.bar_guard else {}
            ids = sample_tokens(model, [data.get_dict()[t] for t in prefix[:12]], id2tok, system, data.max_seq_len(),
                               temperature=args.temperature, top_k=args.top_k, **guard)
            toks = [id2tok[i] for i in ids]
            notes = tokens_to_notes(toks[12:], int(toks[4][7:-1]))
            if not notes:
                continue
            rows.append(score(toks, notes))
            if k < args.wav:
                pm = to_midi(notes, bpm_of(prefix))
                stem = f"{k:02d}_" + "_".join(t[1:-1].split("_", 1)[1].replace("/", "-") for t in prefix[1:12])
                pm.write(str(out / f"{stem}.mid"))
                write_wav(pm, out / f"{stem}.wav")
        results[name] = summarize(rows) | {"generated": len(rows)}

    print(json.dumps(results, indent=2))
    for name in args.names:
        with open(Path("experiments") / name / ("samples_guard" if args.bar_guard else "samples") / "scores.json", "w") as f:
            json.dump({"reference": results["reference"], name: results[name]}, f, indent=2)


if __name__ == "__main__":
    main()
