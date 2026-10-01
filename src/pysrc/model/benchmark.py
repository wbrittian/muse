import json
from statistics import median
from time import perf_counter

import torch

from pysrc.test.test_cpp_parity import MODEL_DIR, PREFIX, load_models


def time_runs(fn, runs: int) -> float:
    fn()
    times = []
    for _ in range(runs):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return median(times)


def main(runs: int = 5) -> None:
    torch_model, cpp_model = load_models()
    id2tok = {int(i): t for i, t in json.loads((MODEL_DIR / "tokens.json").read_text()).items()}
    tok2id = {t: i for i, t in id2tok.items()}
    prefix = [tok2id[t] for t in PREFIX]
    max_len = torch_model.pos_embed.shape[1]
    allowed = [i for i, t in id2tok.items() if t.startswith(("<NOTE_", "<PITCH_", "<REST_"))]

    def torch_generate(system: torch.device):
        model = torch_model.to(system)
        mask = torch.full((len(id2tok),), -torch.inf, device=system)
        mask[allowed] = 0.0
        output = list(prefix)
        with torch.no_grad():
            while len(output) < max_len:
                logits = model(torch.LongTensor([output]).to(system))[0, -1] + mask
                vals, idxs = torch.topk(logits, 8)
                output.append(idxs[torch.multinomial(torch.softmax(vals, dim=-1), 1)].item())
        return output

    print(f"one full melody: {max_len - len(prefix)} new tokens, top-k 8, median of {runs}")
    results = {"c++ (float32, kv cache)": time_runs(lambda: cpp_model.generate(prefix, max_len, allowed_tokens=allowed), runs)}
    results["pytorch cpu"] = time_runs(lambda: torch_generate(torch.device("cpu")), runs)
    if torch.backends.mps.is_available():
        results["pytorch mps"] = time_runs(lambda: torch_generate(torch.device("mps")), runs)

    for name, seconds in results.items():
        print(f"  {name:<24} {seconds * 1000:9.1f} ms")


if __name__ == "__main__":
    main()
