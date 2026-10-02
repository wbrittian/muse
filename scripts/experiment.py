"""Train on a set of sources and score on held-out BiMMuDa songs.

    poetry run python scripts/experiment.py --name baseline --sources bimmuda
    poetry run python scripts/experiment.py --name pop909 --sources bimmuda pop909 --d_model 256 --num_layers 4

Writes experiments/<name>/{museformer.pt,museformer.bin,config.json,history.json}.
Run from the repo root.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pysrc.data_client.data_client import DataClient
from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.train_model import train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sources", nargs="+", default=["bimmuda"])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--p_drop", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="experiment name to start from (fine-tuning); its model shape is reused")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    system = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    data = DataClient(args.sources)
    data.load()
    train, val = data.split(seed=0)   # fixed split: every run scores the same songs
    print(f"train {len(train)}  val {len(val)}  device {system}")

    params = {k: getattr(args, k) for k in ["d_model", "num_heads", "num_layers", "dim_ff", "p_drop", "num_epochs"]}
    if args.init:
        with open(Path("experiments") / args.init / "config.json") as f:
            init = json.load(f)
        params.update({k: init[k] for k in ["d_model", "num_heads", "num_layers", "dim_ff"]})
        params["init"] = args.init
    params["sources"] = args.sources
    model = PytorchModel(data.vocab_size(), data.max_seq_len(), params)
    if args.init:
        model.load_state(str(Path("experiments") / args.init / "museformer.pt"), system)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    out = Path("experiments") / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(params, f, indent=2)

    history = train_model(model, train, system, str(out / "museformer.pt"), args.num_epochs,
                          val_data=val, patience=args.patience, lr=args.lr)
    with open(out / "history.json", "w") as f:
        json.dump({"train_size": len(train), "val_size": len(val), "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
