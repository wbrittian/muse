from copy import deepcopy
from torch import device, no_grad
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.export_weights import export_weights
from pysrc.data_client.data_client import DataClient

PREFIX_LEN = 11   # control tokens after SOS; targets from here on are melody/EOS


def evaluate(museformer: PytorchModel, data: DataClient, system: device) -> dict[str, float]:
    """Loss over all non-PAD targets, plus loss and accuracy on melody targets only."""
    museformer.eval()
    criterion = CrossEntropyLoss(ignore_index=2, reduction="sum")
    loader = DataLoader(data, batch_size=64, collate_fn=DataClient.collate)

    tot_loss = tot_n = mel_loss = mel_n = mel_correct = 0.0
    with no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(system), targets.to(system)
            logits = museformer(inputs)
            V = logits.size(-1)

            tot_loss += criterion(logits.reshape(-1, V), targets.reshape(-1)).item()
            tot_n += (targets != 2).sum().item()

            mel_logits, mel_targets = logits[:, PREFIX_LEN:], targets[:, PREFIX_LEN:]
            mel_loss += criterion(mel_logits.reshape(-1, V), mel_targets.reshape(-1)).item()
            mask = mel_targets != 2
            mel_n += mask.sum().item()
            mel_correct += ((mel_logits.argmax(-1) == mel_targets) & mask).sum().item()

    return {"loss": tot_loss / tot_n, "mel_loss": mel_loss / mel_n, "mel_acc": mel_correct / mel_n}


def train_model(
        museformer: PytorchModel,
        data_client: DataClient, system:
        device, save_path: str,
        num_epochs=10,
        val_data: DataClient | None = None,
        patience: int = 10,
        lr: float = 5e-4,
        log=print,
) -> list[dict[str, float]]:
    """Train; with val_data, keep the weights from the best validation epoch
    and stop after `patience` epochs without improvement."""
    museformer.to(system)

    optimizer = AdamW(museformer.parameters(), lr=lr, weight_decay=1e-2)
    criterion = CrossEntropyLoss(ignore_index=2)

    loader = DataLoader(data_client, batch_size=32, shuffle=True, collate_fn=DataClient.collate)

    def train_epoch():
        museformer.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(system), targets.to(system)
            optimizer.zero_grad()
            logits = museformer(inputs)

            B, L, V = logits.shape
            loss = criterion(logits.view(B*L, V), targets.view(B*L))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    history = []
    best, best_state, stale = None, None, 0
    for epoch in range(1, num_epochs+1):
        row = {"epoch": epoch, "train_loss": train_epoch()}
        if val_data is not None:
            row.update({f"val_{k}": v for k, v in evaluate(museformer, val_data, system).items()})
        history.append(row)
        log("epoch {epoch:3d}  ".format(**row) + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))

        if val_data is not None:
            if best is None or row["val_loss"] < best["val_loss"]:
                best, best_state, stale = row, deepcopy(museformer.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is not None:
        museformer.load_state_dict(best_state)
        log("best epoch {epoch}: ".format(**best) + "  ".join(f"{k} {v:.4f}" for k, v in best.items() if k.startswith("val_")))

    museformer.save_state(save_path)
    export_weights(museformer, save_path.replace(".pt", ".bin"))
    return history
