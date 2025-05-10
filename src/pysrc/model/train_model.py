from torch import device
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.cuda as cuda

from pysrc.model.pytorch_model import PytorchModel
from pysrc.data_client.data_client import DataClient

def train_model(
        museformer: PytorchModel, 
        data_client: DataClient, system: 
        device, save_path: str, 
        num_epochs=10
):
    museformer.to(system)

    optimizer = AdamW(museformer.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = CrossEntropyLoss(ignore_index=0)

    loader = DataLoader(data_client, batch_size=32, shuffle=True)

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
    
    for epoch in range(1, num_epochs+1):
        avg_loss = train_epoch()
        print(f"loss for epoch {epoch:2d}: {avg_loss:.4f}")

    museformer.save_state(save_path)