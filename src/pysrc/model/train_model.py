from torch import device
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pysrc.model.pytorch_model import PytorchModel
from pysrc.data_client.data_client import DataClient

def train_model(museformer: PytorchModel, data_client: DataClient):
    cpu = device("cpu")
    museformer.to(cpu)

    optimizer = AdamW(museformer.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = CrossEntropyLoss(ignore_index=0)

    loader = DataLoader(data_client, batch_size=32, shuffle=True)

    def train_epoch():
        museformer.train()