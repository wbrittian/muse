from torch.optim import AdamW

from pysrc.data_client.data_client import DataClient
from pysrc.pytorch_model import PytorchModel

class Muse:
    tokenfile_path = "../../../data/processed_data/tokens.json"

data_client = DataClient()
data_client.load()

V = data_client.V()
model = PytorchModel(V)

# optimizer = AdamW()