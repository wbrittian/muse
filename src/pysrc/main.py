from torch.optim import AdamW

from pysrc.data_client.data_client import DataClient
from pysrc.model.pytorch_model import PytorchModel

class Muse:
    def __init__(self):
        self.data_client = DataClient()
        self.museformer = None

    def train_model(self):
        self.data_client.load()
        self.museformer = PytorchModel(self.data_client.vocab_size())
        
        train_model()