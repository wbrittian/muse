import torch
import torch.nn as nn
from torch import device
from typing import Any

class PytorchModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            params: dict[str, Any]
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, params["d_model"])
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_seq_len, params["d_model"]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params["d_model"],
            nhead=params["num_heads"],
            dim_feedforward=params["dim_ff"],
            dropout=params["p_drop"],
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=params["num_layers"])
        self.output_proj = nn.Linear(params["d_model"], vocab_size)

    def forward(self, x: torch.LongTensor):
        tok = self.token_embed(x)
        tok += self.pos_embed[:, :tok.size(1), :]

        enc = self.encoder(tok).transpose(0, 1)
        logits = self.output_proj(enc)
        return logits
    
    def save_state(self, save_path: str):
        torch.save(self.state_dict(), save_path)
        print("model saved")

    def load_state(self, model_path: str, system: device):
        self.load_state_dict(torch.load(model_path, map_location=system))
        self.to(system)