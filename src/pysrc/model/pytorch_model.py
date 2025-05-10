import torch
import torch.nn as nn

class PytorchModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            max_seq_len,
            d_model=128,
            num_heads=4,
            num_layers=2,
            dim_ff=512,
            p_drop=.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=p_drop,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.LongTensor):
        tok = self.token_embed(x)
        tok += self.pos_embed[:, :tok.size(1), :]

        enc = self.encoder(tok).transpose(0, 1)
        logits = self.output_proj(enc)
        return logits