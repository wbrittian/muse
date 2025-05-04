import torch
import torch.nn as nn

class PytorchModel(nn.Module):
    def __init__(
            self,
            V,
            d_model=128,
            h=4,
            N=2,
            dim_ff=512,
            p_drop=.1
    ):
        super().__init__()
        self.token_embed = nn.Embedding(V, d_model)
        self.pos_embed   = nn.Embedding(512, d_model)   # max seq length
        decoder_layer    = nn.TransformerDecoderLayer(d_model, h, dim_ff, p_drop)
        self.transformer = nn.TransformerDecoder(decoder_layer, N)
        self.output_fc   = nn.Linear(d_model, V)

    def forward(self, input_ids):
        T, B = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(input_ids.device)
        x = self.transformer(x, x, tgt_mask=mask)
        return self.output_fc(x)