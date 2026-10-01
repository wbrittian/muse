import struct
import numpy as np
from pathlib import Path

from pysrc.model.pytorch_model import PytorchModel


def _write_matrix(f, arr: np.ndarray) -> None:
    rows, cols = arr.shape
    f.write(struct.pack('ii', rows, cols))
    f.write(arr.astype(np.float64).tobytes())

def _write_vector(f, arr: np.ndarray) -> None:
    f.write(struct.pack('i', arr.shape[0]))
    f.write(arr.astype(np.float64).tobytes())


def export_weights(model: PytorchModel, path: str) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    num_layers = len(model.encoder.layers)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        _write_matrix(f, sd['token_embed.weight'])
        _write_matrix(f, sd['pos_embed'].squeeze(0))  # (1, max_seq_len, d_model) → (max_seq_len, d_model)

        for i in range(num_layers):
            p = f'encoder.layers.{i}.'
            _write_matrix(f, sd[p + 'self_attn.in_proj_weight'])
            _write_vector(f, sd[p + 'self_attn.in_proj_bias'])
            _write_matrix(f, sd[p + 'self_attn.out_proj.weight'])
            _write_vector(f, sd[p + 'self_attn.out_proj.bias'])
            _write_matrix(f, sd[p + 'linear1.weight'])
            _write_vector(f, sd[p + 'linear1.bias'])
            _write_matrix(f, sd[p + 'linear2.weight'])
            _write_vector(f, sd[p + 'linear2.bias'])
            _write_vector(f, sd[p + 'norm1.weight'])
            _write_vector(f, sd[p + 'norm1.bias'])
            _write_vector(f, sd[p + 'norm2.weight'])
            _write_vector(f, sd[p + 'norm2.bias'])

        _write_matrix(f, sd['output_proj.weight'])
        _write_vector(f, sd['output_proj.bias'])

    print(f"weights exported to {path}")