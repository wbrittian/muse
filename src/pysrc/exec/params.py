from typing import Any
from json import load, dump

def get_params(config_path: str) -> dict[str, Any]:
    print("please input params below")
    print("[enter] to accept default value")

    d_model = input("d_model (128) > ")
    num_heads = input("num_heads (4) > ")
    num_layers = input("num_layers (2) > ")
    dim_ff = input("dim_ff (512) > ")
    p_drop = input("p_drop (.1) > ")

    if d_model == "" or d_model == " ":
        d_model = 128
    else:
        d_model = int(d_model)

    if num_heads == "" or num_heads == " ":
        num_heads = 4
    else:
        num_heads = int(num_heads)

    if num_layers == "" or num_layers == " ":
        num_layers = 2
    else:
        num_layers = int(num_layers)

    if dim_ff == "" or dim_ff == " ":
        dim_ff = 512
    else:
        dim_ff = int(dim_ff)

    if p_drop == "" or p_drop == " ":
        p_drop = .1
    else:
        p_drop = float(p_drop)
    
    params = {
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dim_ff": dim_ff,
        "p_drop": p_drop
    }

    with open(config_path, "w") as f:
        dump(params, f, indent=2)

    return params

def load_params(config_path) -> dict[str, Any]:
    with open(config_path) as f:
        params = load(f)

    params["d_model"] = int(params["d_model"])
    params["num_heads"] = int(params["num_heads"])
    params["num_layers"] = int(params["num_layers"])
    params["dim_ff"] = int(params["dim_ff"])
    params["p_drop"] = float(params["p_drop"])

    return params