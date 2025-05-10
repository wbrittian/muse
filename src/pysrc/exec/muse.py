from typing import Any
from pretty_midi import PrettyMIDI
from pathlib import Path
from torch import device
import torch.cuda as cuda

from pysrc.data_client.data_client import DataClient
from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.train_model import train_model
from pysrc.exec.params import get_params, load_params

class Muse:
    def __init__(self) -> None:
        self.data_client = DataClient()
        self.museformer = None

        self.config_path = Path("model/config.json")
        self.model_path = Path("model/museformer.pt")

        self.system = device("cuda" if cuda.is_available() else "cpu")

    def _load_model(self, params: dict[str, Any]) -> None:
        self.museformer = PytorchModel(
            self.data_client.vocab_size(), 
            self.data_client.max_seq_len(),
            params
        )
        self.museformer.load_state(str(self.model_path), self.system)
        print("model loaded")

    def _train_model(self, params: dict[str, Any]) -> None:
        print("loading data...")
        self.data_client.load()
        self.museformer = PytorchModel(
            self.data_client.vocab_size(), 
            self.data_client.max_seq_len(),
            params
        )
        
        print("training model...")
        train_model(self.museformer, self.data_client, self.system, str(self.model_path))
        print("model loaded")

    def _to_tokens(self, input_txt: str) -> list[int]:
        pass

    def _generate(self, input_seq: list[int]) -> PrettyMIDI:
        pass

    def _send_to_fl(self, midi: PrettyMIDI) -> None:
        pass

    def run(self) -> None:
        self.data_client.load()
        if Path.exists(self.config_path):
            params = load_params(str(self.config_path))
            self._load_model(params)

        while True:
            cmd = input("> ")

            match cmd:
                case "generate" | "g":
                    input_txt = input("input prompt > ")
                    input_seq = self._to_tokens(input_txt)

                    output_midi = self._generate(input_seq)
                    self._send_to_fl(output_midi)
                    

                case "configure" | "c":
                    params = get_params(str(self.config_path))
                    self._train_model(params)

                case "retrain" | "r":
                    params = load_params(str(self.config_path))
                    self._train_model(params)

                case "info" | "i":
                    print(
                        "list of commands: \n" +
                        "   [g]enerate - generate a midi sequence from your input prompt\n" +
                        "   [c]onfigure - configure model hyperparams and train model\n" +
                        "   [r]etrain - retrain model based on given params\n" +
                        "   [i]nfo - print list of commands\n" + 
                        "   [h]elp - get help on using muse\n" +
                        "   [q]uit - quit the program\n"
                    )
                
                case "help" | "i":
                    print("help page coming soon")

                case "quit" | "q":
                    break

                case _:
                    print("please input a valid command")