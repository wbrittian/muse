from pretty_midi import PrettyMIDI
from pathlib import Path

from pysrc.data_client.data_client import DataClient
from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.train_model import train_model

class Muse:
    def __init__(self) -> None:
        self.data_client = DataClient()
        self.museformer = None

        self.config_path = Path("data/model/config.json")
        self.model_path = Path("data/model/museformer.pt")

    def _load_model(self) -> None:
        pass

    def _train_model(self) -> None:
        self.data_client.load()
        self.museformer = PytorchModel(self.data_client.vocab_size())
        
        train_model(self.museformer, self.data_client)

    def _to_tokens(self, input_txt: str) -> list[int]:
        pass

    def _generate(self, input_seq: list[int]) -> PrettyMIDI:
        pass

    def _send_to_fl(self, midi: PrettyMIDI) -> None:
        pass

    def load(self) -> None:
        if Path.exists(self.config_path) and Path.exists(self.model_path):
            self._load_model()
        else:
            self._train_model()

    def run(self) -> None:
        while True:
            cmd = input("> ")

            match cmd:
                case "generate" | "g":
                    input_txt = input("input prompt > ")
                    input_seq = self._to_tokens(input_txt)

                    output_midi = self._generate(input_seq)
                    self._send_to_fl(output_midi)
                    

                case "retrain" | "r":
                    self._train_model()

                case "help" | "h":
                    print(
                        "list of commands: \n" +
                        "   [g]enerate - generate a midi sequence from your input prompt\n" +
                        "   [r]etrain - retrain the model from the training data\n" +
                        "   [h]elp - print list of commands\n" +
                        "   [q]uit - quit the program\n"
                    )

                case "quit" | "q":
                    break

                case _:
                    print("please input a valid command")