import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "True"

from typing import Any
from time import time, sleep
from pretty_midi import PrettyMIDI
from pathlib import Path
from torch import device, LongTensor, no_grad, multinomial, softmax, topk, full, inf
import torch.cuda as cuda
import torch.backends.mps as mps
import pygame.midi as midi

from pysrc.data_client.data_client import DataClient
from pysrc.data_client.tokenizer import feature_to_token
from pysrc.model.pytorch_model import PytorchModel
from pysrc.model.train_model import train_model
from pysrc.exec.params import get_params, load_params
from pysrc.exec.prompt import parse_prompt
from pysrc.exec.utils import get_input, tokens_to_midi

class Muse:
    def __init__(self) -> None:
        self.data_client = DataClient()
        self.museformer = None

        self.config_path = Path("model/config.json")
        self.model_path = Path("model/museformer.pt")

        self.system = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")

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
        train_model(self.museformer, self.data_client, self.system, str(self.model_path), params["num_epochs"])
        print("model loaded")

    def _get_input_tokens(self) -> tuple[list[int], float]:
        print(
            "please input parameters or accept defaults\n\n" +
            "options:\n" +
            "bpm: 60-180" +
            "ts: 4/4, 3/4, 6/8, 12/8, 9/8\n" +
            "bars: 2-80\n" +
            "first: 21-108\n" +
            "last: 21-108\n" +
            "key: Ab Major-G# minor\n" +
            "genre: [P]op, [R]ock, [F]unk/Soul, R&[B], [H]ip-hop, [O]ther\n" +
            "era: 1950s-2020s"
        )

        seq = []

        bpm = input("bpm (120) > ")
        if bpm == "":
            bpm = 120
            seq.append("120")
        else:
            bpm = int(bpm)
            if bpm <= 60:
                seq.append("60")
            elif bpm >= 180:
                seq.append("180")
            else:
                seq.append(str((bpm // 10) * 10))
        bpm = float(bpm)
            
        seq.append(get_input("ts (4/4) > ", "4/4", ["4/4", "3/4", "6/8", "12/8", "9/8"]))
        seq.append(get_input("bars (16) > ", "16", [str(i) for i in range(2, 81)]))
        seq.append(get_input("first (60) > ", "60", [str(i) for i in range(21, 109)]))
        seq.append(get_input("last (60) > ", "60", [str(i) for i in range(21, 109)]))
        seq.append(get_input("mode (major) > ", "major", ["major", "minor"]))
        seq.append(get_input("genre (P) > ", "Pop", [
            "P", "R", "F", "B", "H", "O",
            "Pop", "Rock", "Funk/Soul", "R&B", "Hip-hop", "Other"
        ]))
        seq.append(get_input("era (2000s) > ", "2000s", [
            "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"
        ]))
        seq.append(get_input("contour (arch) > ", "arch", ["ascending", "descending", "arch", "valley"]))
        seq.append(get_input("density (moderate) > ", "moderate", ["sparse", "moderate", "dense"]))
        seq.append(get_input("range (moderate) > ", "moderate", ["narrow", "moderate", "wide"]))

        match seq[6]:
            case "P":
                seq[6] = "Pop"
            case "R":
                seq[6] = "Rock"
            case "F":
                seq[6] = "Funk/Soul"
            case "B":
                seq[6] = "R&B"
            case "H":
                seq[6] = "Hip-hop"
            case "O":
                seq[6] = "Other"

        features = ["BPM", "TS", "BARS", "FIRST", "LAST", "MODE", "GENRE", "ERA", "CONTOUR", "DENSITY", "RANGE"]
        tokens = []
        tok2id = self.data_client.get_dict()
        for  key, val in zip(features, seq):
            tokens.append(feature_to_token(key, val, tok2id))

        return ([0] + tokens, bpm)

    def _prompt_to_tokens(self) -> tuple[list[int], float]:
        text = input("describe your melody > ")
        print("thinking...")
        params = parse_prompt(text)

        print(
            f"\n  bpm: {params['bpm']}  ts: {params['ts']}  bars: {params['bars']}\n"
            f"  mode: {params['mode']}  genre: {params['genre']}  era: {params['era']}\n"
            f"  contour: {params['contour']}  density: {params['density']}  range: {params['range']}\n"
        )

        tok2id = self.data_client.get_dict()
        features = ["BPM", "TS", "BARS", "FIRST", "LAST", "MODE", "GENRE", "ERA", "CONTOUR", "DENSITY", "RANGE"]
        values = [
            str(params["bpm"]), params["ts"], str(params["bars"]),
            str(params["first"]), str(params["last"]),
            params["mode"], params["genre"], params["era"],
            params["contour"], params["density"], params["range"]
        ]

        tokens = [feature_to_token(k, v, tok2id) for k, v in zip(features, values)]
        return ([0] + tokens, float(params["bpm"]))

    def _generate(
            self, input_seq: list[int], max_tokens: int, bpm: float,
            temperature: float = 1.0, top_k: int = 8
    ) -> PrettyMIDI:
        self.museformer.eval()
        id2tok = self.data_client.get_dict(reverse=True)

        # only melody tokens (and EOS) may follow the control prefix
        allowed = full((self.data_client.vocab_size(),), -inf, device=self.system)
        for i, tok in id2tok.items():
            if tok == "<EOS>" or tok.startswith(("<NOTE_", "<PITCH_", "<REST_")):
                allowed[i] = 0.0

        output = list(input_seq)
        with no_grad():
            while len(output) < max_tokens:
                cur = LongTensor([output]).to(self.system)
                logits = self.museformer(cur)[0, -1] / temperature + allowed

                vals, idxs = topk(logits, top_k)
                next_id = idxs[multinomial(softmax(vals, dim=-1), 1)].item()
                if next_id == 1:
                    break
                output.append(next_id)

        output = [id2tok[i] for i in output]
        return tokens_to_midi(output[12:], bpm, int(output[4][7:-1]))


    def _send_to_fl(self, pm: PrettyMIDI) -> None:
        midi.init()
        port = midi.get_default_output_id()
        for pid in range(midi.get_count()):
            _, name, _, is_output, _ = midi.get_device_info(pid)
            if is_output and name.decode() == "IAC Driver MUSE":
                port = pid

        if port == -1:
            print("no MIDI output found, skipping FL Studio")
            midi.quit()
            return
        out = midi.Output(port, latency=0)

        events = []
        for inst in pm.instruments:
            channel = inst.program
            for note in inst.notes:
                on_status = 0x90 | (channel & 0x0F)
                events.append((note.start, on_status, note.pitch, note.velocity))
                off_status = 0x80 | (channel & 0x0F)
                events.append((note.end, off_status, note.pitch, 0))

        print("streaming to FL Studio...")
        events.sort(key=lambda e: e[0])
        t0 = time()
        for event_time, status, data1, data2 in events:
            wait = event_time - (time() - t0)
            if wait > 0:
                sleep(wait)

            out.write_short(status, data1, data2)

        out.close()
        midi.quit()

    def run(self) -> None:
        self.data_client.load()
        if Path.exists(self.config_path):
            params = load_params(str(self.config_path))
            if Path.exists(self.model_path):
                self._load_model(params)
            else:
                self._train_model(params)

        while True:
            cmd = input("> ")

            match cmd:
                case "generate" | "g":
                    input_seq, bpm = self._prompt_to_tokens()
                    output_midi = self._generate(input_seq, self.data_client.max_seq_len(), bpm)
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