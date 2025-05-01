from pysrc.data_loader.data_loader import DataLoader

class Muse:
    tokenfile_path = "../../../data/processed_data/tokens.json"

loader = DataLoader()
loader.load("data/")