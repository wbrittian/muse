from pysrc.data_client.data_client import DataClient

class Muse:
    tokenfile_path = "../../../data/processed_data/tokens.json"

data_client = DataClient()
data_client.load()