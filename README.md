Northwestern KTP Spring 2025 Capstone Project
William Brittian

## Training data

BiMMuDa ships in `data/`. Two more melody sources are optional:

    ./scripts/download_data.sh    # HookTheory (CC BY-NC-SA 3.0) + POP909 (MIT) into data/raw/

Training uses every source that is present (or the `"sources"` list in
`model/config.json`). To train and score a model on held-out BiMMuDa songs:

    poetry run python scripts/experiment.py --name <run> --sources bimmuda pop909 hooktheory
    poetry run python scripts/sample_eval.py <run> --bar_guard --wav 8

`docs/research/melody-datasets.md` has the dataset survey and the results.
