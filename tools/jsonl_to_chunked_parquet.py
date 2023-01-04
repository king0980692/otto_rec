import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()


data_path = Path('./data/raw_data')
chunksize = 100_000

Path(f"{args.out}/chunked_parquet/").mkdir(parents=True, exist_ok=True)

## Split Train
chunks = pd.read_json(data_path / 'train.jsonl', lines=True, chunksize=chunksize)

Path(f"{args.out}/chunked_parquet/train_parquet").mkdir(parents=True, exist_ok=True)

for e, chunk in enumerate(tqdm(chunks, total=129)):
    event_dict = {
            'session': [], 'aid': [], 'ts': [],
            'type': [],
            }

    for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
        for event in events:
            event_dict['session'].append(session)
            event_dict['aid'].append(event['aid'])
            event_dict['ts'].append(event['ts'])
            event_dict['type'].append(event['type'])

    # save DataFrame
    start = str(e*chunksize).zfill(9)
    end = str(e*chunksize+chunksize).zfill(9)
    pd.DataFrame(event_dict).to_parquet(f"./{args.out}/chunked_parquet/train_parquet/{start}_{end}.parquet")

## Split Ttest

chunks = pd.read_json(data_path / 'test.jsonl', lines=True, chunksize=chunksize)

Path(f"{args.out}/chunked_parquet/test_parquet").mkdir(parents=True, exist_ok=True)

for e, chunk in enumerate(tqdm(chunks, total=17)):
    event_dict = {
            'session': [],
            'aid': [],
            'ts': [],
            'type': [],
            }

    for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
        for event in events:
            event_dict['session'].append(session)
            event_dict['aid'].append(event['aid'])
            event_dict['ts'].append(event['ts'])
            event_dict['type'].append(event['type'])

    # save DataFrame
    start = str(e*chunksize).zfill(9)
    end = str(e*chunksize+chunksize).zfill(9)
    pd.DataFrame(event_dict).to_parquet(f"./{args.out}/chunked_parquet/test_parquet/{start}_{end}.parquet")
