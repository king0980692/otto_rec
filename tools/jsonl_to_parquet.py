from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_jsonl', type=str, required=True)
parser.add_argument('--test_jsonl', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

chunks_size = 100_000
def jsonl_to_df(fn):
    sessions = []
    aids = []
    tss = []
    types = []
    
    chunks = pd.read_json(fn, lines=True, chunksize=chunks_size)


    for chunk in chunks:
        for row_idx, session_data in tqdm(chunk.iterrows()):
            num_event = len(session_data.events)
            sessions += ([session_data.session] * num_event)

            for event in session_data.events:
                aids.append(event['aid'])
                tss.append(event['ts'])
                types.append(event['type'])

    return pd.DataFrame(data = {'session': sessions, 'aids': aids, 'ts':tss, 'types': type})

Path("{args.out}/parquet_data").mkdir(parents=True, exist_ok=True)
train_df = jsonl_to_df(args.train_jsonl)
train_df.type = train_df.type.astype(np.uint8)
train_df.to_parquet(f'{args.out}/train.pqrquet', index=False)

test_df = jsonl_to_df(args.test_jsonl)
test_df.type = test_df.type.astype(np.uint8)
test_df.to_parquet(f'{args.out}/test.pqrquet', index=False)

