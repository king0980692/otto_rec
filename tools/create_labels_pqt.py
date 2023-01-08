import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
import glob
import pandas as pd
from beartype import beartype
from pandas.io.json._json import JsonReader
from tqdm.auto import tqdm

@beartype
def ground_truth(events: list[dict]):
    prev_labels = {"clicks": None, "carts": set(), "orders": set()}

    for event in reversed(events):
        event["labels"] = {}

        for label in ['clicks', 'carts', 'orders']:
            if prev_labels[label]:
                if label != 'clicks':
                    event["labels"][label] = prev_labels[label].copy()
                else:
                    event["labels"][label] = prev_labels[label]

        if event["type"] == "clicks":
            prev_labels['clicks'] = event["aid"]
        if event["type"] == "carts":
            prev_labels['carts'].add(event["aid"])
        elif event["type"] == "orders":
            prev_labels['orders'].add(event["aid"])

    return events[:-1]

class setEncoder(json.JSONEncoder):

    def default(self, obj):
        return list(obj)

@beartype
def split_events(events: list[dict], split_idx=None):
    test_events = ground_truth(deepcopy(events))
    if not split_idx:
        split_idx = random.randint(1, len(test_events))
    test_events = test_events[:split_idx]
    labels = test_events[-1]['labels']
    for event in test_events:
        del event['labels']
    return test_events, labels


def load_test(files):    
    dfs = []

    # type_labels = {'clicks':0, 'carts':1, 'orders':2}
    for e, chunk_file in enumerate(glob.glob(str(files))):
        chunk = pd.read_parquet(chunk_file)
        # chunk.ts = (chunk.ts/1000).astype('int32')
        # chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)

    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

@beartype
def create_kaggle_testset(sessions: pd.DataFrame, sessions_output: Path, labels_output: Path):
    import IPython;IPython.embed(color='neutral');exit(1) 

    last_labels = []
    splitted_sessions = []

    for _, session in tqdm(sessions.iterrows(), desc="Creating trimmed testset", total=len(sessions)):
        session = session.to_dict()
        splitted_events, labels = split_events(session['type'])
        last_labels.append({'session': session['session'], 'labels': labels})
        splitted_sessions.append({'session': session['session'], 'events': splitted_events})

    with open(sessions_output, 'w') as f:
        for session in splitted_sessions:
            f.write(json.dumps(session) + '\n')

    with open(labels_output, 'w') as f:
        for label in last_labels:
            f.write(json.dumps(label, cls=setEncoder) + '\n')


@beartype
def main(pqt_path: Path, output_path: Path):

    test_sessions = load_test(pqt_path/'*.parquet')

    test_file_full = output_path / 'train_sessions_full.jsonl'

    # test_sessions = pd.read_json(test_file_full, lines=True)
    test_sessions_file = output_path / 'train_sessions.jsonl'
    test_labels_file = output_path / 'train_labels.jsonl'
    create_kaggle_testset(test_sessions, test_sessions_file, test_labels_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()

    main(args.file, args.output_path)
