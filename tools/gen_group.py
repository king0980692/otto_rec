import pickle5 as pickle
import numpy as np
import pandas as pd, numpy as np
import glob, gc
from pandas.core.common import maybe_make_list
from tqdm.auto import tqdm, trange
tqdm.pandas()
from pathlib import Path

VER = 1
type_labels = {'clicks':0, 'carts':1, 'orders':2}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str , required=True)
parser.add_argument('--valid_dir', type=str , required=True)
parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--out', type=str , required=True)
args = parser.parse_args()

Path(f"{args.out}/group").mkdir(parents=True, exist_ok=True)

def load_pqt(files):
    dfs = []
    for e, chunk_file in enumerate(glob.glob(files)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

## Train
print("Loading train ...")
train = load_pqt(args.train_dir)


print("Sorting session")
# split every session into group
temp = [group for name, group in train.sort_values(["session", "ts"]).groupby(["session"])]



PIECES = 500
Path(f"{args.out}/group").mkdir(parents=True, exist_ok=True)

for PART in trange(396,PIECES):

    #print('### PART {} - from {} to {}'.format(PART + 1, PART * len(temp)//PIECES, (PART+1) * len(temp) // PIECES))

    # len(temp) = total number of session
    session_list = []
    try:
        for h in temp[PART * len(temp)//PIECES:(PART+1) * len(temp) // PIECES]:
            session_list.append((h.session.iloc[0], h.aid.to_list(), h.type.to_list()))
    except:
        import IPython;IPython.embed(color='neutral');exit(1) 


    with open(f'{args.out}/group/train_group_tolist_{PART}_{VER}.pkl', 'wb') as f:
        pickle.dump(session_list, f)

del train
del temp
gc.collect()

## Valid
# valid = load_pqt('./data/split_chunked_parquet/test_parquet/*')
valid = load_pqt(args.valid_dir)


temp = [group for name, group in valid.sort_values(["session", "ts"]).groupby(["session"])]


PIECES = 5

for PART in range(PIECES):

    print('### PART {} - from {} to {}'.format(PART + 1, PART * len(temp)//PIECES, (PART+1) * len(temp) // PIECES))


    try:
        mylist = [[h.session.iloc[0], h.aid.to_list(), h.type.to_list()] for h in temp[PART * len(temp)//PIECES:(PART+1) * len(temp) // PIECES]]
    except:
        import IPython;IPython.embed(color='neutral');exit(1) 


    with open(f'{args.out}/group/valid_group_tolist_{PART}_{VER}.pkl', 'wb') as f:
        pickle.dump(mylist, f)

del valid
del temp
gc.collect()


## Test

test = load_pqt('./data/chunked_parquet/test_parquet/*')

temp = [group for name, group in test.sort_values(["session", "ts"]).groupby(["session"])]
print(len(temp))


PIECES = 5

Path(f"{args.out}/group").mkdir(parents=True, exist_ok=True)
for PART in range(PIECES):

    print('### PART {} - from {} to {}'.format(PART + 1, PART * len(temp)//PIECES, (PART+1) * len(temp) // PIECES))

    mylist = [[h.session.iloc[0], h.aid.to_list(), h.type.to_list()] for h in temp[PART * len(temp)//PIECES:(PART+1) * len(temp) // PIECES]]

    with open(f'{args.out}/group/test_group_tolist_{PART}_{VER}.pkl', 'wb') as f:
        pickle.dump(mylist, f)
