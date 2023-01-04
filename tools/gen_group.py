import pandas as pd, numpy as np
import pickle, glob, gc
from tqdm.auto import tqdm
tqdm.pandas()
import argparse
from pathlib import Path

VER = 1
type_labels = {'clicks':0, 'carts':1, 'orders':2}

parser = argparse.ArgumentParser()
parser.add_argument('--valid_dir', type=str , required=True)
parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--out', type=str , required=True)
args = parser.parse_args()

def load_test(files):
    dfs = []
    for e, chunk_file in enumerate(glob.glob(files)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

## Valid
# valid = load_test('./data/split_chunked_parquet/test_parquet/*')
valid = load_test(args.valid_dir)
print('Valid data has shape',valid.shape)
valid.head()


temp = [group for name, group in valid.sort_values(["session", "ts"]).groupby(["session"])]


PIECES = 5
Path(f"{args.out}/group").mkdir(parents=True, exist_ok=True)

for PART in range(PIECES):

    print('### PART {} - from {} to {}'.format(PART + 1, PART * len(temp)//PIECES, (PART+1) * len(temp) // PIECES))


    mylist = [[h.session.iloc[0], h.aid.to_list(), h.type.to_list()] for h in temp[PART * len(temp)//PIECES:(PART+1) * len(temp) // PIECES]]

    with open(f'{args.out}/group/valid_group_tolist_{PART}_{VER}.pkl', 'wb') as f:
        pickle.dump(mylist, f)



## Test

test = load_test('./data/chunked_parquet/test_parquet/*')
print('Test data has shape',test.shape)
test.head()

temp = [group for name, group in test.sort_values(["session", "ts"]).groupby(["session"])]
print(len(temp))


PIECES = 5

Path(f"{args.out}/group").mkdir(parents=True, exist_ok=True)
for PART in range(PIECES):

    print('### PART {} - from {} to {}'.format(PART + 1, PART * len(temp)//PIECES, (PART+1) * len(temp) // PIECES))

    mylist = [[h.session.iloc[0], h.aid.to_list(), h.type.to_list()] for h in temp[PART * len(temp)//PIECES:(PART+1) * len(temp) // PIECES]]

    with open(f'{args.out}/group/test_group_tolist_{PART}_{VER}.pkl', 'wb') as f:
        pickle.dump(mylist, f)
