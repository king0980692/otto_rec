from tqdm.auto import tqdm
from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)

import polars as pl
# from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import glob

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
torch_geometric.__version__
import gc

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--valid_dir', type=str , required=True)
parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--train_dir', type=str , required=True)
parser.add_argument('--worker', type=int , default=16)
# parser.add_argument('--out', type=str , required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_df = pl.read_parquet(args.train_dir,
                           columns=['session','aid'],
                           low_memory= True,
                          )
test_df = pl.read_parquet(args.test_dir,columns=['session','aid'])

def lagged_df(df):
    df = df.with_column(pl.col("aid").shift(periods=1).over("session")
                              #.cast(pl.Int32)
                              #.fill_null(pl.col("aid"))
                              .alias("prev_aid"))
    return df

train_df = lagged_df(train_df)
test_df = lagged_df(test_df)

df = pl.concat([
    train_df,
    test_df
], how="vertical")
edges_torch_T = torch.tensor(np.transpose(df[['prev_aid','aid']].to_numpy()),dtype=torch.long)
torch.save(edges_torch_T,"./exp/all_edges_train_and_test.pt")

edges_tensor = torch.load("./exp/all_edges_train_and_test.pt")

data = Data(edge_index=edges_tensor)
print(data)

del edges_tensor
gc.collect()

## ---

model = Node2Vec(data.edge_index, embedding_dim=32, 
                 walk_length=15,                        # lenght of rw
                 context_size=5, walks_per_node=10,
                 num_negative_samples=2, 
                 p=0.2, q=0.4,                             # bias parameters
                 sparse=True).to(device)

loader = model.loader(batch_size=64, shuffle=True,
                      num_workers=args.worker)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

'''
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
'''

for epoch in range(0, 10):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss / len(loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

Path('{args.out}/n2v').mkdir(parents=True, exist_ok=True)
model.save(f'{args.out}/n2v/n2v.ckpt')

## ---



type_labels = {'clicks':0, 'carts':1, 'orders':2}
def load_df(files):    
    dfs = []
    for e, chunk_file in enumerate(glob.glob(files)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

## -------
train = pl.DataFrame(load_df(args.train_dir))
test = pl.DataFrame(load_df(args.test_dir))
# valid = pl.DataFrame(load_df(args.valid_dir))

# train = pl.DataFrame(load_df('./data/split_chunked_parquet/train_parquet/*'))
# test = pl.DataFrame(load_df('./data/split_chunked_parquet/test_parquet/*'))

# train = pl.read_parquet('./data/parquet_data/train.parquet')
# test = pl.read_parquet ('./data/parquet_data/test.parquet')


sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)

sentences = sentences_df['sentence'].to_list()

'''
w2vec = Word2Vec.load('./w2v.model')
w2vec = Word2Vec.load('./w2v.model')
'''

print("Train w2v model ..")
# w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=2, workers=32, window=3,negative=10, sg=1, epochs=1)
# w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=48, epochs=5, negative=10, window=3)
w2vec = Word2Vec(sentences=sentences, vector_size=50, epochs=5, sg=1, window=3, sample=1e-3, ns_exponent=1, min_count=1, workers=args.worker)
w2vec.save('w2v.model')
print("Train w2v model done !!")


