from annoy import AnnoyIndex
import pandas as pd
import numpy as np

from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)

import polars as pl
# from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import glob

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--valid_dir', type=str , required=True)
parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--train_dir', type=str , required=True)
parser.add_argument('--worker', type=int , default=16)
# parser.add_argument('--out', type=str , required=True)
args = parser.parse_args()

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


