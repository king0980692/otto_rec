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
w2vec = Word2Vec(sentences=sentences, vector_size=50, epochs=5, sg=1, window=3, sample=1e-3, ns_exponent=1, min_count=1, workers=4)
w2vec.save('w2v.model')
print("Train w2v model done !!")


aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
index = AnnoyIndex(32, 'angular')

for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx])

index.build(10)
import IPython;IPython.embed(color='neutral');exit(1) 





print("Start Recommendation ...")
session_types = ['clicks', 'carts', 'orders']

test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

labels = []

type_weight_multipliers = {0: 1, 1: 6, 2: 3}

for AIDs, types in zip(test_session_AIDs, test_session_types):
    if len(AIDs) >= 20:
        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types):
            aids_temp[aid]+= w * type_weight_multipliers[t]

        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:20])
    else:
        # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))

        # let's grab the most recent aid
        most_recent_aid = AIDs[0]

        # and look for some neighbors!
        nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]

        labels.append((AIDs+nns)[:20])



labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.to_csv(f'submission_{st}.csv', index=False)
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv('submission.csv', index=False)
