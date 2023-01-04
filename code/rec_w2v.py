from tqdm.auto import tqdm
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

from collections import defaultdict
import logging
import glob
logging.basicConfig(level=logging.INFO)

from gensim.models import Word2Vec
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--model_dir', type=str , required=True)
parser.add_argument('--out', type=str , required=True)
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

test = load_df(args.test_dir)

w2vec = Word2Vec.load(f'{args.model_dir}/w2v.model')
print("Start Recommendation ...")

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
index = AnnoyIndex(50, 'angular')

for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx])

index.build(10)


session_types = ['clicks', 'carts', 'orders']

test_session_AIDs = test.reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.reset_index(drop=True).groupby('session')['type'].apply(list)

labels = []

type_weight_multipliers = {0: 1, 1: 6, 2: 3}

for AIDs, types in tqdm(zip(test_session_AIDs, test_session_types)):

    '''
    if len(AIDs) >= 20:
        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1

        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types):
            aids_temp[aid]+= w * type_weight_multipliers[t]

        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:20])
    else:
    '''
    # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
    AIDs = list(dict.fromkeys(AIDs[::-1]))

    # let's grab the most recent aid
    most_recent_aid = AIDs[0]

    # and look for some neighbors!
    nns = [w2vec.wv.index_to_key[i] 
           for i in index.get_nns_by_item(
               aid2idx[most_recent_aid], 21)[1:]
          ]

    labels.append((AIDs+nns)[:20])



labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.to_csv(f'{args.out}/submission_{st}.csv', index=False)
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv(f'{args.out}/submission.csv', index=False)
