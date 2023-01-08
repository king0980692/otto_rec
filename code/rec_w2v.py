import itertools
from tqdm.auto import tqdm, trange
tqdm.pandas()
from annoy import AnnoyIndex
import pandas as pd
import numpy as np
import pickle as pickle
from collections import defaultdict
import logging
import glob
logging.basicConfig(level=logging.INFO)
from multiprocessing import Pool
from gensim.models import Word2Vec
from collections import Counter
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'valid', 'test'] , required=True)
parser.add_argument('--data', type=str , required=True)
parser.add_argument('--group', type=str , required=True)
parser.add_argument('--embed', type=str , required=True)
parser.add_argument('--dim', type=int, required=True)
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

def df_parallelize_run(func, t_split):
    num_cores = 32
    '''
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    '''

    with Pool(num_cores) as p:
        df = list(tqdm(p.imap_unordered(func, t_split), total=len(t_split))) 

    return df

def suggest_clicks(df, top_n = 20):
    session = df[0]
    aids = df[1]
    types = df[2]
    unique_aids = list(dict.fromkeys(aids[::-1]))
    # Rerank candidates using weights
    if len(unique_aids) >= top_n:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # Rerank based on repeat items and type of items
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(top_n)]
        return session, sorted_aids
    
    # Use "clicks" co-visitation matrix
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))
    # Rerank candidates
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(top_n) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:top_n - len(unique_aids)]
    
    # Use top20 test clicks
    return session, result + list(top_clicks)[:top_n - len(result)]

def rec_by_emb(s, top_n=20):
    global embed, aid2idx, index_to_key

    session = s[0]
    aids    = s[1]
    types   = s[2]
    # unique_aids = list(dict.fromkeys(aids[::-1]))

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
    aids = list(dict.fromkeys(aids[::-1]))

    # let's grab the most recent aid
    most_recent_aid = aids[0]

    labels = []
    if most_recent_aid in aid2idx:
        nns = [ index_to_key[i] 
                for i in index.get_nns_by_item(
                    aid2idx[most_recent_aid], top_n+1)[1:]
              ]
        labels.append((aids+nns)[:top_n])
    else:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1

        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(aids,weights,types):
            aids_temp[aid]+= w * type_weight_multipliers[t]

        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:top_n])


    return session, labels


def rec_w2v(session_IDs, session_types):

    labels = []

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}


    for id, (AIDs, types) in tqdm(enumerate(zip(session_IDs, session_types)), total=len(session_types)):

        # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))

        # let's grab the most recent aid
        most_recent_aid = None
        for _aid in AIDs:
            if _aid in aid2idx:
                most_recent_aid = _aid
                break



        if most_recent_aid == None:
            weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1

            aids_temp=defaultdict(lambda: 0)
            for aid,w,t in zip(AIDs,weights,types):
                aids_temp[aid]+= w * type_weight_multipliers[t]

            sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            labels.append(sorted_aids[:20])

        else:
            # if most_recent_aid in aid2idx:
            nns = [ index_to_key[i] 
                    for i in index.get_nns_by_item(
                        aid2idx[most_recent_aid], 21)[1:]
                  ]
            labels.append((AIDs+nns)[:20])
    return labels



## -------
if args.mode == 'train':
    df = load_df('./data/split_chunked_parquet/train_parquet/')
    PIECES = 500

elif args.mode == 'valid':
    test = load_df("./data/split_chunked_parquet/test_parquet/")
    # df = load_df("./data/split_raw_paruet/test.pqrquet")
    PIECES = 5

elif args.mode == 'test':
    df = load_df("./data/chunked_parquet/test_parquet/")
    PIECES = 5

embed = pickle.load(open(args.embed, 'rb'))
# aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
aid2idx = {aid: i for i, (aid,emb) in enumerate(embed.items())}
index_to_key = {i: aid for i, (aid,emb) in enumerate(embed.items())}

index = AnnoyIndex(args.dim, 'angular')

for aid, idx in aid2idx.items():
    index.add_item(idx, embed[aid])

index.build(10)


session_list = []
for PART in trange(PIECES, desc='Reading group: '):
    with open(f'{args.group}/group/{args.mode}_group_tolist_{PART}_1.pkl', 'rb') as f:
        session_list.extend(pickle.load(f))
print("Total Session number: {}".format(len(session_list)))


# train_session_AIDs = train.reset_index(drop=True).groupby('session')['aid'].progress_apply(list)
# train_session_types = train.reset_index(drop=True).groupby('session')['type'].progress_apply(list)

session_IDs = [ s[0] for s in session_list]
session_AIDs = [ s[1] for s in session_list]
session_types = [ s[2] for s in session_list]

print("Start Recommendation ...")

labels = []

type_weight_multipliers = {0: 1, 1: 6, 2: 3}

'''
    recommendation function
'''
labels = rec_w2v(session_AIDs, session_types)

labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]
predictions = pd.DataFrame(data={'session_type': session_IDs, 'labels': labels_as_strings})

prediction_dfs = []

# session_types = ['clicks', 'carts', 'orders']
for st in ['clicks', 'carts', 'orders']:
    modified_predictions = predictions.copy()
    modified_predictions.to_csv(f'{args.out}/w2v_{args.mode}_predictions_{st}.csv', index=False)
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

# submission = pd.concat(prediction_dfs).reset_index(drop=True)
# submission.to_csv(f'{args.out}/submission.csv', index=False)
