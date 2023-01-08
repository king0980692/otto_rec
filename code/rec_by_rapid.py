from re import sub
from tqdm.auto import tqdm
tqdm.pandas()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

VER = 1
import pandas as pd, numpy as np
import pickle, glob, gc

from collections import Counter
import itertools
from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--pqt', type=str, required=True)
parser.add_argument('--group', type=str, required=True)
parser.add_argument('--mode', choices=['valid', 'test', 'train'])
parser.add_argument('--out', type=str, required=True)

args = parser.parse_args()



# ## Validation

# In[3]:


type_labels = {'clicks':0, 'carts':1, 'orders':2}

#type_weight_multipliers = {'clicks': 1, 'carts': 6, 'orders': 3}
type_weight_multipliers = {0: 1, 1: 6, 2: 3}

def load_df(files):    
    dfs = []
    for e, chunk_file in enumerate(glob.glob(files)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

# LOAD THREE CO-VISITATION MATRICES
def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.progress_apply(list).to_dict()

def df_parallelize_load(func, paths):
    num_cores = 32
    pool = Pool(num_cores)
    df = pool.map(func, paths)
    pool.close()
    pool.join()
    
    return df
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

def suggest_buys(df, top_n=20):
    # USE USER HISTORY AIDS AND TYPES
    session = df[0]
    aids = df[1]
    types = df[2]

    unique_aids = list(dict.fromkeys(aids[::-1] ))
    unique_buys = list(dict.fromkeys( [f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1] ))

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=top_n:
        
        weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
        for aid in aids3: aids_temp[aid] += 0.1
        sorted_aids = [k for k,v in aids_temp.most_common(top_n)]
        return session, sorted_aids
            
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in unique_aids if aid in top_20_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(top_n) if aid2 not in unique_aids] 
    result = unique_aids + top_aids2[:top_n - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return session, result + list(top_orders)[:top_n-len(result)]



## ----------------------

if args.mode == 'train':
    print("train")
    df = load_df('./data/split_chunked_parquet/train_parquet/*')
    PIECES = 500
    top_from = 'valid'

elif args.mode == 'valid':
    print("valid")
    df = load_df('./data/split_chunked_parquet/test_parquet/*')
    PIECES = 5
    top_from = 'valid'
elif args.mode == 'test':
    print("test")
    df = load_df('./data/chunked_parquet/test_parquet/*')
    PIECES = 5
    top_from = 'test'

## ----------------------
DISK_PIECES = 4

## Top 20 Clicks
print("load top_20_clicks ...")
temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'{args.pqt}/top_20_{top_from}_clicks_v1_{v}.pqt') for v in range(0, DISK_PIECES)])

temp[0].update(temp[1])
temp[0].update(temp[2])
temp[0].update(temp[3])
top_20_clicks = temp[0]

## Top 20 Buys
print("load top_20_buys ...")
temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'{args.pqt}/top_15_{top_from}_carts_orders_v1_{v}.pqt') for v in range(0, 4)])

temp[0].update(temp[1])
temp[0].update(temp[2])
temp[0].update(temp[3])
top_20_buys = temp[0]


## Top 20 Buy2buy
print("load buy2buy ...")
top_20_buy2buy = pqt_to_dict( pd.read_parquet(f'{args.pqt}/top_15_{top_from}_buy2buy_v1_0.pqt') )

del temp
gc.collect()


# TOP CLICKS AND ORDERS IN TEST
top_clicks = df.loc[df['type']==0, 'aid'].value_counts().index.values[:20]
top_orders = df.loc[df['type']==2, 'aid'].value_counts().index.values[:20]

print('Here are size of our 3 co-visitation matrices:')
print( len( top_20_clicks ), len( top_20_buy2buy ), len( top_20_buys ) )



# valid_labels = pd.read_parquet('./data/split_chunked_parquet/test_labels.parquet')
# import IPython;IPython.embed(color='neutral');exit(1) 

valid_bysession_list = []
for PART in range(PIECES):
    with open(f'{args.group}/group/{args.mode}_group_tolist_{PART}_1.pkl', 'rb') as f:
        valid_bysession_list.extend(pickle.load(f))
print(len(valid_bysession_list))


print(f'Predict {args.mode} clicks ...')
# Predict on all sessions in parallel
temp = df_parallelize_run(suggest_clicks, valid_bysession_list)
top_clicks = pd.Series([f[1] for f in temp], index=[f[0] for f in temp])


print(f'Predict {args.mode} buys ...')
# Predict on all sessions in parallel
temp = df_parallelize_run(suggest_buys, valid_bysession_list)
top_buys = pd.Series([f[1]  for f in temp], index=[f[0] for f in temp])


# Generate three type of submission to evaluation
def series_to_csv(series):
    out_df = pd.DataFrame({'session_type': series.index, 'labels': series.to_list()})
    out_df['labels'] = [' '.join(map(str, l)) for l in out_df['labels'] ]
    return out_df

for _type, ser in zip(['clicks', 'carts', 'orders'], [top_clicks, top_buys, top_buys]):
    out_df = series_to_csv(ser) 
    out_df.to_csv(f'{args.out}/cov_{args.mode}_predictions_{_type}.csv', index=False)

