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

parser.add_argument('--mode', choices=['valid', 'test'])

args = parser.parse_args()



# ## Validation

# In[3]:


type_labels = {'clicks':0, 'carts':1, 'orders':2}

#type_weight_multipliers = {'clicks': 1, 'carts': 6, 'orders': 3}
type_weight_multipliers = {0: 1, 1: 6, 2: 3}

def load_test(files):    
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
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    
    return df

def suggest_clicks(df):
    session = df[0]
    aids = df[1]
    types = df[2]
    unique_aids = list(dict.fromkeys(aids[::-1]))
    # Rerank candidates using weights
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # Rerank based on repeat items and type of items
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return session, sorted_aids
    
    # Use "clicks" co-visitation matrix
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))
    # Rerank candidates
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    
    # Use top20 test clicks
    return session, result + list(top_clicks)[:20 - len(result)]

def suggest_buys(df):
    # USE USER HISTORY AIDS AND TYPES
    session = df[0]
    aids = df[1]
    types = df[2]

    unique_aids = list(dict.fromkeys(aids[::-1] ))
    unique_buys = list(dict.fromkeys( [f for i, f in enumerate(aids) if types[i] in [1, 2]][::-1] ))

    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=20:
        
        weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
        for aid in aids3: aids_temp[aid] += 0.1
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return session, sorted_aids
            
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in unique_aids if aid in top_20_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(20) if aid2 not in unique_aids] 
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return session, result + list(top_orders)[:20-len(result)]

def hits(b):
    # b[0] : session id
    # b[1] : ground truth
    # b[2] : aids prediction 
    return b[0], len(set(b[1]).intersection(set(b[2]))), np.clip(len(b[1]), 0, 20)


def otto_metric_piece(values, typ, verbose=True):

    global valid_labels , benchmark
    """计算单一指标的recall
    c1
              session                                             labels
    0        11098528  [11830, 1732105, 588923, 884502, 1157882, 5717...
    1        11098529  [1105029, 295362, 132016, 459126, 890962, 1135...
    2        11098530  [409236, 264500, 1603001, 963957, 254154, 5830..
    """
    c1 = pd.DataFrame(values, columns=["labels"]).reset_index().rename({"index":"session"}, axis=1)

    """a 加入了两列：type == order和 ground_truth
             session    type                                       ground_truth                labels
    0       11098528  orders  [990658, 950341, 1462506, 1561739, 907564, 369...            [11830, 1732105, 588923, 884502, 1157882, 5717...
    1       11098530  orders                                           [409236]            [409236, 264500, 1603001, 963957, 254154, 5830...
    2       11098531  orders                                          [1365569]            [396199, 1271998, 452188, 1728212, 1365569, 62... 
    """
    import IPython;IPython.embed(color='neutral');exit(1) 

    a = valid_labels.loc[valid_labels['type'] == typ].merge(c1, how='left', on=['session'])
    b = [[a0, a1, a2] for a0, a1, a2 in zip(a['session'], a['ground_truth'], a['labels'])]
    c = df_parallelize_run(hits, b)
    """c
    [[11098528        1       11]
     [11098530        1        1]
     [11098531        1        1]
    """
    c = np.array(c)
    recall = c[:, 1].sum() / c[:, 2].sum()
    print('{} recall = {:.5f} (vs {:.5f} in benchmark)'.format(typ ,recall, benchmark[typ]))
    
    return recall


# #### 计算三项type的recall
def otto_metric(clicks, carts, orders, verbose = True):
    score = 0
    score += weights["clicks"] * otto_metric_piece(clicks, "clicks", verbose = verbose)
    score += weights['carts'] * otto_metric_piece(carts, "carts", verbose = verbose)
    score += weights["orders"] * otto_metric_piece(orders, "orders", verbose = verbose)
    if verbose:
        print('=============')
        print('Overall Recall = {:.5f} (vs {:.5f} in benchmark)'.format(score, benchmark["all"]))
        print('=============')
    
    return score

## ----------------------

if args.mode == 'valid':
    valid = load_test('./data/split_chunked_parquet/test_parquet/*')
    print('Valid data has shape',valid.shape)
    valid.head()


    DISK_PIECES = 4

    print("load top_20_clicks ...")
    temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'./exp2/top_20_valid_clicks_v1_{v}.pqt') for v in range(0, 4)])

    temp[0].update(temp[1])
    temp[0].update(temp[2])
    temp[0].update(temp[3])
    top_20_clicks = temp[0]



    print("load top_20_buys ...")
    temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'exp2/top_15_valid_carts_orders_v1_{v}.pqt') for v in range(0, 4)])

    temp[0].update(temp[1])
    temp[0].update(temp[2])
    temp[0].update(temp[3])
    top_20_buys = temp[0]

    del temp
    gc.collect()

    print("load buy2buy ...")
    top_20_buy2buy = pqt_to_dict( pd.read_parquet(f'./exp2/top_15_valid_buy2buy_v1_0.pqt') )


    '''
    top_20_clicks = pqt_to_dict( pd.read_parquet(f'../input/otto-co-visitation-matrices/top_20_valid_clicks_v{VER}_0.pqt') )
    for k in range(1, DISK_PIECES): 
        top_20_clicks.update( pqt_to_dict( pd.read_parquet(f'../input/otto-co-visitation-matrices/top_20_valid_clicks_v{VER}_{k}.pqt') ) )


    top_20_buys = pqt_to_dict( pd.read_parquet(f'../input/otto-co-visitation-matrices/top_15_valid_carts_orders_v{VER}_0.pqt') )
    for k in range(1, DISK_PIECES): 
        top_20_buys.update( pqt_to_dict( pd.read_parquet(f'../input/otto-co-visitation-matrices/top_15_valid_carts_orders_v{VER}_{k}.pqt') ) )
        
    '''

    # TOP CLICKS AND ORDERS IN TEST
    top_clicks = valid.loc[valid['type']==0, 'aid'].value_counts().index.values[:20]
    top_orders = valid.loc[valid['type']==2, 'aid'].value_counts().index.values[:20]

    print('Here are size of our 3 co-visitation matrices:')
    print( len( top_20_clicks ), len( top_20_buy2buy ), len( top_20_buys ) )


    # In[5]:

    valid_labels = pd.read_parquet('./data/split_chunked_parquet/test_labels.parquet')

    PIECES = 5
    valid_bysession_list = []
    for PART in range(PIECES):
        with open(f'./exp/group/valid_group_tolist_{PART}_1.pkl', 'rb') as f:
            valid_bysession_list.extend(pickle.load(f))
    print(len(valid_bysession_list))


    print('Predict val clicks ...')
    # Predict on all sessions in parallel
    temp = df_parallelize_run(suggest_clicks, valid_bysession_list)
    val_clicks = pd.Series([f[1] for f in temp], index=[f[0] for f in temp])


    _ = otto_metric_piece(val_clicks, "clicks")
    import IPython;IPython.embed(color='neutral');exit(1) 

    print('Predict val buys ...')
    # Predict on all sessions in parallel
    temp = df_parallelize_run(suggest_buys, valid_bysession_list)
    val_buys = pd.Series([f[1]  for f in temp], index=[f[0] for f in temp])






    benchmark = {"clicks":0.5255597442145808, "carts":0.4093328152483512, "orders":0.6487936598117477, "all":.5646320148830121}
    weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}


    _ = otto_metric_piece(val_buys, "orders")

    _ = otto_metric_piece(val_buys, "carts")


    _ = otto_metric(val_clicks, val_buys, val_buys)


elif args.mode == 'test':
    # ##------------------------------------

    # ## Test

    # Here a submission file is created.

    # In[ ]:


    test = load_test('./data/chunked_parquet/test_parquet/*')
    print('Test data has shape',test.shape)
    test.head()


    # In[ ]:


    print("load buy2buy ...")
    top_20_buy2buy = pqt_to_dict( pd.read_parquet(f'./exp2/top_15_test_buy2buy_v1_0.pqt') )

    print("load top_20_clicks ...")
    temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'./exp2/top_20_test_clicks_v1_{v}.pqt') for v in range(0, 4)])

    temp[0].update(temp[1])
    temp[0].update(temp[2])
    temp[0].update(temp[3])
    top_20_clicks = temp[0]

    print("load top_20_buys ...")
    temp = df_parallelize_load(pqt_to_dict, [pd.read_parquet(f'exp2/top_15_test_carts_orders_v1_{v}.pqt') for v in range(0, 4)])

    temp[0].update(temp[1])
    temp[0].update(temp[2])
    temp[0].update(temp[3])
    top_20_buys = temp[0]

    del temp
    gc.collect()

    # TOP CLICKS AND ORDERS IN TEST
    top_clicks = test.loc[test['type']==0, 'aid'].value_counts().index.values[:20]
    top_orders = test.loc[test['type']==2, 'aid'].value_counts().index.values[:20]

    print('Here are size of our 3 co-visitation matrices:')
    print( len( top_20_clicks ), len( top_20_buy2buy ), len( top_20_buys ) )




    PIECES = 5
    test_bysession_list = []
    for PART in range(PIECES):
        with open(f'./exp/group/test_group_tolist_{PART}_1.pkl', 'rb') as f:
            test_bysession_list.extend(pickle.load(f))
    print(len(test_bysession_list))




    # Predict on all sessions in parallel
    temp = df_parallelize_run(suggest_clicks, test_bysession_list)
    clicks_pred_df = pd.Series([f[1] for f in temp], index=[f[0] for f in temp])
    clicks_pred_df = clicks_pred_df.add_suffix("_clicks")
    clicks_pred_df.head()




    # Predict on all sessions in parallel
    temp = df_parallelize_run(suggest_buys, test_bysession_list)
    buys_pred_df = pd.Series([f[1] for f in temp], index=[f[0] for f in temp])
    orders_pred_df = buys_pred_df.add_suffix("_orders")
    carts_pred_df = buys_pred_df.add_suffix("_carts")



    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df]).reset_index()
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.progress_apply(lambda x: " ".join(map(str,x)))
    pred_df.to_csv("submission.csv", index=False)
    pred_df.head()

