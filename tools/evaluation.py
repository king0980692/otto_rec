from pickle import load
from tqdm.auto import tqdm
tqdm.pandas()
import numpy as np
import sys
import pandas as pd
from multiprocessing import Pool

benchmark = {"clicks":0.5255597442145808, "carts":0.4093328152483512, "orders":0.6487936598117477, "all":.5646320148830121}
weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}

## --------

def hits(b):
    # b[0] : session id
    # b[1] : ground truth
    # b[2] : aids prediction 
    return b[0], len(set(b[1]).intersection(set(b[2]))), np.clip(len(b[1]), 0, 20)

def df_parallelize_run(func, t_split):
    num_cores = 32
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    
    return df
def otto_metric_piece(values, typ):
    """计算单一指标的recall
    c1
              session                                             labels
    0        11098528  [11830, 1732105, 588923, 884502, 1157882, 5717...
    1        11098529  [1105029, 295362, 132016, 459126, 890962, 1135...
    2        11098530  [409236, 264500, 1603001, 963957, 254154, 5830..
    """

    '''
    values['session'] = values['session_type']
    values.drop('session_type', axis=1)
    '''

    c1 = pd.DataFrame(values, columns=["labels"]).reset_index().rename({"index":"session"}, axis=1)
    """a 加入了两列：type == order和 ground_truth
             session    type                                  ground_truth            labels
    0       11098528  orders  [990658, 950341, 1462506, 1561739, 907564, 369..]   [11830, 1732105, 588923, 884502, 1157882, 5717...
    1       11098530  orders                                           [409236]   [409236, 264500, 1603001, 963957, 254154, 5830...
    2       11098531  orders                                          [1365569]   [396199, 1271998, 452188, 1728212, 1365569, 62... 
    """

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
    # print('{} recall = {:.5f} (vs {:.5f} in benchmark)'.format(typ ,recall, benchmark[typ]))
    
    return recall


# #### 计算三项type的recall
def otto_metric(clicks, carts, orders, verbose = True):
    score = 0
    clicks_recall = otto_metric_piece(clicks, "clicks")
    carts_recall = otto_metric_piece(clicks, "carts")
    orders_recall = otto_metric_piece(clicks, "orders")

    score += weights["clicks"] * clicks_recall
    score += weights['carts']  * carts_recall
    score += weights["orders"] * orders_recall
    if verbose:
        print('=============')
        print('Overall Recall = {:.5f} (vs {:.5f} in benchmark)'.format(score, benchmark["all"]))
        print('\t clicks_recall = {:.5f}'.format(clicks_recall))
        print('\t carts_recall = {:.5f}'.format(carts_recall))
        print('\t orders_recall = {:.5f}'.format(orders_recall))
        print('=============')
    
    return score


## ------

valid_labels = pd.read_parquet('./data/split_chunked_parquet/test_labels.parquet')

def load_type_df(path, type):
    df = pd.read_csv(f'./submission_{type}.csv')
    df = pd.Series(df.labels.to_list() , index = df.session_type.to_list())
    df = df.str.split().apply(lambda x: [int(i) for i in x])

    return df

def convert_(df):
    df = pd.Series(df.labels.to_list() , index = df.session_type.to_list())
    df = df.str.split().apply(lambda x: [int(i) for i in x])

    return df

'''
df = pd.read_csv(f'./submission.csv')

clicks_df = df[df['session_type'].str.contains('_clicks')]
clicks_df['session'] = clicks_df['session_type'].apply(lambda x:x[:-7])
clicks_df = clicks_df.drop('session_type', axis=1)

clicks_df = pd.Series(clicks_df.labels.to_list() , index = clicks_df.session.to_list())
clicks_df = clicks_df.str.split().apply(lambda x: [int(i) for i in x])

carts_df = df[df['session_type'].str.contains('_carts')]
carts_df = carts_df['session_type'].apply(lambda x:x[:-7])

orders_df = df[df['session_type'].str.contains('_orders')]
orders_df = orders_df['session_type'].apply(lambda x:x[:-7])
'''

clicks_df = load_type_df('./', 'clicks')
carts_df = load_type_df('./', 'carts')
orders_df = load_type_df('./', 'orders')


# recall = otto_metric_piece( clicks_df, sys.argv[2])
otto_metric(clicks_df, carts_df, orders_df)
