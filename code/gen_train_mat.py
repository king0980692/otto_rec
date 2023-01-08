import time
import gc
import numpy as np
import itertools
from collections import Counter
from prompt_toolkit.filters import cli
from tqdm.auto import tqdm, trange
tqdm.pandas()
from pathlib import Path
import pandas as pd
import argparse
import pickle
from multiprocessing import Pool
from gensim.models import Word2Vec
from functools import reduce

parser = argparse.ArgumentParser()
# parser.add_argument('--valid_dir', type=str , required=True)
# parser.add_argument('--test_dir', type=str , required=True)
parser.add_argument('--group', type=str , required=True)
parser.add_argument('--pqt', type=str , required=True)
parser.add_argument('--type', choices=['clicks', 'carts', 'orders'] , required=True)
parser.add_argument('--candidates', nargs='+', type=str , required=True)
parser.add_argument('--embeds', nargs='+', type=str , required=True)
args = parser.parse_args()

type_weight_multipliers = {0: 1, 1: 6, 2: 3}
type_dict = {"clicks": 0, "carts": 1, "orders": 2}

def pqt_to_dict(df):
    # df.groupby('aid_x').aid_y.progress_apply(list).to_dict()
    df['aid_wgt'] = list(zip(df.aid_y, df.wgt))
    return df.groupby('aid_x').aid_wgt.progress_apply(list).to_dict()

def load_pickle(path):
    # with open(f'{args.group}/group/train_group_tolist_{PART}_1.pkl', 'rb') as f:
    with open(path, 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def parallelized_load(func, paths):

    num_cores = 32

    '''
    pool = Pool(num_cores)
    df = pool.map(func, paths)
    pool.close()
    pool.join()
    '''

    with Pool(num_cores) as p:
        df = list(tqdm(p.imap_unordered(func, paths), total=len(paths)))

    return df

def df_parallelize_run(func, t_split):
    num_cores = 16
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()

    return df


def gen_by_emb(df):
    """
        from training record to generate training instance
    """
    global embeds, cand_dict, type_dict

    session = df[0]
    aids = df[1]
    types = df[2]

    unique_aids = list(dict.fromkeys(aids[::-1]))
    unique_aids = list(map(str, unique_aids))

    candidates = cand_dict[session] + unique_aids
    candidates = candidates[:40]
    # candidates = cand_dict[session]

    # if len(candidates) < 40:
        # candidates = candidates + unique_aids[:40 - len(candidates)]


    fea = []
    truths = []
    feas = []
    for embed in embeds:
        dim = next(iter(embed.values())).shape[0]
        truth = []
        fea = []
        cand_emb = np.array([
                        embed[int(ID)] if int(ID) in embed 
                        else np.zeros((dim))
                        for ID in candidates 
                    ])


        flag = False
        last_type = None
        for aid, _type in zip(aids[::-1], types[::-1]):
            if _type != type_dict[args.type]:
                continue
            if last_type == None:
                last_type = aid
            else:
                flag = True
                if aid in embed:
                    q_embed = embed[aid]

                    score = np.sum(q_embed*cand_emb, axis=1)
                    # dot products
                    fea.append(score.reshape((cand_emb.shape[0], 1)))
                    truth = [
                        1. if r == last_type
                        else 0.
                        for r in candidates
                    ]


                else:
                    fea.append(np.full((cand_emb.shape[0], 1), 0.))
                    truth = [0]*cand_emb.shape[0]
                break

        if flag == False:
            return [],[]



        # horizonal combined all embed feature
        dots = np.hstack(fea)
        truths = np.hstack(truth)
        truths = np.expand_dims(truths, axis=1)

        fea = np.hstack(
                (
                    dots,
                    np.sum(dots, axis=1).reshape(( cand_emb.shape[0], 1)),
                    np.amax(dots, axis=1).reshape((cand_emb.shape[0], 1)),
                    #cosines,
                    #np.sum(cosines, axis=1).reshape((len(candidates), 1)),
                    #np.amax(cosines, axis=1).reshape((len(candidates), 1))
                )
            )

        '''
        if fea.shape != (40,3):
            import IPython;IPython.embed(color='neutral');exit(1) 
        '''
        # first feas
        if not len(feas):
            feas = fea
        else:
            feas = np.hstack((feas, fea))

    return feas, truths
            

    

## --------------------

PIECES = 500
# Path(f"{args.group}/group").mkdir(parents=True, exist_ok=True)

print('Loading Group')
'''
temp = parallelized_load(load_pickle, [f'{args.group}/group/train_group_tolist_{PART}_1.pkl' for PART in range(PIECES)])
train_bysession_list = list(itertools.chain(*temp))
'''
train_bysession_list = []
for PART in trange(PIECES):
    with open(f'{args.group}/group/train_group_tolist_{PART}_1.pkl', 'rb') as f:
        train_bysession_list.extend(pickle.load(f))

print('Loading Candidates')
s_t = time.time()
temp = []
cand_dict = {}
for c_p in args.candidates:
    temp.append(pd.read_csv(c_p))
    '''
    df = pd.read_csv(c_p)
    df['labels'] = df['labels'].str.split()
    df = df.set_index('session_type')
    dict = df.to_dict()['labels']
    temp.append(dict)
    # cand_dict.update(df.to_dict()['labels'])
    '''

# temp = parallelized_load(pd.read_csv, args.candidates)


'''
for k in temp[0].keys():
    # df['labels'] = df['labels'].str.split()
    # df = df.set_index('session_type')
    # _dict = df.to_dict()['labels']

    cand_dict[k] = list(itertools.chain([d[k] for d in temp]))
'''

# '''
print('Merge Candidates ...')
cand_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='session_type'), temp)
cand_df['labels'] = cand_df['labels_x'] + " " +cand_df['labels_y']
cand_df = cand_df.drop(['labels_x', 'labels_y'], axis=1)
cand_df['labels'] = cand_df['labels'].str.split()
cand_df = cand_df.set_index('session_type')
cand_dict = cand_df.to_dict()['labels']
# '''

print('costing time {}'.format(time.time() - s_t))

del temp
gc.collect()


print("Loading embeddings")

embeds = []
for p in args.embeds:
    with open(p, 'rb') as p:
        embeds.append(pickle.load(p))

print("Loading embeddings done !")

'''
w2vec = Word2Vec.load('./w2v.model')

aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}

embed = { aid: w2vec.wv.vectors[idx] for aid, idx in aid2idx.items()}
'''




'''
DISK_PIECES = 4

print("load top_20_clicks ...")

chunked_pqt = [pd.read_parquet(f'{args.pqt}/valid_clicks_v1_{v}.pqt') for v in range(0, DISK_PIECES)]

print("gen wgt_clicks ...")
temp = parallelized_load(pqt_to_dict, chunked_pqt)

temp[0].update(temp[1])
temp[0].update(temp[2])
temp[0].update(temp[3])
wgt_clicks = temp[0]

print("wgt_clicks done .")
'''

def gen_X_y_g(
    is_train,
    sessions,
    purchases,
    ranks,
    candidates,
    pop_retrievals,
    item_map,
    embeds,
    sims,
    date_sims,
    date_feas,
    backtracking,
    loop,
):
    Xs, ys, gs, cs = [], [], [], []
    matched, missed, data_size = 0., 0., 0.
    for s_id, aids, types in tqdm(train_bysession_list, total=len(train_bysession_list)):
        '''
        if not ranks[session_id]:
            print('--ranks has wrong file')
            exit()
        '''

        # seen = {item_id:1 for item_id, date in sessions[session_id]}
        # _sessions = sessions[session_id]

        retrievals = [item_id for item_id in cand_dict[s_id] ]
        y = [0. for _ in retrievals]
        if purchases:
            y = [
                1. if r == purchases[session_id]
                    #else 1. if r in seen
                        else 0.
                            for r in retrievals
            ]

        if sum(y) > 0:
            matched += 1.
        else:
            missed += 1.
            if is_train:
                continue


        data_size += len(y)
        cs += retrievals
        Xs.append(
            np.hstack(
                (
                    #gen_date_sim_fea(date_sims, _sessions, retrievals, backtracking),
                    #gen_idate_fea(date_feas, _sessions, retrievals),
                    #gen_raw_fea([u2is[-1]], session_id, retrievals),

                    gen_u2i_fea(u2is, session_id, retrievals),
                    gen_pop_fea(pops, _sessions, retrievals),
                    gen_meta_fea(metas, _sessions, retrievals, backtracking),
                    gen_embed_last_fea(embeds, _sessions, retrievals, backtracking),
                    gen_sim_last_fea(sims, _sessions, retrievals, backtracking),
                )
            )
        )

        gs += [len(y)]
        ys += y[:]

    Xs = np.concatenate(Xs)
    Xs = np.nan_to_num(Xs, neginf=0)
    #Xs = np.hstack((Xs, get_group_features(Xs, gs)))
    print(f'data groups:\t{len(gs)}')
    print(f'data size:\t{data_size}')
    print(f'data Recall:\t{matched/(matched+missed)}')
    return Xs, np.array(ys), np.array(gs), np.array(cs)

# train_X, train_Y =  df_parallelize_run(gen_by_emb, train_bysession_list)

ids = []
Xs = []
Ys = []
for t in tqdm(train_bysession_list[:], desc='Gen X, Y'):
    X, Y = gen_by_emb(t)
    Xs.append(X)
    Ys.append(Y)
import IPython;IPython.embed(color='neutral');exit(1) 

train_Xs = np.vstack(Xs)
train_Ys = np.vstack(Ys)

with open('./trainX.npy', 'wb') as f:
    np.save(f, train_Xs)
    np.save(f, train_Ys)

