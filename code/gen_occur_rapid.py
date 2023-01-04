VER = 1
from tqdm.auto import tqdm
tqdm.pandas()
import pandas as pd, numpy as np
import glob, gc
import cudf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['valid', 'test'])
args = parser.parse_args()

print('We will use RAPIDS version', cudf.__version__)



# CACHE FUNCTIONS
def read_file(f):
    return cudf.DataFrame( data_cache[f] )

def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype('int32')
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df

## --------

def heart_carts_order(df, from_, to_, ndays = 1, type_weight = {0:1, 1:6, 2:3}):
    
    df = df.sort_values(['session','ts'], ascending = [True, False])
    
    # USE TAIL OF SESSION
    df = df.reset_index(drop = True)
    df['n'] = df.groupby('session').cumcount()
    df = df.loc[df.n < 30].drop('n', axis = 1)
    
    # CREATE PAIRS
    df = df.merge(df,on='session')
    df = df.loc[ ((df.ts_x - df.ts_y).abs()< ndays * 24 * 60 * 60) & (df.aid_x != df.aid_y) ]
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= from_) & (df.aid_x < to_)]
    
    # ASSIGN WEIGHTS
    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])

    df['wgt'] = df.type_y.map(type_weight)
    df = df[['aid_x', 'aid_y', 'wgt']]
    df.wgt = df.wgt.astype('float32')

    #df.groupby(['aid_x', 'aid_y']).wgt.sum()
    return df.groupby(['aid_x', 'aid_y']).wgt.sum()

## -------------

def heart_buy2buy(df, from_, to_, ndays = 14):
    
    # ONLY WANT CARTS AND ORDERS
    df = df.loc[df['type'].isin([1,2])] 
    df = df.sort_values(['session','ts'],ascending=[True,False])
    
    # USE TAIL OF SESSION
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount()
    df = df.loc[df.n<30].drop('n',axis=1)
    
    # CREATE PAIRS
    df = df.merge(df,on='session')
    df = df.loc[ ((df.ts_x - df.ts_y).abs()< ndays * 24 * 60 * 60) & (df.aid_x != df.aid_y) ] 
    
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= from_) & (df.aid_x < to_)]
    
    # ASSIGN WEIGHTS
    df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
    df['wgt'] = 1
    df = df[['aid_x','aid_y','wgt']]
    df.wgt = df.wgt.astype('float32')
    
    return df.groupby(['aid_x','aid_y']).wgt.sum()


## -------------

def heart_clicks(df, from_, to_, ndays = 1):

    df = df.sort_values(['session','ts'],ascending=[True,False])
    
    # USE TAIL OF SESSION
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount()
    df = df.loc[df.n<30].drop('n',axis=1)
    
    # CREATE PAIRS
    df = df.merge(df,on='session')
    df = df.loc[ ((df.ts_x - df.ts_y).abs()< ndays * 24 * 60 * 60) & (df.aid_x != df.aid_y) ]
    
    # MEMORY MANAGEMENT COMPUTE IN PARTS
    df = df.loc[(df.aid_x >= from_)&(df.aid_x < to_)]
    
    # ASSIGN WEIGHTS
    df = df[['session', 'aid_x', 'aid_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
    df['wgt'] = 1 + 3 * (df.ts_x - 1659304800)/(1662328791-1659304800)
    df = df[['aid_x','aid_y','wgt']]
    df.wgt = df.wgt.astype('float32')
                                     
    return df.groupby(['aid_x','aid_y']).wgt.sum()


## -------------

def create_covisitation(fn_heart, DISK_PIECES, save_top, output_name):
    
    SIZE = 1.86e6 / DISK_PIECES
    
    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
        print()
        print('--- DISK PART',PART+1)
        print("from : {}, to : {}".format(PART * SIZE,  (PART+1) *SIZE))
    
        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min( (j + 1) * CHUNK, len(files) )
            print(f'Processing files {a} thru {b - 1} in groups of {READ_CT}...')
        
            # => INNER CHUNKS
            for k in range(a,b,READ_CT):
                # READ FILE
                df = [read_file(files[k])]
                print(k,', ',end = '')
                # print("i: ")
                for i in range(1, READ_CT): 
                    if k+i<b: df.append( read_file(files[k+i]) )
                    print(k+i, end=' ')
                print()

                df = cudf.concat(df, ignore_index=True,axis=0)
                
                df = fn_heart(df, from_ = PART * SIZE, to_ = (PART+1) *SIZE)

                # COMBINE INNER CHUNKS
                if k == a: tmp2 = df
                else: tmp2 = tmp2.add(df, fill_value = 0)
            print()
            # COMBINE OUTER CHUNKS
            if a == 0: tmp = tmp2
            else: tmp = tmp.add(tmp2, fill_value = 0)
            del tmp2, df
            gc.collect()
            
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending = [True,False])
        
        # SAVE TOP 
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()

        # tmp = tmp.loc[tmp.n < save_top].drop('n',axis=1)

        tmp = tmp.loc[tmp.n < save_top].drop('n',axis=1)
        
        # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(f'exp2/top_{save_top}_{output_name}_v{VER}_{PART}.pqt')


## Validation  ---------------
if args.mode =='valid':
    # CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
    data_cache = {}
    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    files = glob.glob('./data/split_chunked_parquet/*_parquet/*')

    for f in files: data_cache[f] = read_file_to_cache(f)

    # CHUNK PARAMETERS
    READ_CT = 5
    CHUNK = int( np.ceil( len(files) / 6 ))
    print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')
    print('gen carts order')
    # carts order
    create_covisitation(fn_heart=heart_carts_order, DISK_PIECES=4, save_top = 15, output_name = "valid_carts_orders")

    print('gen buy 2 buy')
    # buy2buy
    create_covisitation(fn_heart=heart_buy2buy, DISK_PIECES = 1, save_top = 15, output_name = "valid_buy2buy")


    print('gen click')
    # clicks
    create_covisitation(fn_heart=heart_clicks, DISK_PIECES = 4, save_top = 20, output_name = "valid_clicks")
    del data_cache
    gc.collect()

## Test  ---------------
elif args.mode == 'test':

    data_cache = {}
    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    files = glob.glob('./data/chunked_parquet/*_parquet/*')
    for f in files: data_cache[f] = read_file_to_cache(f)

    # CHUNK PARAMETERS
    READ_CT = 5
    CHUNK = int( np.ceil( len(files) / 6 ))
    print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')



    create_covisitation(fn_heart = heart_carts_order, DISK_PIECES = 4, save_top = 15, output_name = "test_carts_orders")

    create_covisitation(fn_heart = heart_buy2buy, DISK_PIECES = 1, save_top = 15, output_name = "test_buy2buy")

    create_covisitation(fn_heart = heart_clicks, DISK_PIECES = 4, save_top = 20, output_name = "test_clicks")

