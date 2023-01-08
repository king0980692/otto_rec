import pandas as pd


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str , required=True)
parser.add_argument('--model', type=str , required=True)
parser.add_argument('--mode', choices=['train', 'valid', 'test'] , required=True)
args = parser.parse_args()

submission_dfs = []
for st in ['clicks', 'carts', 'orders']:
    df = pd.read_csv(f'{args.pred}/{args.model}_{args.mode}_predictions_{st}.csv')
    df['session_type'] = df['session_type'].astype(str) + f"_{st}"

    submission_dfs.append(df)

submission_dfs = pd.concat(submission_dfs).reset_index(drop=True)
print("Saving Merging csv file into {}".format(f'{args.pred}/{args.model}_submission.csv'))
submission_dfs.to_csv(f'{args.pred}/{args.model}_submission.csv', index=False)

