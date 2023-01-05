import argparse, logging
import json, pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def swing2knn(
    input_path: Path,
    knn_path: Path,
    top_k: int = 100,
):
    scores = defaultdict(dict)
    logger.info('Load from')
    logger.info(input_path)
    with input_path.open('r') as f:
        for line in tqdm(f):
            aid1, aid2, value = line.rstrip('\n').split('\t')
            scores[aid1][aid2] = float(value)

    knns = []
    logger.info('Save to')
    logger.info(knn_path)
    with knn_path.open('w') as f:
        for aid1 in tqdm(scores, total=len(scores)):
            orders = sorted(scores[aid1], key=scores[aid1].get, reverse=True)[:50]
            orders = [f"{aid2}:{scores[aid1][aid2]}" for aid2 in orders]
            orders = ' '.join(orders)
            knns.append(f"{aid1}\t{orders}")
        f.write('\n'.join(knns))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--swing', help='path to swing.score')
    parser.add_argument('--knn', default=None, help='path to saving knn')
    args = parser.parse_args()

    swing2knn(
        input_path = Path(args.swing),
        knn_path = Path(args.knn),
    )

