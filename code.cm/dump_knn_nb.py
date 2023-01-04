import argparse, logging
import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import numba as nb
from numba import njit
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def dump_knn(
    knn,
    path: Path,
    logger,
):
    logger.info('save to')
    logger.info(f"{path}")
    with path.open('w') as f:
        for key in tqdm(knn, total=len(knn)):
            nn = ' '.join(knn[key])
            line = f"{key}\t{nn}\n"
            f.write(line)

def load_json_list(
    jsonl: Path,
    skip_lines: int,
):
    json_list = []
    with jsonl.open('r') as f:
        for _ in tqdm(range(skip_lines)):
            next(f)
        for line in tqdm(f):
            json_list.append(json.loads(line))
        return json_list

@njit
def get_i2i(
    json_list,
):
    s = 0
    for data in json_list:
        data = json.loads(line)
        s += len(data['events'])
    print(s)
    '''
    events = data['events']
    session = data['session']
    session_len = len(data['events'])
    for idx1 in range(session_len):
        tp1 = events[idx1]['type']
        aid1 = events[idx1]['aid']
        for idx2 in range(session_len):
            tp2 = events[idx2]['type']
            if idx1 == idx2:
                continue
            aid2 = events[idx2]['aid']
            if (aid1, aid2) not in i2i_session[tp1][tp2]:
                i2i_session[tp1][tp2] = 0
            i2i_next[tp1][tp2][(aid1, aid2)] += 1.
            if idx1 < idx2:
                if (aid1, aid2) not in i2i_next[tp1][tp2]:
                    i2i_next[tp1][tp2] = 0
                i2i_session[tp1][tp2][(aid1, aid2)] += 1.
    '''


def compute_i2i(
    jsonl: Path,
    skip_lines: int,
    logger: logging.Logger,
):
    with jsonl.open('r') as f:
        for _ in tqdm(range(skip_lines)):
            next(f)
        for line in tqdm(f):
            data = json.loads(line)
            events = data['events']
            session = data['session']
            session_len = len(data['events'])
            for idx1 in range(session_len):
                tp1 = events[idx1]['type']
                aid1 = events[idx1]['aid']
                for idx2 in range(session_len):
                    tp2 = events[idx2]['type']
                    if idx1 == idx2:
                        continue
                    aid2 = events[idx2]['aid']
                    if idx1 < idx2:
                        i2i_next[tp1][tp2][aid1][aid2] += 1.
                    i2i_session[tp1][tp2][aid1][aid2] += 1.

    knn_session = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    knn_next = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tp1 in i2i_session:
        for tp2 in i2i_session[tp1]:
            logger.info(tp1)
            logger.info(tp2)
            for aid1 in tqdm(i2i_session[tp1][tp2], total=len(i2i_session[tp1][tp2])):
                knn_session[tp1][tp2][aid1] = [
                    f"{aid2}:{i2i_session[tp1][tp2][aid1][aid2]}"
                        for aid2 in sorted(
                            i2i_session[tp1][tp2][aid1],
                            key=i2i_session[tp1][tp2][aid1].get,
                            reverse=True
                        )[:100]
                ]
            for aid1 in tqdm(i2i_next[tp1][tp2], total=len(i2i_next[tp1][tp2])):
                knn_next[tp1][tp2][aid1] = [
                    f"{aid2}:{i2i_next[tp1][tp2][aid1][aid2]}"
                        for aid2 in sorted(
                            i2i_next[tp1][tp2][aid1],
                            key=i2i_next[tp1][tp2][aid1].get,
                            reverse=True
                        )[:100]
                ]

    return knn_session, knn_next

def load_tops(
    jsonl: Path,
    skip_lines: int,
    logger: logging.Logger,
):
    logger.info('load from')
    logger.info(jsonl)
    pops = {
        'clicks': defaultdict(lambda: 0.),
        'carts': defaultdict(lambda: 0.),
        'orders': defaultdict(lambda: 0.),
    }
    with jsonl.open('r') as f:
        for _ in tqdm(range(skip_lines)):
            next(f)
        for line in tqdm(f):
            data = json.loads(line)
            session = data['session']
            for event in data['events']:
                pops[event['type']][event['aid']] += 1.
    tops = {}
    for tp in pops:
        tops[tp] = sorted(pops[tp], key=pops[tp].get, reverse=True)[:100]
        tops[tp] = list(map(str, tops[tp]))

    logger.info(tops)
    return tops


def load_embeds(
    embed_path: Path,
    logger: logging.Logger,
):
    logger.info('load from')
    logger.info(embed_path)
    aids, embeds, aid2index = [], [], {}
    with embed_path.open('r') as f:
        for line in tqdm(f):
            entity, embed = line.rstrip('\n').split('\t')
            if 's' in entity:
                continue
            aid = entity
            aid2index[aid] = len(aid2index)
            aids.append(entity)
            embeds.append(embed.split(' '))
    embeds = np.array(embeds, dtype='float32')
    index = faiss.IndexFlatIP(len(embeds[0]))
    index.train(embeds)
    index.add(embeds)
    distances, neighbors = index.search(
        np.array(embeds, dtype='float32'),
        100,
    )
    knn = {
        aids[e]: [aids[n] for n in ns]
            for e, ns in enumerate(neighbors)
    }

    return knn


def load_i2i(
    i2i_path: Path,
    leave_topk: int,
    logger: logging.Logger,
):
    logger.info('load from')
    logger.info(i2i_path)
    i2i, knn = defaultdict(dict), {}
    with i2i_path.open('r') as f:
        for line in tqdm(f):
            aid1, aid2, score = line.rstrip('\n').split('\t')
            i2i[aid1][aid2] = float(score)
    for aid1 in tqdm(i2i, total=len(i2i)):
        knn[aid1] = list(sorted(i2i[aid1], key=i2i[aid1].get, reverse=True))[:leave_topk]
    return i2i, knn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--train_jsonl', help='path to train.jsonl')
    parser.add_argument('--test_jsonl', help='path to train.jsonl')
    parser.add_argument('--embed_path', default=None, help='path to embeddings')
    parser.add_argument('--i2i_path', default=None, help='path to i2i scores')
    args = parser.parse_args()

    '''
    knn_tops = load_tops(
        jsonl = Path(args.train_jsonl),
        skip_lines = 6000000,
        logger = logger,
    )
    dump_knn(
        knn_tops,
        path = Path(f"{args.train_jsonl}.knn"),
        logger = logger,
    )
    knn_tops = load_tops(
        jsonl = Path(args.test_jsonl),
        skip_lines = 0,
        logger = logger,
    )
    dump_knn(
        knn_tops,
        path = Path(f"{args.test_jsonl}.knn"),
        logger = logger,
    )
    '''

    i2i_session, i2i_next = {}, {}
    for tp1 in ['clicks', 'carts', 'orders']:
        i2i_session[tp1] = {}
        i2i_next[tp1] = {}
        for tp2 in ['clicks', 'carts', 'orders']:
            i2i_session[tp1][tp2] = {}
            i2i_next[tp1][tp2] = {}

    logger.info('Load')
    logger.info(args.train_jsonl)
    json_list = load_json_list(
        jsonl = Path(args.train_jsonl),
        skip_lines = 10000000,
    )
    #for line in tqdm(lines, total=len(lines)):
    #    data = json.loads(line)
    get_i2i(
        json_list = nb.typed.List(json_list),
    )

    exit()
    for tp1 in knn_session:
        for tp2 in knn_session[tp1]:
            dump_knn(
                knn_session[tp1][tp2],
                path = Path(f"exp/i2i.train.session.{tp1}.{tp2}.knn"),
                logger = logger,
            )
            dump_knn(
                knn_next[tp1][tp2],
                path = Path(f"exp/i2i.train.next.{tp1}.{tp2}.knn"),
                logger = logger,
            )
    knn_session, knn_next = compute_i2i(
        jsonl = Path(args.test_jsonl),
        skip_lines = 0,
        logger = logger,
    )
    for tp1 in knn_session:
        for tp2 in knn_session[tp1]:
            dump_knn(
                knn_session[tp1][tp2],
                path = Path(f"exp/i2i.test.session.{tp1}.{tp2}.knn"),
                logger = logger,
            )
            dump_knn(
                knn_next[tp1][tp2],
                path = Path(f"exp/i2i.test.next.{tp1}.{tp2}.knn"),
                logger = logger,
            )
    exit()


    if args.i2i_path:
        i2i, knn_i2i = load_i2i(
            i2i_path = Path(args.i2i_path),
            leave_topk = 100,
            logger = logger,
        )
        dump_knn(
            knn_i2i,
            path = Path(f"{args.i2i_path}.knn"),
            logger = logger,
        )

    if args.embed_path:
        knn_embed = load_embeds(
            embed_path = Path(args.embed_path),
            logger = logger,
        )
        dump_knn(
            knn_embed,
            path = Path(f"{args.embed_path}.knn"),
            logger = logger,
        )
