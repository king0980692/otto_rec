import argparse, logging
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def load_tops(
    path: Path,
    logger: logging.Logger,
):
    logger.info('Load from')
    logger.info(path)
    tops = {}
    with path.open('r') as f:
        for line in f:
            tp, aids = line.rstrip('\n').split('\t')
            tops[tp] = aids.split(' ')
    return tops

def load_knn_tp(
    paths: List[Path],
    logger: logging.Logger,
):
    knn_tp = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.))))
    for path in paths:
        logger.info('load from')
        logger.info(path)
        tp1, tp2 = path.name.split('.')[-3], path.name.split('.')[-2]
        logger.info(tp1)
        logger.info(tp2)
        with path.open('r') as f:
            for line in tqdm(f):
                aid, rid_scores = line.rstrip('\n').split('\t')
                rid_scores = [rs.split(':') for rs in rid_scores.split(' ')]
                for rid, score in rid_scores:
                    knn_tp[tp1][tp2][aid][rid] += float(score)
    return knn_tp


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
            aid = int(entity)
            aid2index[aid] = len(aid2index)
            aids.append(entity)
            embeds.append(embed.split(' '))
    embeds = np.array(embeds, dtype='float32')
    return aids, embeds, aid2index


def make_knn_submit(
    test_jsonl: Path,
    submit_path: Path,
    knn_tp,
    tops,
    logger: logging.Logger,
):
    logger.info('read from')
    logger.info(test_jsonl)
    predicitons = ['session_type,labels']

    sessions, query_embeds, observed = [], [], []
    with test_jsonl.open('r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            session = data['session']
            scores = defaultdict(lambda: defaultdict(lambda: 0.))
            observed = defaultdict(lambda: 0.)

            all_counter = Counter([str(event['aid']) for event in data['events']])
            recent_counter = Counter([str(event['aid']) for event in data['events']])

            # all
            for event in data['events']:
                aid1 = str(event['aid'])
                tp1 = event['type']
                for tp2 in ['clicks', 'carts', 'orders']:
                    if aid1 in knn_tp[tp1][tp2]:
                        for rid in knn_tp[tp1][tp2][aid1]:
                            scores[tp2][rid] += knn_tp[tp1][tp2][aid1][rid]

            # latest
            for event in data['events'][-3:]:
                aid1 = str(event['aid'])
                tp1 = event['type']
                for tp2 in ['clicks', 'carts', 'orders']:
                    if aid1 in knn_tp[tp1][tp2]:
                        for rid in knn_tp[tp1][tp2][aid1]:
                            scores[tp2][rid] += knn_tp[tp1][tp2][aid1][rid]
                    if tp1 in ['carts', 'orders']:
                        scores[tp2][aid1] += 10.

            # frequent
            for rid in all_counter:
                if all_counter[rid] > 3.:
                    for tp2 in ['clicks', 'carts', 'orders']:
                        scores[tp2][rid] += 10. * observed[rid]
            for rid in recent_counter:
                if recent_counter[rid] > 2.:
                    for tp2 in ['clicks', 'carts', 'orders']:
                        scores[tp2][rid] += 10. * observed[rid]

            for tp in ['clicks', 'carts', 'orders']:
                recs = []
                if tp in scores:
                    recs = list(sorted(scores[tp], key=scores[tp].get, reverse=True))[:20]
                while len(recs) < 20:
                    for rid in tops[tp]:
                        if rid not in recs:
                            recs.append(rid)

                recs = ' '.join(recs)
                predicitons.append(f"{session}_{tp},{recs}")

    logger.info('save to')
    logger.info(submit_path)
    with submit_path.open('w') as f:
        f.write('\n'.join(predicitons))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--test_jsonl', help='path to train.jsonl')
    parser.add_argument('--embed_path', default=None, help='path to embeddings')
    parser.add_argument('--i2i_path', default=None, help='path to i2i scores')
    parser.add_argument('--knn_tp_paths', nargs='+', help='path to knn files')
    parser.add_argument('--top_path', help='path to top-knn file')
    parser.add_argument('--submit_path', help='save to ...')
    args = parser.parse_args()

    tops = load_tops(
        path = Path(args.top_path),
        logger = logger,
    )
    knn_tp_paths = [Path(path) for path in args.knn_tp_paths]
    knn_tp = load_knn_tp(
        paths = knn_tp_paths,
        logger = logger,
    )
    make_knn_submit(
        test_jsonl = Path(args.test_jsonl),
        submit_path = Path(args.submit_path),
        knn_tp = knn_tp,
        tops = tops,
        logger = logger,
    )

