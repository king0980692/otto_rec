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

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


class FeatureGen:
    def __init__(
        self,
        pop_path: Path,
        knn_paths: List[Path],
        embed_paths: List[Path],
        top_path: Path,
    ):
        self.pop = self.load_pop(
            path = pop_path
        )
        self.top = self.load_top(
            path = top_path,
        )
        self.knn = self.load_knn(
            paths = knn_paths,
        )
        self.embeds = self.load_embeds(
            paths = embed_paths,
        )
        logger.info('pre-compute top candidates')
        self.top_candidates = set()
        for candidates in self.top.values():
            self.top_candidates = self.top_candidates | set(candidates)

        logger.info('pre-compute knn candidates')
        self.candidates = defaultdict(set)
        for name in self.knn:
            logger.info('Merge KNN from')
            logger.info(name)
            for tps in self.knn[name]:
                logger.info(tps)
                for aid1 in tqdm(self.knn[name][tps], total=len(self.knn[name][tps])):
                    for aid2 in self.knn[name][tps][aid1]:
                        if self.knn[name][tps][aid1][aid2] > 1.:
                            self.candidates[aid1].add(aid2)

    @staticmethod
    def load_pop(
        path: Path,
    ):
        pop = {tp: {} for tp in ['clicks', 'carts', 'orders']}
        logger.info('load pop from')
        logger.info(path)
        with path.open('r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                for event in data['events']:
                    tp = event['type']
                    aid = str(event['aid'])
                    if aid not in pop[tp]:
                        pop[tp][aid] = 0.
                    pop[tp][aid] += 1.
        return pop

    @staticmethod
    def load_top(
        path: Path,
    ):
        top = {}
        with path.open('r') as f:
            for line in f:
                tp, rids = line.rstrip('\n').split('\t')
                top[tp] = rids.split(' ')
        return top

    @staticmethod
    def load_knn(
        paths: List[Path],
    ):
        knn = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.))))
        for path in paths:
            logger.info('Load from')
            logger.info(path)
            with path.open('r') as f:
                name = '.'.join(path.name.split('.')[:-3])
                tp1 = path.name.split('.')[-3]
                tp2 = path.name.split('.')[-2]
                tps = f'{tp1}_{tp2}'
                logger.info(tps)
                for line in tqdm(f):
                    aid, rid_scores = line.rstrip('\n').split('\t')
                    rid_scores = [rs.split(':') for rs in rid_scores.split(' ')]
                    for rid, score in rid_scores:
                        knn[name][tps][aid][rid] += float(score)

        knn = default_to_regular(knn)
        return knn


    @staticmethod
    def load_embeds(
        paths: List[Path],
    ):
        embeds = {}
        for path in paths:
            logger.info('Load from')
            logger.info(path)
            dim = 0
            embeds[path.name] = {}
            with path.open('r') as f:
                for line in tqdm(f):
                    ent, embed = line.rstrip('\n').split('\t')
                    if 's' in ent:
                        continue
                    embeds[path.name][ent] = np.array(embed.split(' '), dtype='float32')
                    dim = embeds[path.name][ent].shape[0]
            embeds[path.name]['COLD'] = np.zeros(dim, dtype='float32')
        return embeds


    def gen_features_counter(
        self,
        events,
        candidates,
    ):
        depth_range = [1, 5, 10, 20, 40]
        tp_range = ['clicks', 'carts', 'orders']

        # history features
        counter = defaultdict(dict)
        for depth in depth_range:
            for tp in tp_range:
                counter[depth][tp] = Counter(
                    [str(event['aid']) for event in events[-depth:] if event['type'] == tp]
                )
        return [
            [
                len(events)
            ]
            +
            [
                counter[depth][tp][cid] if cid in counter[depth][tp]
                    else 0.
                        for depth in depth_range
                            for tp in tp_range
            ]
            +
            [
                self.pop[tp][cid] if cid in self.pop[tp] else -1 for tp in self.pop
            ]
            for cid in candidates
        ]


    def gen_features_embed(
        self,
        events,
        candidates,
    ):
        X = []
        for key in self.embeds:
            query_embed = self.embeds[key]['COLD']
            candidate_embeds = [
                self.embeds[key][cid] if cid in self.embeds[key]
                    else self.embeds[key]['COLD']
                        for cid in candidates
            ]
            for idx in range(2):
                aid = 'COLD'
                if idx < len(events) and str(events[-idx]['aid']) in self.embeds[key]:
                    aid = str(events[-idx]['aid'])
                query_embed = self.embeds[key][aid]
                features = np.sum(query_embed*candidate_embeds, axis=1).reshape(-1, 1)
                X.append(features)
        return np.hstack(X)


    def gen_features_knn(
        self,
        events,
        candidates,
    ):
        X = []
        # knn features
        tp2onehot = {
            '-': [0, 0, 0],
            'clicks': [1, 0, 0],
            'carts': [0, 1, 0],
            'orders': [0, 0, 1],
        }
        for idx in range(2):
            aid, tp1 = '-1', '-'
            if idx < len(events):
                aid = str(events[-idx]['aid'])
                tp1 = events[-idx]['type']
            X.append(
                [
                    tp2onehot[tp1]
                    +
                    [
                        self.knn[name][tps][aid][cid]
                            if aid in self.knn[name][tps] and cid in self.knn[name][tps][aid]
                                else 0
                                    for name in self.knn
                                        for tps in self.knn[name]
                    ]
                    for cid in candidates
                ]
            )
        return np.hstack(X)


    # events, candidates, labels
    def gen_candidates_labels(
        self,
        events,
        events_future,
        tps: List[str],
    ):
        ## answers
        answers = {tp:{} for tp in tps}
        for tp in answers:
            if tp == 'clicks':
                answers[tp] = [
                    str(event['aid']) for event in events_future if event['type']==tp
                ][:3]
                answers[tp] = {cid:1 for cid in answers[tp]}
            else:
                answers[tp] = {
                    str(event['aid']):1 for event in events_future if event['type']==tp
                }

        ## candidates
        candidates = self.top_candidates.copy()
        candidates = candidates | set(str(event['aid']) for event in events)
        for event in events[-3:]:
            aid = f"{event['aid']}"
            candidates = candidates | self.candidates[aid]
        candidates = list(candidates)

        ## labels
        labels = {tp:[] for tp in answers}
        for tp in labels:
            labels[tp] = [1 if cid in answers[tp] else 0 for cid in candidates]

        return candidates, labels


class Ranker:

    def __init__(
        self,
        feature_gen: FeatureGen,
        gbms: Dict[str, lgb.LGBMRanker] = {},
        tps: List[str] = ['clicks', 'carts', 'orders'],
    ):
        self.gen_features = {
            'counter': feature_gen.gen_features_counter,
            'embed': feature_gen.gen_features_embed,
            'knn': feature_gen.gen_features_knn,
        }
        self.tps = tps
        self.feature_gen = feature_gen
        self.gbms = defaultdict(
            lambda:
                lgb.LGBMRanker(
                    boosting_type='gbdt',
                    objective="lambdarank",
                    metric="map",
                    max_depth=8,
                    num_leaves=110,
                    learning_rate=0.05,
                    n_estimators=500,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    num_threads=16
                )
        )
        if gbms:
            self.gbms = gbms


    def dump_data(
        self,
        jsonl: Path,
        data_path: Path,
        random_cut: bool = False,
    ):
        data_path.mkdir(parents=True, exist_ok=True)

        events, candidates, labels, groups = \
            defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        with jsonl.open('r') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                data = json.loads(line)
                events_given, events_future = data['events'], []
                if random_cut:
                    cut_off = random.randint(1, len(data['events'])-1)
                    events_given = data['events'][:cut_off]
                    events_future = data['events'][cut_off:]
                    _candidates, _labels = self.feature_gen.gen_candidates_labels(
                        events = events_given,
                        events_future = events_future,
                        tps = self.tps,
                    )
                    for tp in self.tps:
                        if any(_labels[tp]):
                            events[tp].append(events_given)
                            candidates[tp].append(_candidates)
                            labels[tp] += _labels[tp]
                            groups[tp].append(len(_candidates))
                else:
                    _candidates, _labels = self.feature_gen.gen_candidates_labels(
                        events = events_given,
                        events_future = [],
                        tps = self.tps,
                    )
                    for tp in self.tps:
                        events[tp].append(events_given)
                        candidates[tp].append(_candidates)
                        labels[tp] += _labels[tp]
                        groups[tp].append(len(_candidates))

        for tp in self.tps:
            logger.info('Transform data')
            logger.info(tp)
            events[tp] = np.array(events[tp])
            candidates[tp] = np.array(candidates[tp])
            labels[tp] = np.array(labels[tp], dtype='int')
            groups[tp] = np.array(groups[tp], dtype='int')
            logger.info('detected events length')
            logger.info(len(events[tp]))

            logger.info('Save to')
            logger.info(data_path/f"{tp}")
            with (data_path/f"{tp}").open('wb') as f:
                np.save(f, events[tp])
                np.save(f, candidates[tp])
                np.save(f, labels[tp])
                np.save(f, groups[tp])


    def dump_features(
        self,
        data_path: Path,
        feature_name: str,
    ):
        for tp in self.tps:
            logger.info('load data from')
            logger.info(data_path/f"{tp}")
            with (data_path/f"{tp}").open('rb') as f:
                events = np.load(f, allow_pickle=True)
                candidates = np.load(f, allow_pickle=True)
                labels = np.load(f, allow_pickle=True)
                groups = np.load(f, allow_pickle=True)

            logger.info('feature size')
            shape_0 = sum(groups)
            shape_1 = len(self.gen_features[feature_name](events[0], candidates[0])[0])
            features = np.empty((shape_0, shape_1), dtype='float32')
            logger.info(shape_0)
            logger.info('x')
            logger.info(shape_1)

            logger.info('save features to')
            logger.info(data_path/f"{tp}.{feature_name}")
            with (data_path/f"{tp}.{feature_name}").open('wb') as f:
                position = 0
                for idx, group in tqdm(enumerate(groups), total=len(groups)):
                    features[position:position+group] = \
                        self.gen_features[feature_name](
                            events = events[idx],
                            candidates = candidates[idx],
                        )
                    position += group
                np.save(f, features)


    def train_gbm_model(
        self,
        train: Path,
        valid: Path,
        feature_names: List[str],
        model_folder: Path,
    ):

        for tp in self.tps:
            logger.info('Load from')
            logger.info(train/tp)
            train_features, valid_features = [], []
            with (train/tp).open('rb') as f:
                _ = np.load(f, allow_pickle=True)
                _ = np.load(f, allow_pickle=True)
                train_labels = np.load(f, allow_pickle=True)
                train_groups = np.load(f, allow_pickle=True)
            for fname in feature_names:
                logger.info(train/f"{tp}.{fname}")
                with (train/f"{tp}.{fname}").open('rb') as f:
                    train_features.append(np.load(f, allow_pickle=True))

            valid_features, valid_labels = [], []
            logger.info(valid/tp)
            with (valid/tp).open('rb') as f:
                _ = np.load(f, allow_pickle=True)
                _ = np.load(f, allow_pickle=True)
                valid_labels = np.load(f, allow_pickle=True)
                valid_groups = np.load(f, allow_pickle=True)
            for fname in feature_names:
                logger.info(valid/f"{tp}.{fname}")
                with (valid/f"{tp}.{fname}").open('rb') as f:
                    valid_features.append(np.load(f, allow_pickle=True))

            train_features = np.hstack(train_features)
            valid_features = np.hstack(valid_features)
            print(train_features.shape)
            print(train_labels.shape)
            print(train_groups.shape)

            logger.info('Train lgb')
            self.gbms[tp].fit(
                X=train_features,
                y=train_labels,
                group=train_groups,
                eval_set=[(train_features, train_labels), (valid_features, valid_labels)],
                eval_group=[train_groups, valid_groups],
                eval_at=100,
                verbose=5,
                callbacks=[lgb.early_stopping(stopping_rounds=30)]
            )
            model_path = model_folder / f"lgb.{tp}"
            logger.info('save model to')
            logger.info(model_path)
            self.gbms[tp].booster_.save_model(
                str(model_path),
                num_iteration = self.gbms[tp].best_iteration_
            )


    def make_submit(
        self,
        jsonls: List[Path],
        data_path: Path,
        feature_names: List[str],
        submit_path: Path,
    ):
        submit_path.parent.mkdir(parents=True, exist_ok=True)

        submits = ['session_type,labels']
        for jsonl in jsonls:
            sessions = []
            name = jsonl.name
            logger.info('load sessions from')
            logger.info(jsonl)
            with jsonl.open('r') as f:
                for line in f:
                    data = json.loads(line)
                    sessions.append(data['session'])

            for tp in self.tps:
                features, groups, candidates = [], [], []
                logger.info('load features from')
                path = data_path/name/tp
                logger.info(path)
                with path.open('rb') as f:
                    _ = np.load(f, allow_pickle=True)
                    candidates = np.load(f, allow_pickle=True)
                    _ = np.load(f, allow_pickle=True)
                    groups = np.load(f, allow_pickle=True)
                for fname in feature_names:
                    with Path(f"{path}.{fname}").open('rb') as f:
                        features.append(np.load(f, allow_pickle=True))
                features = np.hstack(features)

                logger.info('Compute top recommendations')
                preds = self.gbms[tp].predict(features, num_threads=16)
                idx = 0
                for candidate, session in tqdm(zip(candidates, sessions), total=len(sessions)):
                    g_len = len(candidate)
                    scores = {cid:p for cid, p in zip(candidate, preds[idx:idx+g_len])}
                    rec_ids = sorted(scores, key=scores.get, reverse=True)[:20]
                    rec_ids = ' '.join(rec_ids)
                    submits.append(f"{session}_{tp},{rec_ids}")
                    idx += g_len
        logger.info('make submit to')
        logger.info(submit_path)
        with submit_path.open('w') as f:
            f.write('\n'.join(submits))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--given_jsonl', help='path to given.jsonl')
    parser.add_argument('--train_jsonl', help='path to train.jsonl')
    parser.add_argument('--valid_jsonl', help='path to valid.jsonl')
    parser.add_argument('--test_jsonls', nargs='+', help='path to test.jsonl')
    parser.add_argument('--train_path', default=None, help='path to saving train data')
    parser.add_argument('--valid_path', default=None, help='path to loading valid data')
    parser.add_argument('--test_path', default=None, help='path to loading test data')
    parser.add_argument('--embed_paths', nargs='+', default=None, help='path to embeddings')
    parser.add_argument('--feature_gen', default=None, help='path to saving/loading feature pickle')
    parser.add_argument('--feature_names', nargs='+', help='feature name as surffix')
    parser.add_argument('--knn_paths', nargs='+', default=None, help='path to knn')
    parser.add_argument('--top_path', default=None, help='path to knn')
    parser.add_argument('--model_folder', help='folder for saving models')
    parser.add_argument('--submit_path', help='path for saving models')
    args = parser.parse_args()

    feature_gen = None
    if Path(args.feature_gen).is_file():
        logger.info('load feature pickle from')
        logger.info(args.feature_gen)
        with Path(args.feature_gen).open('rb') as f:
            feature_gen = pickle.load(f)
    else:
        feature_gen = FeatureGen(
            pop_path = Path(args.train_jsonl),
            top_path = Path(args.top_path),
            knn_paths = [Path(path) for path in args.knn_paths],
            embed_paths = [Path(path) for path in args.embed_paths],
        )
        if args.save_feature_gen:
            logger.info('save feature pickle to')
            logger.info(args.feature_gen)
            with Path(args.feature_gen).open('wb') as f:
                pickle.dump(feature_gen, f)

    ranker = Ranker(
        feature_gen = feature_gen
    )
    '''
    ranker.dump_data(
        jsonl = Path(args.train_jsonl),
        data_path = Path(args.train_path),
        random_cut = True
    )
    ranker.dump_data(
        jsonl = Path(args.valid_jsonl),
        data_path = Path(args.valid_path),
        random_cut = True
    )
    for test_jsonl in args.test_jsonls:
        name = Path(test_jsonl).name
        ranker.dump_data(
            jsonl = Path(test_jsonl),
            data_path = Path(args.test_path)/name,
            random_cut = False
        )
    '''
    for feature_name in args.feature_names:
        ranker.dump_features(
            data_path = Path(args.train_path),
            feature_name = feature_name,
        )
        ranker.dump_features(
            data_path = Path(args.valid_path),
            feature_name = feature_name,
        )
        for test_jsonl in args.test_jsonls:
            name = Path(test_jsonl).name
            ranker.dump_features(
                data_path = Path(args.test_path)/name,
                feature_name = feature_name,
            )
    exit()
    ranker.train_gbm_model(
        train = Path(args.train_path),
        valid = Path(args.valid_path),
        feature_names = args.feature_names,
        model_folder = Path(args.model_folder),
    )
    '''
    gbms = {}
    for path in Path(args.model_folder).iterdir():
        gbms[path.suffix[1:]] = lgb.Booster(model_file=str(path))
    ranker = Ranker(
        feature_gen = feature_gen,
        gbms = gbms,
    )
    ranker.make_submit(
        jsonls = [Path(jsonl) for jsonl in args.test_jsonls],
        data_path = Path(args.test_path),
        feature_names = args.feature_names,
        submit_path = Path(args.submit_path),
    )
    '''
