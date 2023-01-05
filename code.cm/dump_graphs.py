import argparse, logging
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def dump_graphs(
    train_jsonl: Path,
    save_path: Path,
    skip_lines: int,
    logger: logging.Logger,
):
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info('load from')
    logger.info(train_jsonl)
    session2aid, session2aid_tp, session2aid_w10  = [], [], []
    with train_jsonl.open('r') as f:
        for _ in tqdm(range(skip_lines)):
            next(f)
        for line in tqdm(f):
            data = json.loads(line)
            session = data['session']
            aids = [event['aid'] for event in data['events']]
            aid_tps = [f"{event['aid']}-{event['type']}" for event in data['events']]
            counter = Counter(aids)
            for aid in counter:
                session2aid.append(f"s{session}\t{aid}\t{counter[aid]}")
            counter = Counter(aid_tps)
            for aid_tp in counter:
                session2aid_tp.append(f"s{session}\t{aid_tp}\t{counter[aid_tp]}")

            start = 0
            end = start + 10
            window = 0
            while (start < len(aids)):
                counter = Counter(aids[start:end])
                for aid in counter:
                    session2aid_w10.append(f"s{session}-w{window}\t{aid}\t{counter[aid]}")
                start += 5
                end += 5
                window += 1

    logger.info('save to')
    logger.info(save_path/'s2a.tsv')
    with (save_path/'s2a.tsv').open('w') as f:
        f.write('\n'.join(session2aid))
        f.write('\n')

    logger.info('save to')
    logger.info(save_path/'s2a_tp.tsv')
    with (save_path/'s2a_tp.tsv').open('w') as f:
        f.write('\n'.join(session2aid_tp))
        f.write('\n')

    logger.info('save to')
    logger.info(save_path/'s2a_w10.tsv')
    with (save_path/'s2a_w10.tsv').open('w') as f:
        f.write('\n'.join(session2aid_w10))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--train_jsonl', help='path to train.jsonl')
    parser.add_argument('--save_path', help='save as ...')
    parser.add_argument('--skip_lines', type=int, default=6000000, help='save as ...')
    args = parser.parse_args()

    dump_graphs(
        train_jsonl = Path(args.train_jsonl),
        save_path = Path(args.save_path),
        skip_lines = args.skip_lines,
        logger = logger
    )
