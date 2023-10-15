from training import train, evaluate
from utils import create_argparser
import logging
import os
from braindecode.util import set_random_seeds
import yaml
import argparse
import torch.distributed as dist
import torch.nn.parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml')
    args, unknown = parser.parse_known_args()
    config = yaml.safe_load(open(args.config_file))
    args = create_argparser(config).parse_args(unknown)
    args.local_rank = int(os.environ["LOCAL_RANK"]) if args.use_ddp and os.environ.get("LOCAL_RANK") else 0

    log_dir = os.path.join('logs', args.dataset)
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    logging.basicConfig(handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(log_dir, 'log.txt'))],
                        level=logging.INFO)
    logging.info(args) if args.local_rank == 0 else None

    set_random_seeds(seed=15485485, cuda=True)
    if args.use_ddp and args.train:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    train(args) if args.train else None
    evaluate(args)


if __name__ == "__main__":
    main()
