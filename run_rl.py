import argparse
import random

import numpy as np
import torch
import yaml

from builder import Builder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--config', type=str,
                        # default='config/lunarlander_ga_uberai.yaml',
                        default='config/workflow_scheduling_es_openai.yaml',
                        # default='config/fisherymanagement_es_openai.yaml',
                        # default='config/cartpole_ga_uberai.yaml',
                        help='A config path for env,  policy, and optim')
    parser.add_argument('--processor-num', type=int, default=16, help='Specify processor number for multiprocessing')
    parser.add_argument('--eval-ep-num', type=int, default=1, help='Set evaluation number per iteration')
    # Settings related to logs
    parser.add_argument("--log", action="store_true", help="Use log")
    parser.add_argument('--save-model-freq', type=int, default=20, help='Save model every a few iterations')

    # Overwrite some common values in YAML with command-line options, if needed.
    parser.add_argument('--seed', type=int, default=None, help='Replace seed value in  YAML')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # Replace seed value if command-line options on seed is not None
    if args.seed is not None:
        config['env']['seed'] = args.seed

    # Set global running seed
    set_seed(config['env']['seed'])

    # Start assembling RL and training process
    Builder(args, config).build().train()


if __name__ == "__main__":
    main()
