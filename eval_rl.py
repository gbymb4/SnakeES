import os
import argparse
import random

import numpy as np
import torch
import yaml

from builder import Builder


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    freq = np.arange(0, 3000, 20, dtype=int)
    for fr in freq:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str,
                            default="logs/workflow_scheduling-v2/20220318164126406127/profile.yaml")
        parser.add_argument("--policy-path", type=str,
                            default=f'logs/workflow_scheduling-v2/20220318164126406127/saved_models/ep_{fr}.pt',
                            help='saved model directory')
        parser.add_argument("--rms-path", type=str,
                            default=f'logs/workflow_scheduling-v2/20220318164126406127/saved_models/ob_rms_{fr}.pickle',
                            help='saved run-time mean and std directory')
        parser.add_argument('--processor-num', type=int, default=16,
                            help='Specify processor number for multiprocessing')
        parser.add_argument('--eval-ep-num', type=int, default=5, help='Set evaluation number per iteration')
        parser.add_argument("--log", action="store_true", help="Use log")
        parser.add_argument("--save-gif", action="store_true")
        parser.add_argument('--save-model-freq', type=int, default=20, help='Save model every a few iterations')
        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        if args.save_gif:
            run_num = args.ckpt_path.split("/")[-3]
            save_dir = f"test_gif/{run_num}/"
            os.makedirs(save_dir)

        Builder(args, config).build().eval()


if __name__ == "__main__":
    main()
