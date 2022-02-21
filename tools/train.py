import argparse
import json
import os
import sys

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import torch
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", help="the pretrained checkpoint file to load from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    
    args = parse_args()
    assert args.work_dir is not None, 'need to specify work_dir for aws'
    if args.local_rank == 0:
        # copy important files to backup
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
            os.system("cp -r %s %s/" % (args.config, args.work_dir))
            os.system("cp -r det3d %s/" % args.work_dir)
            print(f"Backup source files to {args.work_dir}")
        print(f"Using packages from {os.path.join(args.work_dir, 'det3d')}")

    sys.path.insert(0, os.path.abspath(args.work_dir))
    from det3d.datasets import build_dataset
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis import (
        build_optimizer,
        get_root_logger,
        init_dist,
        set_random_seed,
        train_detector,
    )
    
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if cfg.voxel_generator.get('nsectors', 1) >= 8:
        torch.multiprocessing.set_sharing_strategy('file_system')
    cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if (args.load_from is None) and (args.resume_from is None):
        if os.path.exists(os.path.join(cfg.work_dir, 'latest.pth')):
            cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    
    
    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


if __name__ == "__main__":
    main()
