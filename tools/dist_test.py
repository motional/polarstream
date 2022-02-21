import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel
import pickle
import time 

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # np.random.seed(0)

    args = parse_args()
    #sys.path.insert(0, os.path.abspath(args.work_dir))

    from det3d import torchie
    from det3d.datasets import build_dataloader, build_dataset
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis import (
        batch_processor,
        get_root_logger,
    )
    
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    from det3d.torchie.trainer import get_dist_info, load_checkpoint
    from det3d.torchie.trainer.utils import all_gather, synchronize
    
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus
    if cfg.local_rank == 0:
        # init logger before other steps
        logger = get_root_logger(cfg.log_level)
        logger.info("Distributed testing: {}".format(distributed))
        logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        if args.testset:
            print("Use Test Set")
        else:
            print("Use Val Set")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        dataset = build_dataset(cfg.data.test)
    else:
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        #model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    model.eval()
    mode = "val"

    if args.local_rank == 0:
        logger.info(f"work dir: {args.work_dir}")
        if 'det' in cfg.super_tasks:
            det_prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)
        if 'seg' in cfg.super_tasks:
            seg_prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    segmentation = {}
    instances = {}
    cpu_device = torch.device("cpu")


    start = int(len(data_loader) / 3)
    end = int(len(data_loader) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.perf_counter()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        if 'det' in cfg.super_tasks:
            for output in outputs['det']:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output, }
                )
                if args.local_rank == 0:
                    det_prog_bar.update()
        if 'seg' in cfg.super_tasks:
            for output in outputs['seg']:
                for token, preds in output.items():
                    segmentation.update({token: preds.cpu().numpy()})
                if args.local_rank == 0:
                    seg_prog_bar.update()
        if 'ins' in outputs:
            for output in outputs['ins']:
                for token, preds in output.items():
                    instances.update({token: preds.cpu().numpy()})

    synchronize()

    all_dets = all_gather(detections)
    all_segs = all_gather(segmentation)
    all_ins = all_gather(instances)
    if args.local_rank == 0:
        print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    final_dets = {}
    for p in all_dets:
        final_dets.update(p)
    final_segs = {}
    for p in all_segs:
        final_segs.update(p)
    final_ins = {}
    for p in all_ins:
        final_ins.update(p)


    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(final_dets, args.work_dir)

    det_dict, seg_dict, pan_dict = dataset.evaluation(copy.deepcopy(final_dets), final_segs, final_ins, output_dir=args.work_dir, testset=args.testset)
    if det_dict is not None:
        for k, v in det_dict["results"].items():
            print(f"Evaluation {k}: {v}")
    if seg_dict is not None:
        for key, val in seg_dict.items():
            if key == "iou_per_class":
                for k, v in seg_dict[key].items():
                    print(f"iou_{k}: {v}")
            else:
                print(f"{key}: {val}")
        

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
