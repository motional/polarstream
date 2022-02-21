import sys, os, numba
import pickle
import json
import random
import operator
import numpy as np

from pathlib import Path
from xlwt import Workbook
from det3d import torchie
from scipy.spatial import distance_matrix
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.lidarseg.evaluate import LidarSegEval
    from nuscenes.utils.data_io import load_bin_file
    
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from det3d.datasets.registry import DATASETS
THINGS = {
                'barrier': 1,
                'bicycle': 2,
                'bus': 3,
                'car': 4,
                'construction_vehicle': 5,
                'motorcycle': 6,
                'pedestrian': 7,
                'traffic_cone': 8,
                'trailer': 9,
                'truck': 10,
            }

ths = [0.1, 0.1, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        mode='train',
        version="v1.0-trainval",
        **kwargs,
    ):
        """
        Initiate NuScenes dataset class
                :param mode: 'train', 'val' or 'test'
                :param version: 'v1.0-trainval' or 'v1.0-test'
                :param kwargs['super_tasks']: list. set to ['det', 'seg'] for multi tasks. If None, set to ['det']
                :param kwargs['transform_type']: string. 'point': point warp. 'feature': feature warp. If None, set to 'point'
                :param kwargs['nsectors']: int. For 16 and 32 sectors, set nsectors to cut full sweep into several groups of 8 sectors to avoid memory error. Do not set it for bidirectional padding.
                """
        self.mode = mode
        self.nsectors = kwargs.get('nsectors', 1)
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=(mode != 'train'), class_names=class_names
        )
        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps, 'sweeps')

        self._info_path = info_path
        self._class_names = class_names

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.painted = kwargs.get('painted', False)
        if self.painted:
            self._num_point_features += 10 

        self.version = version
        self.eval_version = "detection_cvpr_2019"
        self.super_tasks = kwargs.get('super_tasks', ['det'])
        self.transform_type = kwargs.get('transform_type', 'point')

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

    def __len__(self):
        """
        for 1, 2, 4, 8 sectors, return the true dataset length.
        for self.nsectors > 8, we train 8 sectors at a time. Otherwise there is memory error when there is a large number of tensors.

        An exception is bidirectional context padding. Do not set self.nsectors > 8 because we need full-sweep feature maps.
        The collate_fn will handle this case and store everything from dataloader in numpy first.
        """
        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)
        if (self.mode == 'train') and (self.nsectors > 8):
            ngroups = self.nsectors // 8
            return ngroups * len(self._nusc_infos)
        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx):
        if (self.mode == 'train') and (self.nsectors > 8):
            ngroups = self.nsectors // 8
            group = idx % ngroups
            idx = idx // ngroups
        
        info = self._nusc_infos[idx]
        if (self.mode == 'train') and (self.nsectors > 8):
            info['load_range'] = range(group * 8, (group + 1) * 8)
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
                "transform_type": self.transform_type,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": self.mode,
            "painted": self.painted,
        }
        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, segmentation, instances={}, output_dir=None, testset=False):
        """
                        Evaluate detection, semantic segmentation and panoptic segmentation.
                        :param detections: list of dict. predictions of entire dataset split
                        :param segmentation: {sample_token: prediction <n,> array}. predictions of entire dataset split
                        :param instances: {sample_token: prediction <n,> array}. predictions of entire dataset split. If len(instances)==0, use global fusion.
                        :param output_dir: str. Output directory to store result .xls
                        """
        ret_det = None
        if len(detections)>0:
            ret_det = self.eval_det(detections, output_dir, testset)
        ret_seg = None
        if len(segmentation) > 0:
            ret_seg = self.eval_seg(segmentation, output_dir, testset)
        ret_pan = None
        if len(detections) and len(segmentation) and (not testset):
            ret_pan = self.eval_panoptic(detections, segmentation, instances, output_dir)
        return ret_det, ret_seg, ret_pan
                
    def eval_panoptic(self, detections, segmentation, instances={}, output_dir=None):
        """
                Evaluate panoptic segmentation.
                :param detections: list of dict. predictions of entire dataset split
                :param segmentation: {sample_token: prediction <n,> array}. predictions of entire dataset split
                :param instances: {sample_token: prediction <n,> array}. predictions of entire dataset split. If len(instances)==0, use global fusion.
                :param output_dir: str. Output directory to store result .xls
                """
        from ..utils.panoptic_eval import PanopticEval
        import time
        times = []
        version = 'v1.0-trainval'
        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)
        evalset = 'val'
        os.makedirs(os.path.join(output_dir, 'instance', evalset), exist_ok=True)
        lidar_seg_eval = LidarSegEval(nusc, output_dir, evalset, verbose=True)
        
        pq_eval = PanopticEval(17, ignore=[0], min_points=20)
        for sample_token in lidar_seg_eval.sample_tokens:
            sample = nusc.get('sample', sample_token)
            sd_token = sample['data']['LIDAR_TOP']
            lidarseg_label_filename = os.path.join(nusc.dataroot,
                                                   nusc.get('lidarseg', sd_token)['filename'])
            lidarseg_label = load_bin_file(lidarseg_label_filename)
            lidarseg_label = lidar_seg_eval.mapper.convert_label(lidarseg_label)
            instance_label_filename = lidarseg_label_filename.replace('lidarseg', 'instance_all')
            instance_labels = np.fromfile(instance_label_filename,dtype=int).reshape(lidarseg_label.shape)
            valid = (lidarseg_label > 0) & (instance_labels >= 0)
            lidarseg_label = lidarseg_label[valid]
            instance_labels = instance_labels[valid]
            if sample_token not in segmentation: continue
            lidarseg_pred = segmentation[sample_token].astype(np.uint8)
            lidarseg_pred = lidarseg_pred[valid]
            if len(instances) > 0:
                instance_preds = instances[sample_token]
                instance_preds = instance_preds[valid]
            else:
                instance_preds = np.zeros(instance_labels.shape, dtype=instance_labels.dtype)
            
                lidar_path = nusc.get_sample_data_path(sd_token)
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
                points = points[valid]
                det = detections[sample_token]
                
                def get_instance_preds(pair):
                    k, v = pair
                    seg_mask = lidarseg_pred == v
                    if seg_mask.sum() == 0: return
                    box_label = self._class_names.index(k)
                    bbox_mask = (det['label_preds'] == box_label) & (det['scores'] > 0.3)
                    if bbox_mask.sum() == 0: return
                    bboxes = det['box3d_lidar'][bbox_mask].cpu().numpy()
                    point_id = np.where(seg_mask)[0]
                    cur_points = points[point_id, :2]
                    dist = distance_matrix(cur_points, bboxes[:, :2])
                    bids = np.argmin(dist, axis=1)
                    instance_preds[point_id] = bids
                start = time.time()
                list(map(get_instance_preds, THINGS.items()))
                end = time.time()
                times.append(end-start)
                
            pq_eval.addBatch(lidarseg_pred, instance_preds, lidarseg_label, instance_labels)
            
        
        pq, sq, rq, all_pq, all_sq, all_rq = pq_eval.getPQ()
        iou, all_iou = pq_eval.getSemIoU()
        print(f'PQ: {pq}, SQ: {sq}, RQ: {rq}')
        print(f'all PQ: {all_pq}')
        print(f'all SQ: {all_sq}')
        print(f'all RQ: {all_rq}')
        print(f'iou: {iou}')
        print('panoptic fusion time', 1000 * np.mean(times), 'ms')
        rfile = os.path.join(output_dir, 'pan_metrics.xls')
        wb = Workbook()
        sheet1 = wb.add_sheet('panoptic')
        sheet1.write(1, 0, 'PQ')
        sheet1.write(2, 0, 'SQ')
        sheet1.write(3, 0, 'RQ')
        sheet1.write(1, 1, pq)
        sheet1.write(2, 1, sq)
        sheet1.write(3, 1, rq)
        for i in range(1, 17):
            name = lidar_seg_eval.id2name[i]
            sheet1.write(0, 1+i, name)
            sheet1.write(1, 1+i, all_pq[i])
            sheet1.write(2, 1 + i, all_sq[i])
            sheet1.write(3, 1 + i, all_rq[i])
        wb.save(rfile)
        
    def eval_seg(self, segmentation: dict, output_dir=None, testset=False, distance=None):
        """
        Evaluate segmentation. Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/evaluate.py
        :param segmentation: {sample_token: prediction <n,> array}. predictions of entire dataset split
        :param output_dir: str. Output directory to store bin files or metrics summary
        :param output_dir: bool. If true, store predicted results to bin files. Else save metrics to .xls
        :param distance: list [near, far]. Specify a range of points to evaluate. If None, evaluate all points.ß
        return results: dict. Metrics summary
        """
        
        version = 'v1.0-test' if testset else 'v1.0-trainval'
        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)
        evalset = 'test' if testset else 'val'
        os.makedirs(os.path.join(output_dir, 'lidarseg', evalset), exist_ok=True)
        if not testset:
            lidar_seg_eval = LidarSegEval(nusc, output_dir, evalset, verbose=True)
            assert len(segmentation) == len(lidar_seg_eval.sample_tokens), "number of samples for gt and pred segmentation do not match"

        prog_bar = torchie.ProgressBar(len(segmentation))
        for sample_token, lidarseg_pred in segmentation.items():
            sample = nusc.get('sample', sample_token)
            sd_token = sample['data']['LIDAR_TOP']
            if testset:
                lidarseg_pred_filename = os.path.join(output_dir, 'lidarseg',
                                                      evalset, sd_token + '_lidarseg.bin')
                lidarseg_pred.astype(np.uint8).tofile(lidarseg_pred_filename)
            else:
                lidarseg_label_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', sd_token)['filename'])
                lidarseg_label = load_bin_file(lidarseg_label_filename)
                lidarseg_label = lidar_seg_eval.mapper.convert_label(lidarseg_label)
                lidarseg_pred = lidarseg_pred.astype(np.uint8)
                if distance is not None:
                    lidar_path = nusc.get_sample_data_path(sd_token)
                    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
                    dist = np.linalg.norm(points[:, :2], axis=1)
                    mask = (dist >= distance[0]) & (dist < distance[1])
                    lidarseg_label = lidarseg_label[mask]
                    lidarseg_pred = lidarseg_pred[mask]
                lidar_seg_eval.global_cm.update(lidarseg_label, lidarseg_pred)
            prog_bar.update()
        if not testset:
            iou_per_class = lidar_seg_eval.global_cm.get_per_class_iou()
            miou = lidar_seg_eval.global_cm.get_mean_iou()
            freqweighted_iou = lidar_seg_eval.global_cm.get_freqweighted_iou()

            results = {
                'iou_per_class': {lidar_seg_eval.id2name[i]: class_iou for i, class_iou in enumerate(iou_per_class)},
                'miou': miou,
                'freq_weighted_iou': freqweighted_iou}

            # store eval metrics to xls
            rfile = os.path.join(output_dir, 'seg_metrics.xls')
            wb = Workbook()
            sheet1 = wb.add_sheet('segmentation')
            i = 0
            for key, value in results.items():
                if key == 'iou_per_class':
                    for k, v in results[key].items():
                        sheet1.write(0, i, k)
                        sheet1.write(1, i, v)
                        i += 1
                else:
                    sheet1.write(0, i, key)
                    sheet1.write(1, i, value)
                    i += 1

            wb.save(rfile)
            return results
    
    def eval_det(self, detections, output_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res
