import numpy as np
import numba
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES
import copy
from .utils import _dict_select, drop_arrays_by_name, transform_points

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        self.voxel_shape = cfg.get('voxel_shape', 'cuboid')
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler.enable:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)
        elif self.mode == "debug_gt":
            self.class_names = cfg.class_names

        self.no_augmentation = cfg.get('no_augmentation', False)
        self.super_tasks = kwargs.get('super_tasks', ['det'])

    def __call__(self, res, info):
        res["mode"] = self.mode

        points = res["lidar"]["points"]
        if self.mode in ["train", "debug_gt"]:
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }
                
        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            
            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"]+1, # db contains pt semantic labels
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )
                
                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if 'seg' not in self.super_tasks:
                        # remove semantic labels in db
                        sampled_points = sampled_points[:, :-1]
                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            )
        elif self.mode == "debug_gt" or self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        if self.mode in ["train", "debug_gt"]:
            res["lidar"]["annotations"] = gt_dict
        
        # recover points and segmentation labels
        if ('seg' in self.super_tasks) and (res["mode"] in ['train', 'debug_gt']):
            res["lidar"]["pc_label"] = copy.deepcopy(points[:, -1:])
            # remove point semantic labels
            points = points[:, :-1]

        if res['lidar']['transform_type'] != 'feature':
            # convert xyz to r,\theta,z for cylinder; add r, \theta to cartesian for fair comparison
            res["lidar"]["points"] = transform_points(points, self.voxel_shape)
        else:
            res["lidar"]["points"] = points

        res["voxel_shape"] = self.voxel_shape
        
        return res, info

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.voxel_shape = assigner_cfg.get('voxel_shape', 'cuboid')
        self.super_tasks = kwargs.get('super_tasks', ['det'])
        self.rectify = kwargs.get('rectify', False)
        self.anno_box = []

    @staticmethod
    @numba.njit(cache=True, parallel=False)
    def assign_voxel_labels(label_voxel_pair: np.ndarray, voxel_labels: np.ndarray):
        """
        Custom function to assign labels to grid cells (voxels) using numba to speed up computation.
        :param voxel_labels: <np.int32, grid size>. semantic lables for input voxels.
        :param label_voxel_pair: <np.int32, n_points, 4>. Grid index and label of points
        """
        label_size = 256
        counter = np.zeros((label_size,), dtype=np.uint16)
        counter[label_voxel_pair[0, 3]] = 1
        
        cur_sear_ind = label_voxel_pair[0, :3]
        
        for i in range(1, label_voxel_pair.shape[0]):
            cur_ind = label_voxel_pair[i, :3]
            if not np.all(np.equal(cur_ind, cur_sear_ind)):
                voxel_labels[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
                counter = np.zeros((label_size,), dtype=np.uint16)
                cur_sear_ind = cur_ind
            counter[label_voxel_pair[i, 3]] += 1
        voxel_labels[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    def assign_heatmap_cuboid(self, hms, anno_boxs, inds, masks, cats, gt_dict, tasks, voxel_size, pc_range, feature_map_size, dataset):
        for idx, task in enumerate(tasks):
            if gt_dict['gt_boxes'][idx] is None: continue
            num_objs = min(gt_dict['gt_boxes'][idx].shape[0], self._max_objs)
            
            gt_boxes = gt_dict['gt_boxes'][idx][:num_objs]
            gt_classes = gt_dict['gt_classes'][idx][:num_objs]
            cls_ids = gt_classes - 1
            ws, ls, hs = gt_boxes[:, 3] / voxel_size[0] / self.out_size_factor, gt_boxes[:, 4] / voxel_size[1] / self.out_size_factor, \
                      gt_boxes[:, 5]
            
            for k in range(num_objs):
                cls_id = cls_ids[k]
 
                w, l = ws[k] , ls[k]
                if w > 0 and l > 0:
                    radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))

                    # be really careful for the coordinate system of your box annotation.
                    x, y, z = gt_boxes[k][0], gt_boxes[k][1], \
                              gt_boxes[k][2]

                    coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                     (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                    ct = np.array(
                        [coor_x, coor_y], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    draw_umich_gaussian(hms[idx][cls_id], ct, radius)

                    new_idx = k
                    x, y = ct_int[0], ct_int[1]

                    cats[idx][new_idx] = cls_id
                    inds[idx][new_idx] = y * feature_map_size[0] + x
                    masks[idx][new_idx] = 1

                    if dataset == 'NuScenesDataset':
                        vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                        rot = gt_dict['gt_boxes'][idx][k][8]
                        anno_boxs[idx][new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                             np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        
                    elif dataset == 'WaymoDataset':
                        vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                        rot = gt_dict['gt_boxes'][idx][k][-1]
                        anno_boxs[idx][new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                             np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                    else:
                        raise NotImplementedError("Only Support Waymo and nuScene for Now")

    # @numba.jit(cache=True, parallel=False)
    def assign_heatmap_polar(self, hms, anno_boxs, inds, masks, cats, gt_dict, tasks, voxel_size, pc_range,
                              feature_map_size, dataset):
        """
        Assign center heatmaps to polar grid. Called by assign_centerpoint().
        """
        for idx, task in enumerate(tasks):
            if gt_dict['gt_boxes'][idx] is None: continue
            num_objs = min(gt_dict['gt_boxes'][idx].shape[0], self._max_objs)

            gt_boxes = gt_dict['gt_boxes'][idx][:num_objs]
            gt_classes = gt_dict['gt_classes'][idx][:num_objs]
            cls_ids = gt_classes - 1

            corners = box_np_ops.center_to_corner_box2d(gt_boxes[:, :2], gt_boxes[:, 3:5], angles=gt_boxes[:, 6])
            rhos, azs = np.linalg.norm(corners, axis=-1), np.arctan2(corners[:, :, 1],
                                                         corners[:, :, 0])
            min_az = np.min(azs, axis=1)
            max_az = np.max(azs, axis=1)
            min_rho = np.min(rhos, axis=1)
            max_rho = np.max(rhos, axis=1)
            drs, das, hs = (max_rho - min_rho) / voxel_size[0] / self.out_size_factor, (max_az - min_az) / voxel_size[
                1] / self.out_size_factor, \
                         gt_boxes[:, 5]
            
            crs, cas = np.linalg.norm(gt_boxes[:,:2], axis=-1), np.arctan2(gt_boxes[:,1], gt_boxes[:,0])
            for k in range(num_objs):
                cls_id = cls_ids[k]

                dr, da = drs[k], das[k]
                if dr > 0 and da > 0:
                    r, a, z = crs[k], cas[k], gt_boxes[k][2]
                    radius = gaussian_radius((dr, da), min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius)-int(r>30))
                    # be really careful for the coordinate system of your box annotation.
                    
                    coor_r, coor_a = (r - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                     (a - pc_range[1]) / voxel_size[1] / self.out_size_factor

                    ct = np.array(
                        [coor_r, coor_a], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    ct_int[1] = np.clip(ct_int[1], 0, feature_map_size[1] - 1)
                    if not (0 <= ct_int[0] < feature_map_size[0]):
                        continue

                    draw_umich_gaussian(hms[idx][cls_id], ct, radius)

                    new_idx = k
                    r, a = ct_int[0], ct_int[1]
                    
                    r_real, a_real = r * self.out_size_factor *  voxel_size[0] + pc_range[0], \
                                        a * self.out_size_factor *  voxel_size[1] + pc_range[1]
                    x, y = r_real * np.cos(a_real), r_real * np.sin(a_real)
                    
                    cats[idx][new_idx] = cls_id
                    inds[idx][new_idx] = a * feature_map_size[0] + r
                    masks[idx][new_idx] = 1
                    

                    if dataset == 'NuScenesDataset':
                        vx, vy = gt_boxes[k][6:8]
                        rot = gt_boxes[k][8]
                        if self.rectify:
                            rot -= cas[k]
                            # rot -= a_real
                            vr = np.sqrt(vx*vx+vy*vy)
                            va = np.arctan2(vy, vx)
                            va -= cas[k]
                            # va -= a_real
                            vx, vy = vr * np.cos(va), vr * np.sin(va)
                        anno_boxs[idx][new_idx] = np.concatenate(
                            (gt_boxes[k, :2] - (x, y), z, np.log(gt_boxes[k][3:6]),
                             np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
 
                    elif dataset == 'WaymoDataset':
                        vx, vy = gt_boxes[k][6:8]
                        rot = gt_boxes[k][-1]
                        if self.rectify:
                            rot -= cas[k]
                            vr = np.sqrt(vx*vx+vy*vy)
                            va = np.arctan2(vy, vx)
                            va -= cas[k]
                            vx, vy = vr * np.cos(va), vr * np.sin(va)
                        anno_boxs[idx][new_idx] = np.concatenate(
                            (gt_boxes[k, :2] - (x, y), z, np.log(gt_boxes[k][3:6]),
                             np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                    else:
                        raise NotImplementedError("Only Support Waymo and nuScene for Now")

    def assign_centerpoint(self, res):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"] 
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] in ["train", "debug_gt"]:
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    if len(m[0]):
                        task_box.append(gt_dict["gt_boxes"][m])
                        task_class.append(gt_dict["gt_classes"][m] - flag2)
                        task_name.append(gt_dict["gt_names"][m])
                if len(task_box):
                    task_boxes.append(np.concatenate(task_box, axis=0))
                    task_classes.append(np.concatenate(task_class))
                    task_names.append(np.concatenate(task_name))
                else:
                    task_boxes.append(None)
                    task_classes.append(None)
                    task_names.append(None)
                flag2 += len(mask)

            for task_box in task_boxes:
                if task_box is not None:
                    # limit rad to [-pi, pi]
                    task_box[:, -1] = box_np_ops.limit_period(
                        task_box[:, -1], offset=0.5, period=np.pi * 2
                    )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian
            
            hms, anno_boxs, inds, masks, cats = [], [], [], [], []
            for idx in range(len(self.tasks)):
                hms.append(np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32))
                inds.append(np.zeros((max_objs), dtype=np.int64))
                masks.append(np.zeros((max_objs), dtype=np.uint8))
                cats.append(np.zeros((max_objs), dtype=np.int64))
                if res['type'] in ['NuScenesDataset', 'WaymoDataset']:
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_boxs.append(np.zeros((max_objs, 10), dtype=np.float32))
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")
            if self.voxel_shape == 'cuboid':
                self.assign_heatmap_cuboid(hms, anno_boxs, inds, masks, cats, gt_dict, self.tasks, voxel_size, pc_range,
                                           feature_map_size, res['type'])
            else:
                self.assign_heatmap_polar(hms, anno_boxs, inds, masks, cats, gt_dict, self.tasks, voxel_size, pc_range,
                                           feature_map_size, res['type'])


            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example
        return res

        

    def __call__(self, res, info):
        if 'det' in self.super_tasks:
            if 'sweeps' in res:
                for i in range(len(res['sweeps'])):
                    res['sweeps'][i] = self.assign_centerpoint(res['sweeps'][i])
            elif 'sectors' in res:
                for i in range(len(res['sectors'])):       
                    res['sectors'][i] = self.assign_centerpoint(res['sectors'][i])
            else:
                res = self.assign_centerpoint(res)
        return res, info
