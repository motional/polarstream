import numpy as np
from det3d.core.bbox import box_np_ops

from det3d.core.input.voxel_generator import VoxelGenerator

from ..registry import PIPELINES
import copy
from collections import defaultdict
from .utils import filter_gt, transform_points
from .preprocess import AssignLabel

@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.dynamic = cfg.get('dynamic', False)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size

        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num,
                                                                                  int) else cfg.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )
        self.return_density = cfg.get('return_density', False)
        self.double_flip = cfg.get('double_flip', False)
        self.super_tasks = kwargs.get('super_tasks', ['det'])
        self.nsectors = cfg.get('nsectors', 1)
        self.return_pc_grid_ind = False
        if 'seg' in self.super_tasks:
            self.return_pc_grid_ind = True
            assert not self.double_flip, "currently not supporting double flip for segmentation"
        self.times = defaultdict(list)
        self.avg_times = []

    def get_grid_ind(self, res, pc_grid_ind, grid_size):
        if res["mode"] in ["train", "debug_gt"]:
            # filter points without label
            pc_label = res["lidar"]["pc_label"]
            valid = (pc_label >= 0).squeeze(1)
            pc_grid_ind = pc_grid_ind[valid, :]
            label_voxel_pair = np.concatenate([pc_grid_ind, pc_label[valid].astype(pc_grid_ind.dtype)], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((pc_grid_ind[:, 0], pc_grid_ind[:, 1], pc_grid_ind[:, 2])),
                               :]

            voxel_labels = np.zeros(grid_size[::-1], dtype=np.long)
            AssignLabel.assign_voxel_labels(label_voxel_pair, voxel_labels)

            res["lidar"]["voxels"].update({"labels": voxel_labels[np.newaxis, ...]})

        else:
            pc_grid_ind = pc_grid_ind[:res["lidar"]["n_key_points"]]

        res["lidar"]["voxels"].update({"valid_grid_ind": pc_grid_ind.copy()})

        return res

    def voxelize_hard(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] in ["train", "debug_gt"]:
            filter_gt(res, pc_range)
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points, pc_grid_ind, voxel_pc_density = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels, return_pc_grid_ind=self.return_pc_grid_ind,
            return_density=self.return_density
        )

        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        if 'seg' in self.super_tasks:
            res = self.get_grid_ind(res, pc_grid_ind, grid_size)
            if 'part' in self.super_tasks:
                AssignLabel.assign_part_2d(res)

        if self.return_density:
            res["lidar"]["voxels"].update({"n_points": voxel_pc_density})

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points, _, _ = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points, _, _ = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points, _, _ = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        return res, info

    def voxelize_dynamic(self, res, info, **kwargs):
        """
        Dynamic Voxelization
        """
        grid_size = self.voxel_generator.grid_size
        pc_range = self.voxel_generator.point_cloud_range
        voxel_size = self.voxel_generator.voxel_size

        if res["mode"] in ["train", "debug_gt"]:
            if res["voxel_shape"] != 'cuboid':
                cur_pc_range = pc_range.copy()
                cur_pc_range[1] = -np.pi
                cur_pc_range[5] = np.pi
                filter_gt(res, cur_pc_range)
            else:
                filter_gt(res, pc_range)

        points = res["lidar"]["points"]
        pc_grid_ind = (np.floor(
            np.clip((points[:, :3] - pc_range[:3]) / voxel_size, a_min=0, a_max=grid_size - 1))).astype(
            np.int)[:, ::-1]

        res["lidar"]["voxels"] = dict(grid_ind=pc_grid_ind.copy(),
                                      shape=grid_size,
                                      range=pc_range,
                                      size=voxel_size,
                                      )

        if ('seg' in self.super_tasks) and kwargs.get('seg', True):
            res = self.get_grid_ind(res, pc_grid_ind, grid_size)
            if ('part' in self.super_tasks) and (res["mode"] in ["train", "debug_gt"]):
                AssignLabel.assign_part_2d(res)

        return res, info

    def voxelize_streaming_cart(self, res, info, **kwargs):
        """
        Generate voxels for streaming on-the-fly with cartesian voxels
        """
        grid_size = self.voxel_generator.grid_size
        pc_range = self.voxel_generator.point_cloud_range
        voxel_size = self.voxel_generator.voxel_size

        nsectors = self.nsectors
        interval = 2 * np.pi / nsectors
        r = np.sqrt(pc_range[0] ** 2 + pc_range[1] ** 2)

        cur_grid_size = grid_size.copy()
        ref_pc_range = pc_range.copy()
        if nsectors >= 64:
            cur_grid_size[0] //= 2
            cur_grid_size[1] //= 16
            ref_pc_range[3] = 0
            ref_pc_range[4] = 0
            ref_pc_range[1] /= 8
        if nsectors >= 32:
            cur_grid_size[0] //= 2
            cur_grid_size[1] //= 8
            ref_pc_range[3] = 0
            ref_pc_range[4] = 0
            ref_pc_range[1] /= 4
        elif nsectors >= 16:
            cur_grid_size[0] //= 2
            cur_grid_size[1] //= 4
            ref_pc_range[3] = 0
            ref_pc_range[4] = 0
            ref_pc_range[1] /= 2
        elif nsectors >= 4:
            cur_grid_size[0] //= 2
            cur_grid_size[1] //= 2
            ref_pc_range[3] = 0
            ref_pc_range[4] = 0
        elif nsectors >= 2:
            cur_grid_size[1] //= 2
            ref_pc_range[4] = 0

        sectors = []
        if 'load_range' in info:
            load_range = info['load_range']
        else:
            load_range = range(nsectors)
        for i in load_range:
            cur_pc_range = np.array(
                [0, -np.pi + i * interval, pc_range[2], r, -np.pi + (i + 1) * interval, pc_range[-1]])
            # cur_res = copy.deepcopy(res)
            cur_res = {}
            for k, v in res.items():
                if k in ['metadata', 'mode', 'type', 'voxel_shape', 'transform_matrix']:
                    cur_res[k] = copy.deepcopy(v)
            cur_res['lidar'] = {}
            if res["mode"] in ["train", "debug_gt"]:
                cur_res['lidar']['annotations'] = copy.deepcopy(res['lidar']['annotations'])
                cur_res["voxel_shape"] = 'cylinder'
                filter_gt(cur_res, cur_pc_range)
                cur_res["voxel_shape"] = 'cuboid'
                gt_boxes = cur_res["lidar"]["annotations"]['gt_boxes']
                if len(gt_boxes):
                    angle = cur_pc_range[1] + np.pi  # -np.pi
                    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
                        gt_boxes[:, :3], angle, axis=2
                    )
                    gt_boxes[:, -1] += angle
                    if gt_boxes.shape[1] > 7:
                        gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
                            np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
                            angle,
                            axis=2,
                        )[:, :2]
                    cur_res["lidar"]["annotations"]['gt_boxes'] = gt_boxes

            points = res["lidar"]["points"].copy()

            if i == 0:
                if self.nsectors == 1:
                    points_index = np.arange(len(points))
                else:
                    points_index = np.where(points[:, -1] < cur_pc_range[4])[0]
            elif i == self.nsectors - 1:
                points_index = np.where(points[:, -1] >= cur_pc_range[1])[0]
            else:
                points_index = np.where((points[:, -1] >= cur_pc_range[1]) & (points[:, -1] < cur_pc_range[4]))[0]
            points = points[points_index]

            points[:, -1] -= cur_pc_range[1] + np.pi
            points[:, 0] = points[:, -2] * np.cos(points[:, -1])
            points[:, 1] = points[:, -2] * np.sin(points[:, -1])

            pc_grid_ind = (np.floor(
                np.clip((points[:, :3] - ref_pc_range[:3]) / voxel_size, a_min=0, a_max=cur_grid_size - 1))).astype(
                np.int)[:, ::-1]

            cur_res["lidar"]["points"] = points.copy()

            cur_res["lidar"]["voxels"] = dict(grid_ind=pc_grid_ind.copy(),
                                              shape=cur_grid_size,
                                              range=ref_pc_range,
                                              size=voxel_size,
                                              ).copy()

            if ('seg' in self.super_tasks) and kwargs.get('seg', True):
                if res["mode"] in ["train", "debug_gt"]:
                    cur_res["lidar"]["pc_label"] = res["lidar"]["pc_label"][points_index].copy()
                    if res["mode"] == "debug_gt":
                        key_points_index = points_index[points_index < res['lidar']['n_key_points']]
                        cur_res['lidar']['n_key_points'] = len(key_points_index)
                        cur_res['lidar']['key_points_index'] = key_points_index
                else:
                    key_points_index = points_index[points_index < res['lidar']['n_key_points']]
                    cur_res['lidar']['n_key_points'] = len(key_points_index)
                    cur_res['lidar']['key_points_index'] = key_points_index
                cur_res = self.get_grid_ind(cur_res, pc_grid_ind, cur_grid_size)
                if ('part' in self.super_tasks) and (res["mode"] in ["train", "debug_gt"]):
                    AssignLabel.assign_part_2d(cur_res)
            sectors.append(cur_res)
        res = {'sectors': sectors}
        return res, info

    def voxelize_streaming_polar(self, res, info, **kwargs):
        """
                Generate voxels for streaming on-the-fly with polar voxels
                """
        grid_size = self.voxel_generator.grid_size
        pc_range = self.voxel_generator.point_cloud_range
        voxel_size = self.voxel_generator.voxel_size

        nsectors = self.nsectors
        min_az, max_az = pc_range[1], pc_range[4]
        interval = (max_az - min_az) / nsectors
        cur_grid_size = grid_size.copy()
        cur_grid_size[1] //= nsectors
        sectors = []
        ref_pc_range = pc_range.copy()
        ref_pc_range[4] = min_az + interval
        if 'load_range' in info:
            load_range = info['load_range']
        else:
            load_range = range(nsectors)

        for i in load_range:
            cur_pc_range = pc_range.copy()
            cur_pc_range[1] = min_az + i * interval
            cur_pc_range[4] = min_az + (i + 1) * interval
            cur_res = copy.deepcopy(res)

            if res["mode"] in ["train", "debug_gt"]:
                filter_gt(cur_res, cur_pc_range)
                gt_boxes = cur_res["lidar"]["annotations"]['gt_boxes']
                if len(gt_boxes):
                    angle = cur_pc_range[1] - pc_range[1]
                    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
                        gt_boxes[:, :3], angle, axis=2
                    )
                    gt_boxes[:, -1] += angle
                    if gt_boxes.shape[1] > 7:
                        gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
                            np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
                            angle,
                            axis=2,
                        )[:, :2]
                    cur_res["lidar"]["annotations"]['gt_boxes'] = gt_boxes
            points = cur_res["lidar"]["points"]

            if i == 0:
                points_index = np.where(points[:, 1] < cur_pc_range[4])[0]
            elif i == self.nsectors - 1:
                points_index = np.where(points[:, 1] >= cur_pc_range[1])[0]
            else:
                points_index = np.where((points[:, 1] >= cur_pc_range[1]) & (points[:, 1] < cur_pc_range[4]))[0]
            points = points[points_index]

            points[:, 1] -= cur_pc_range[1] - pc_range[1]
            points[:, 3] = points[:, 0] * np.cos(points[:, 1])
            points[:, 4] = points[:, 0] * np.sin(points[:, 1])

            cur_res["lidar"]["points"] = points

            pc_grid_ind = (np.floor(
                np.clip((points[:, :3] - pc_range[:3]) / voxel_size, a_min=0, a_max=cur_grid_size - 1))).astype(
                np.int)[:, ::-1]

            cur_res["lidar"]["voxels"] = dict(grid_ind=copy.deepcopy(pc_grid_ind),
                                              shape=cur_grid_size,
                                              range=ref_pc_range,
                                              size=voxel_size,
                                              )
            
            if ('seg' in self.super_tasks) and (kwargs.get('seg', True)):
                if res["mode"] in ["train", "debug_gt"]:
                    cur_res["lidar"]["pc_label"] = cur_res["lidar"]["pc_label"][points_index]
                    # points_save = np.concatenate([points[:, 3:5], points[:, 2:3], cur_res["lidar"]["pc_label"]], 1)
                    if res["mode"] == "debug_gt":
                        key_points_index = points_index[points_index < res['lidar']['n_key_points']]
                        cur_res['lidar']['n_key_points'] = len(key_points_index)
                        cur_res['lidar']['key_points_index'] = key_points_index
                else:
                    key_points_index = points_index[points_index < res['lidar']['n_key_points']]
                    cur_res['lidar']['n_key_points'] = len(key_points_index)
                    cur_res['lidar']['key_points_index'] = key_points_index
                cur_res = self.get_grid_ind(cur_res, pc_grid_ind, cur_grid_size)
 
                if ('part' in self.super_tasks) and (res["mode"] in ["train", "debug_gt"]):
                    AssignLabel.assign_part_2d(cur_res)
            sectors.append(cur_res)

        res = {'sectors': sectors}
        return res, info

    def voxelize_streaming_by_sweep(self, res, info):
        """
        voxelize multi sweeps.
        If voxel_shape is 'cuboid', uber's method. If 'cylinder', bidirectional padding.
        Remember to set 'transform_type' to 'feature' in the dataset.
        """
        npoints_sweep = np.cumsum(res['lidar']['npoints_sweep'])
        nsweeps = len(npoints_sweep)
        npoints_sweep = np.insert(npoints_sweep, 0, 0)
        res_sweeps = []
        transform_matrices = []
        if res['voxel_shape'] == 'cuboid':
            for i in range(1, nsweeps):
                tm = np.linalg.inv(res['lidar']['transform_matrices'][i]) @ res['lidar']['transform_matrices'][i - 1]
                tm = tm[:2, :2]
                transform_matrices.append(tm)
            voxelize = self.voxelize_streaming_cart
            seg = True
            for i in range(nsweeps):
                cur_res = copy.deepcopy(res)
                cur_res['lidar']['points'] = cur_res['lidar']['points'][npoints_sweep[i]:npoints_sweep[i + 1]]
                if i > 0:
                    cur_res['lidar']['annotations'] = None
                    cur_res['mode'] = 'eval'
                    cur_res['transform_matrix'] = transform_matrices[i - 1]

                    cur_res['lidar']['points'][:, :3] = (np.hstack(
                        (cur_res['lidar']['points'][:, :3], np.ones((cur_res['lidar']['points'].shape[0], 1)))) @ \
                                                         np.linalg.inv(res['lidar']['transform_matrices'][i]).T)[:, :3]
                    seg = False
                if (i == 0) and ('seg' in self.super_tasks) and (res["mode"] in ["train", "debug_gt"]):
                    cur_res['lidar']['pc_label'] = cur_res['lidar']['pc_label'][npoints_sweep[i]:npoints_sweep[i + 1]]
                cur_res['lidar']['points'] = transform_points(cur_res['lidar']['points'][:, :5], 'cuboid')
                cur_res, _ = voxelize(cur_res, info, seg=seg)
                res_sweeps = cur_res['sectors'] + res_sweeps
        else:
            pivot = nsweeps - 1 # 11 - 1 = 10

            # later 10 sweeps
            cur_res = copy.deepcopy(res)
            cur_res['lidar']['points'] = cur_res['lidar']['points'][:npoints_sweep[pivot]]
            cur_res['lidar']['points'] = transform_points(cur_res['lidar']['points'][:, :5], 'cylinder')
            cur_res, _ = self.voxelize_streaming_polar(cur_res, info)
            res_sweeps = cur_res['sectors'] + res_sweeps

            # previous 10 sweeps
            cur_res = copy.deepcopy(res)
            cur_res['lidar']['points'] = cur_res['lidar']['points'][npoints_sweep[1]:]
            cur_res['lidar']['points'][:, -1] -= cur_res['lidar']['points'][0, -1] # fix time lag

            # warp back to previous sweep because when we read point clouds we warped to the latest sweep
            tm = np.linalg.inv(res['lidar']['transform_matrices'][1])
            cur_res['transform_matrix'] = tm[:2, :2]
            cur_res['lidar']['points'][:, :3] = (np.hstack(
                (cur_res['lidar']['points'][:, :3], np.ones((cur_res['lidar']['points'].shape[0], 1)))) @ \
                                                 tm.T)[:, :3]
            cur_res['lidar']['points'] = transform_points(cur_res['lidar']['points'][:, :5], 'cylinder')

            cur_res, _ = self.voxelize_streaming_polar(cur_res, info, seg=False)
            if ('seg' in self.super_tasks) and (res['mode'] in ['train', 'debug_gt']):
                for i in range(len(cur_res['sectors'])):
                    cur_res['sectors'][i]["lidar"]["voxels"]['labels'] = res_sweeps[i]["lidar"]["voxels"]['labels'].copy()
            res_sweeps = cur_res['sectors'] + res_sweeps
            nsweeps = 2

        return {'sweeps': res_sweeps, 'nsweeps': nsweeps, 'nsectors': len(cur_res['sectors'])}, info

    def __call__(self, res, info):
        if res['lidar']['transform_type'] == 'feature':
            return self.voxelize_streaming_by_sweep(res, info)
        elif self.nsectors > 1:
            if res['voxel_shape'] == 'cuboid':
                return self.voxelize_streaming_cart(res, info)

            else:
                return self.voxelize_streaming_polar(res, info)
        elif not self.dynamic:
            return self.voxelize_hard(res, info)
        else:
            return self.voxelize_dynamic(res, info)
