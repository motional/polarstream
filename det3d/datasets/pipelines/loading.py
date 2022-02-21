import numpy as np

from pathlib import Path
import pickle 
import os 
from ..registry import PIPELINES
NUSCENES_SEMANTIC_MAPPING = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16,
}

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)
    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T
def read_sweep_raw(sweep, painted=False):
    """
    read sweep point clouds, transform to current frame and keep transform matrices
    param sweep: dict. sweep info
    return transformed points, transform matrices.
    """
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)
    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        transform_matrix = sweep["transform_matrix"]
        points_sweep[:3, :] = transform_matrix.dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    else:
        transform_matrix = np.eye(4)
    
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1])).astype(points_sweep.dtype)
    return np.concatenate([points_sweep.T, curr_times.T], 1), transform_matrix.astype(points_sweep.dtype)

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.super_tasks = kwargs.get('super_tasks', ['det'])

    def __call__(self, res, info):
        if res['lidar']['transform_type'] == 'point':
            return self.get_points(res, info)
        else:
            return self.get_raw_points(res, info)
        
    def get_raw_points(self, res, info):
        """
        Get point clouds of multi sweeps sequentiall for feature transform
        param res: dict
        param info: dict
        """
        res["type"] = self.type
        if self.type == "NuScenesDataset":
            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            # record number of points of key frame (in order to assign segmentation labels
            n_key_points = len(points)

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )
            sweep_points_list = [np.concatenate([points, np.zeros((points.shape[0], 1))], 1)]
            npoints_sweep = [n_key_points]
            transform_matrices = [np.eye(4).astype(points.dtype)]
            for i, sweep in enumerate(info["sweeps"]):
                # if sweep['lidar_path'] == lidar_path: break
                # lidar_path = sweep['lidar_path']
                points_sweep, transform_matrix = read_sweep_raw(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                npoints_sweep.append(len(points_sweep))
                transform_matrices.append(transform_matrix)
            points = np.concatenate(sweep_points_list, axis=0, dtype=np.float32)
            res["lidar"]["points"] = points
            if ('seg' in self.super_tasks) and (res["mode"] in ['train', 'debug_gt']):
                # add semantic labels to last dimension of point features to make them consistent after augmentation
                lidarseg_labels_filename = info['lidarseg_path']
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                points_label = np.vectorize(NUSCENES_SEMANTIC_MAPPING.__getitem__)(points_label)

                assert n_key_points == len(points_label), "points and segmentation labels do not match"
                if len(sweep_points_list) > 1:
                    padded_label = np.zeros((len(points), 1), dtype=points.dtype)
                    padded_label.fill(-1)
                    padded_label[:len(points_label)] = points_label
                    res["lidar"]["points"] = np.hstack([points, padded_label])
                else:
                    res["lidar"]["points"] = np.hstack([points, points_label.astype(points.dtype)])
            res["lidar"]["n_key_points"] = n_key_points
            res["lidar"]['npoints_sweep'] = npoints_sweep
            res['lidar']['transform_matrices'] = transform_matrices

        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1:
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError
        return res, info
    def get_points(self, res, info):
        """
        :param res: <dict>. containing data and annotations
        :param info: <dict>. containing sample data token and gt box information
        :return: (res, info, ds_meta). Updated data samples and move on to loading annotations
        """

        res["type"] = self.type
        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            # record number of points of key frame (in order to assign segmentation labels
            n_key_points = len(points)

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]
            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
            # res["lidar"]["points"] = points
            # res["lidar"]["times"] = times

            res["lidar"]["n_key_points"] = n_key_points
            
            if ('seg' in self.super_tasks) and (res["mode"] in ['train', 'debug_gt']):
                # add semantic labels to last dimension of point features to make them consistent after augmentation
                lidarseg_labels_filename = info['lidarseg_path']
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                points_label = np.vectorize(NUSCENES_SEMANTIC_MAPPING.__getitem__)(points_label)

                assert n_key_points == len(points_label), "points and segmentation labels do not match"
                # assign negative labels to non-key frame points
                if nsweeps > 1:
                    padded_label = np.zeros((len(points), 1), dtype=points.dtype)
                    padded_label.fill(-1)
                    padded_label[:len(points_label)] = points_label
                    # res["lidar"]["combined"] = np.hstack([points, times, padded_label])
                    res["lidar"]["points"] = np.hstack([points, times, padded_label])
                else:
                    # res["lidar"]["combined"] = np.hstack([points, times, points_label.astype(points.dtype)])
                    res["lidar"]["points"] = np.hstack([points, times, points_label.astype(points.dtype)])
                
            else:
                # res["lidar"]["combined"] = np.hstack([points, times])
                res["lidar"]["points"] = np.hstack([points, times])

        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError
        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
                
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
