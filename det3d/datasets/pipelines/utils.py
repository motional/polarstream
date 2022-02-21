from det3d.core.sampler import preprocess as prep
import numpy as np
def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def filter_gt(res, pc_range):
    gt_dict = res["lidar"]["annotations"]

    if len(gt_dict["gt_boxes"]) > 0:
        bv_range = pc_range[[0, 1, 3, 4]]
        if res["voxel_shape"] == 'cuboid':
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
        else:
            gt_rho = np.linalg.norm(gt_dict["gt_boxes"][:, :2], axis=1)
            gt_diag = np.linalg.norm(gt_dict["gt_boxes"][:, 3:5], axis=1)
            gt_diag *= 0  # 0.4
            gt_az = np.arctan2(gt_dict["gt_boxes"][:, 1], gt_dict["gt_boxes"][:, 0])
            mask = ((gt_rho - gt_diag) >= bv_range[0]) & ((gt_rho + gt_diag) <= bv_range[2]) & (
                    gt_az >= bv_range[1]) & (gt_az <= bv_range[3])
        _dict_select(gt_dict, mask)

        res["lidar"]["annotations"] = gt_dict
        
def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def transform_points(input_pc: np.ndarray, voxel_shape) -> np.ndarray:
    """
    Convert Cartesian coordinates to Cylindrical coordinates
    :param input_pc: <np.float32: n_points, 3>. x, y, z coordinates, other point features
    :return: <np.float32: n_points, n_feat>. rho, phi, z, x, y coordinates, other point features
    """
    rho = np.sqrt(input_pc[:, 0]**2 + input_pc[:, 1]**2)
    phi = np.arctan2(input_pc[:, 1], input_pc[:, 0])
    if voxel_shape == 'cylinder':
        # x, y are kept because they are important point features
        return np.hstack((rho[..., np.newaxis], phi[..., np.newaxis], input_pc[:, 2:3], input_pc[:, :2], input_pc[:, 3:]))
    elif voxel_shape == 'cuboid':
        # extra features are kept for fair comparison with cylinder
        return np.hstack((input_pc, rho[..., np.newaxis], phi[..., np.newaxis]))
    

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)