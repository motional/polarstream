import pickle
import numpy as np
from det3d.core import box_np_ops
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
class_range = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30
}
info_path = 'data/nuScenes/infos_val_01sweeps_withvelo_filter_True.pkl'
with open(info_path, "rb") as f:
    nusc_infos_all = pickle.load(f)
    
for i, info in enumerate(nusc_infos_all):
    lidar_path = info['lidar_path']
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    lidarseg_labels_filename = info['lidarseg_path']
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
    points_label = np.vectorize(NUSCENES_SEMANTIC_MAPPING.__getitem__)(points_label)

    instances_labels = np.zeros(points_label.shape, dtype=int) - 1

    for k, v in THINGS.items():
        seg_mask = (points_label == v).reshape(-1)
        if seg_mask.sum() == 0: continue
        cur_points = points[seg_mask.reshape(-1)]
        
        bbox_mask = (info['gt_names'] == k).reshape(-1)
        # assert bbox_mask.sum() > 0, f'{k} has semantic points but without boxes'
        if bbox_mask.sum() == 0:
            print(f'{k} has semantic points but no boxes')
            continue
        bboxes = info['gt_boxes'][bbox_mask]

        point_indices = box_np_ops.points_in_rbbox(cur_points, bboxes)
        point_id, instance_id = np.where(point_indices)
        point_id = np.where(seg_mask)[0][point_id]
        unsettled_point_id = np.where(point_indices.sum(1) == 0)[0]
        unsettled_point_id = np.where(seg_mask)[0][unsettled_point_id]
        for pid in unsettled_point_id:
            pt = points[pid, :2]
            norm = np.linalg.norm(bboxes[:, :2] - pt, axis=1)
            bid = np.argmin(norm)
            if norm[bid] < 5:
                point_id = np.append(point_id, pid)
                instance_id = np.append(instance_id, bid)
 
        unique_id, counts = np.unique(instance_id, return_counts=True)
        for id, c in zip(unique_id, counts):
            if c >= 20:
                pid = point_id[instance_id == id]
                instances_labels[pid] = id
        
    for i in range(11, 17):
        seg_mask = (points_label == i).reshape(-1)
        if seg_mask.sum() == 0: continue
        instances_labels[seg_mask.reshape(-1)] = 0

    ratio1 = ((instances_labels == -1).sum()) / len(instances_labels)
    ratio2 = ((instances_labels == -1).sum() - (points_label == 0).sum()) / len(instances_labels)
    print(f'{ratio1} of points are ignored, {ratio2} of points are ignored due to outside any boxes or boxes too sparse')
    instances_filename = lidarseg_labels_filename.replace('lidarseg', 'instance_all')
    instances_labels.tofile(instances_filename)

    