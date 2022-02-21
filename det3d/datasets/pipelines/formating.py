from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        double_flip = kwargs.get('double_flip', False)
        self.double_flip = double_flip
        self.super_tasks = kwargs.get('super_tasks', ['det'])
        
    def __call__(self, res, info):
        if 'sweeps' in res:
            for i in range(len(res['sweeps'])):
                res['sweeps'][i], _ = self.reformat(res['sweeps'][i], info)
        elif 'sectors' in res:
            for i in range(len(res['sectors'])):       
                res['sectors'][i], _ = self.reformat(res['sectors'][i], info)
        else:
            res, info = self.reformat(res, info)

        return res, info
    
    def reformat(self, res, info):
        meta = res["metadata"]
        data_bundle = dict(metadata=meta, )

        voxels = res["lidar"]["voxels"]
        # static voxelization
        if "voxels" in voxels:
            data_bundle.update({
                "voxels": voxels["voxels"],
                "shape": voxels["shape"],
                "num_points": voxels["num_points"],
                "num_voxels": voxels["num_voxels"],
                "coordinates": voxels["coordinates"],
            })
        # dynamic voxelization
        elif "points" in res["lidar"]:
            data_bundle.update({"points": res["lidar"]["points"],
                                "num_points": np.array([len(res["lidar"]["points"])]),
                                "voxel_size": voxels["size"],
                                "pc_range": voxels["range"],
                                "grid_size": voxels["shape"],
                                "grid_ind": voxels["grid_ind"]
                                })
        else:
            raise AssertionError

        if "valid_grid_ind" in voxels:
            data_bundle.update({"valid_grid_ind": voxels['valid_grid_ind']})
        if "key_points_index" in res["lidar"]:
            data_bundle.update({"key_points_index": res['lidar']["key_points_index"]})
        if "labels" in voxels:
            data_bundle.update({'voxel_labels': voxels['labels']})
        if "part_labels" in voxels:
            data_bundle.update({'part_labels': voxels['part_labels']})
        if 'transform_matrix' in res:
            data_bundle.update({'transform_matrix': res['transform_matrix']})

        if ('det' in self.super_tasks) and (res["mode"] in ["train", "debug_gt"]):
            data_bundle.update(res["lidar"]["targets"])
        elif res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

            if self.double_flip:
                # y axis 
                yflip_points = res["lidar"]["yflip_points"]
                yflip_voxels = res["lidar"]["yflip_voxels"] 
                yflip_data_bundle = dict(
                    metadata=meta,
                    points=yflip_points,
                    voxels=yflip_voxels["voxels"],
                    shape=yflip_voxels["shape"],
                    num_points=yflip_voxels["num_points"],
                    num_voxels=yflip_voxels["num_voxels"],
                    coordinates=yflip_voxels["coordinates"],
                )

                # x axis 
                xflip_points = res["lidar"]["xflip_points"]
                xflip_voxels = res["lidar"]["xflip_voxels"] 
                xflip_data_bundle = dict(
                    metadata=meta,
                    points=xflip_points,
                    voxels=xflip_voxels["voxels"],
                    shape=xflip_voxels["shape"],
                    num_points=xflip_voxels["num_points"],
                    num_voxels=xflip_voxels["num_voxels"],
                    coordinates=xflip_voxels["coordinates"],
                )
                # double axis flip 
                double_flip_points = res["lidar"]["double_flip_points"]
                double_flip_voxels = res["lidar"]["double_flip_voxels"] 
                double_flip_data_bundle = dict(
                    metadata=meta,
                    points=double_flip_points,
                    voxels=double_flip_voxels["voxels"],
                    shape=double_flip_voxels["shape"],
                    num_points=double_flip_voxels["num_points"],
                    num_voxels=double_flip_voxels["num_voxels"],
                    coordinates=double_flip_voxels["coordinates"],
                )

                return [data_bundle, yflip_data_bundle, xflip_data_bundle, double_flip_data_bundle], info

        return data_bundle, info



