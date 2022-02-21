from torch import nn
from ..registry import READERS
import torch, torch_scatter


@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()


@READERS.register_module
class DynamicVoxelEncoderV1(nn.Module):
    """
    dynamic version of VoxelFeatureExtractorV3
    """

    def __init__(self, num_input_features=7, out_channels=16, name="DynamicVoxelEncoderV1"
                 ):
        super(DynamicVoxelEncoderV1, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.point_density = False

    def forward(self, data):
        features = data["points"]
        grid_ind = data["grid_ind"]

        unq, unq_inv, unq_cnt = torch.unique(grid_ind, return_inverse=True, return_counts=True, dim=0)
        features = torch_scatter.scatter_mean(features, unq_inv, dim=0)

        return features, unq