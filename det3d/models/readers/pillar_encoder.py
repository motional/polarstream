"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from det3d.models.utils import get_paddings_indicator
from torch import nn
from torch.nn import functional as F
from ..registry import BACKBONES, READERS
from ..utils import build_norm_layer
import torch_scatter
try:
    import spconv
    from spconv import SparseConv3d, SubMConv3d
except:
    print('import spconv failed')
class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs, unq_inv=None):
        if unq_inv is None:
            return self.forward_static(inputs)
        else:
            return self.forward_dynamic(inputs, unq_inv)

    def forward_static(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        
    def forward_dynamic(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = F.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv]], dim=-1)
            return x_concatenated

@READERS.register_module
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.times = []

    def forward(self, features, num_voxels, coors):
        
        device = features.device

        dtype = features.dtype

        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze()


@BACKBONES.register_module
class PointPillarsScatter(nn.Module):
    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):

        self.nx = input_shape[0]
        self.ny = input_shape[1]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas


def get_cluster(points, unq_inv):
    """
    Find distance of x, y, and z from cluster center
    :param points: tensor <n, 3>.
    :param unq_inv: tensor <n, 4>. inverse index for each point
    return f_cluster: tensor <n, 3>
    """

    points_mean = torch_scatter.scatter_mean(points, unq_inv, dim=0)
    f_cluster = points - points_mean[unq_inv]
    return f_cluster

def polar2cart(r_center, a_center):
    """
    convert r, theta to x, y
    :param r_center: tensor<n,>
    :param a_center: tensor<n,>
    return x_center, y_center: tensor<n,>, tensor<n,>
    """
    x_center = r_center * torch.cos(a_center)
    y_center = r_center * torch.sin(a_center)
    return x_center, y_center

def cart2polar(x_center, y_center):
    """
    convert x,y to r, theta(a)
    :param x_center: tensor<n,>
    :param y_center: tensor<n,>
    return r_center, a_center: tensor<n,>, tensor<n,>
    """
    r_center = torch.sqrt(x_center**2 + y_center**2)
    a_center = torch.atan2(y_center, x_center)
    return r_center, a_center

@READERS.register_module
class DynamicPFNet(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        voxel_shape='cuboid',
        xyz_cluster=False,
        raz_cluster=False,
        xy_center=False,
        ra_center=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net with dynamic voxelization
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        :param voxel_shape: string. 'cuboid' or 'cylinder'
        :param xyz_cluster: bool. Whether to use xyz clusters for feature decoration
        :param raz_cluster: bool. Whether to use raz clusters for feature decoration
        :param xy_center: bool. Whether to use xy center for feature decoration
        :param ra_center: bool. Wheter to use ra center for feature decoration
        """
        super().__init__()
        self.name = "DynamicPFNet"
        assert len(num_filters) > 0
        self.num_input = num_input_features
        self.voxel_shape = voxel_shape
        self.xyz_cluster = xyz_cluster
        self.raz_cluster = raz_cluster
        self.xy_center = xy_center
        self.ra_center = ra_center
        if self.xyz_cluster:
            num_input_features += 3
        if self.xy_center:
            num_input_features += 2
        if self.raz_cluster:
            if self.xyz_cluster:
                # ra is enough
                num_input_features += 2
            else:
                num_input_features += 3
        if self.ra_center:
            num_input_features += 2
        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.times = []
        self.name = "DynamicPFNet"
        
    def feature_deco(self, points, unq_inv, grid_ind):
        """
        Feature decoration to each points.
        :param points: tensor <n, n_features>
        :param unq_inv: tensor <n, 4>. Inverse index for each point.
        :param grid_ind: tensor <n, 4>. Index for each point
        return decorated features: tensor<n, m>
        """
        dtype = points.dtype
        features = [points]
        if self.xy_center or self.ra_center:
            # Find distance of x, y, and z from pillar center
            center1 = grid_ind[:, 3:].to(dtype) * self.vx + self.x_offset
            center2 = grid_ind[:, 2:3].to(dtype) * self.vy + self.y_offset
        if self.xyz_cluster or self.xy_center:
            if self.voxel_shape == 'cuboid':
                xyz = points[:, :3]
            else:
                xyz = points[:, [3, 4, 2]] 

            if self.xyz_cluster:
                features.append(get_cluster(xyz, unq_inv))
                
            if self.xy_center:
                if self.voxel_shape == 'cuboid':
                    x_center, y_center = center1, center2
                else:
                    x_center, y_center = polar2cart(center1, center2)
                features.append(xyz[:, 0:1] - x_center)
                features.append(xyz[:, 1:2] - y_center)
        if self.raz_cluster or self.ra_center:
            if self.voxel_shape != 'cuboid':
                ra = points[:, :2]
            else:
                ra = points[:, -2:]
            if self.raz_cluster:
                if self.xyz_cluster:
                    # z was computed already
                    features.append(get_cluster(ra, unq_inv))
                else:
                    z = points[:, 2:3]
                    features.append(get_cluster(torch.cat([ra, z], 1), unq_inv))
            if self.ra_center:
                if self.voxel_shape == 'cuboid':
                    r_center, a_center = cart2polar(center1, center2)
                else:
                    r_center, a_center = center1, center2
                features.append(ra[:, 0:1] - r_center)
                features.append(ra[:, 1:2] - a_center)

        # Combine together feature decorations
        features = torch.cat(features, dim=-1)
        
        return features
        
    def forward(self, data):
        points = data["points"]
        grid_ind = data["grid_ind"]
        

        unq, unq_inv, unq_cnt = torch.unique(grid_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        features = self.feature_deco(points, unq_inv, grid_ind)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        return features, unq
    
@BACKBONES.register_module
class DynamicPPScatter(nn.Module):
    def __init__(self, **kwargs):
        """
        Point Pillar's Scatter with dynamic voxelization
        """

        super().__init__()
        self.name = "DynamicPPScatter"

    def forward(self, voxel_features, unq, batch_size, grid_size):
        """
        :param voxel_features: tensor<n_voxel, n_feature>
        :param unq: tensor<n_voxel, 4>. Index for non-empty voxel
        :param batch_size: int
        :param grid_size: tuple of length 3
        return batch_canvas: tensor<bs, n_features, h, w>
        """
        nx = grid_size[0]
        ny = grid_size[1]
        nchannels = voxel_features.shape[1]
        batch_canvas = voxel_features.new_zeros((batch_size, nchannels, ny, nx))
        batch_canvas[unq[:,0], :, unq[:,2], unq[:,3]] = voxel_features

        return batch_canvas