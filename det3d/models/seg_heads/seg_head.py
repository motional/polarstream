
import torch
from ..registry import SEG_HEAD

from .. import builder

from torch import nn
import numpy as np
from typing import Tuple, Union

semantic2box = [
                'barrier',
                'bicycle',
                'bus',
                'car',
                'construction_vehicle',
                'motorcycle',
                'pedestrian',
                'traffic_cone',
                'trailer',
                'truck',
            ]

def get_labels(labels, pred_shape):
    """
    get semantic labels from input labels. sometimes requires downsampling or projecting from 3D to 2D if output predictions has smaller size.
    :param labels: tensor<n,c,l,h,w>
    :param pred_shape: tuple of length 2 or 3.
    """
    label_shape = labels.shape[1:]
    if label_shape == pred_shape:
        return labels - 1
    kw = label_shape[-2] // pred_shape[-2]
    kl = label_shape[-1] // pred_shape[-1]
    if len(label_shape) == 3:
        if len(pred_shape) == 2:
            kh = label_shape[0]
        else:
            kh = label_shape[0] // pred_shape[0]
        pool = nn.AvgPool3d((kh, kw, kl))
        one_hot = (nn.functional.one_hot(labels).permute((0, 4, 1, 2, 3)))[:, 1:, :, :, :]
    else:
        pool = nn.AvgPool2d((kw, kl))
        one_hot = (nn.functional.one_hot(labels).permute((0, 3, 1, 2)))[:, 1:, :, :]
    labels_down = pool(one_hot.float())
    ignore_mask = labels_down.sum(1) == 0
    labels_down = labels_down.argmax(1)
    labels_down[ignore_mask] = -1
    return labels_down.squeeze(-3)


@SEG_HEAD.register_module
class SingleConvHead(nn.Module):
    """
    Segmentation head with single conv.
    :param num_classes: int. #output semantic classes
    :param in_channels: int.
    :param weight: int. segmentation weight wrt. detection weight
    :param loss: dict. loss function config.
    """
    def __init__(
            self,
            kernel=1,
            num_classes=16,
            in_channels=448,
            weight=1,
            loss=None,
    ):
        super(SingleConvHead, self).__init__()
        self.num_classes=num_classes
        self.conv = nn.Conv2d(in_channels, num_classes, kernel, padding=kernel//2)
        self.weight = weight
        self.loss_func = builder.build_loss(loss)
        
    def forward(self, x1, x2):
        """
        :param x1: tensor<n,c,h1,w1>. canvas
        :param x2: tensor<n,c,h2,w2>. features from RPN
        """
        x = nn.functional.interpolate(x2, size=x1.shape[-2:], mode='bilinear')
        x = torch.cat([x1, x], dim=1)
        x = self.conv(x)
        return {'seg_preds': x}
        

    def loss(self, example, preds_dicts, **kwargs):
        """
        seg loss fuction.
        :param example: dict. info about input data
        :param preds_dicts: dict. info about predictions
        return seg_loss: dict
        """
        seg_preds = preds_dicts['seg_preds']
        seg_labels = get_labels(example['voxel_labels'].squeeze(1), seg_preds.shape[2:])
        seg_loss = self.weight * self.loss_func(seg_preds, seg_labels)
        return {'seg_loss': [seg_loss]}


    @torch.no_grad()
    def predict_panoptic(self, example, preds_dicts, test_cfg, ret_dict, **kwargs):
        """
        stateful panoptic fusion. Not well-defined for streaming.
        :param example: dict. info about input data
        :param preds_dicts: dict. info about predictions
        :param test_cfg: dict.
        :param ret_dict: dict. To store results. Already storing detection results.
        return ret_dict: dict. with seg predictions and instance ids.
        """
        ret_dict['seg'] = []
        ret_dict['ins'] = []
        batch_size = len(example['num_voxels']) if 'num_voxels' in example else len(example['num_points'])
        pred_labels = preds_dicts['seg_preds']
        pred_labels = torch.argmax(pred_labels, dim=1) + 1
        ndim = pred_labels.ndim
        voxel_shape = kwargs.get('voxel_shape', 'cuboid')
        class_names = [item for sublist in kwargs['class_names'] for item in sublist]
        sec_id = kwargs['sec_id']
        point_num = np.insert(np.cumsum(example['num_points'].cpu().numpy(),0), 0, 0)

        if voxel_shape == 'cuboid':
            angle = 2 * np.pi / test_cfg.interval * sec_id
        else:
            angle = test_cfg.interval * sec_id
        rot_sin = np.sin(-angle)
        rot_cos = np.cos(angle)
        rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=torch.float,
                                 device=pred_labels.device)
        th = 0.3
        if ndim == 3:
            pred_labels = pred_labels.unsqueeze(1)
        for count in range(batch_size):
            token = example['metadata'][count]['token']
            pc_grid_ind = example['valid_grid_ind'][count]
            if ndim == 3:
                preds = pred_labels[count][0, pc_grid_ind[:, 1], pc_grid_ind[:, 2]]
            else:
                preds = pred_labels[count][pc_grid_ind[:, 0], pc_grid_ind[:, 1], pc_grid_ind[:, 2]]
            ret_dict['seg'].append({token: preds})

            instance_preds = preds.new_zeros((len(preds),), dtype=int)
            points = example['points'][point_num[count]:point_num[count + 1]]
            if voxel_shape == 'cuboid':
                points = points[:, :2]
            else:
                points = points[:, 3:5]
            for i in range(1, 11):
                thing_inds = torch.nonzero(preds == i, as_tuple=False).reshape(-1)
                if thing_inds.shape[0] == 0: 
                    continue
                box_label = class_names.index(semantic2box[i - 1])

                # only works for single group head
                bbox_mask = (ret_dict['det'][0][count]['label_preds'] == box_label) & (
                            ret_dict['det'][0][count]['scores'] > th)
                bboxes = ret_dict['det'][0][count]['box3d_lidar'][bbox_mask][:, :2] 
                instances = ret_dict['det'][0][count]['instances'][bbox_mask]

                if bbox_mask.sum() == 0: continue
                cur_points = points[thing_inds] @ rot_mat_T
                dists = torch.cdist(cur_points, bboxes)
                bid = torch.argmin(dists, dim=1)
                bid = instances[bid]
                instance_preds[thing_inds] = bid
            
            ret_dict['ins'].append({token: instance_preds})
        ret_dict['seg'] = iter(ret_dict['seg'])
        ret_dict['ins'] = iter(ret_dict['ins'])
        return ret_dict
                    
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """
        predict semantic labels for each point.
        :param example: dict. info about batch input data
        :param preds_dicts: dict. info about predictions
        :param test_cfg: dict.
        return predicted labels: list[dict]. Each element contains a data sample.
        """
        batch_size = len(example['num_voxels']) if 'num_voxels' in example else len(example['num_points'])
        pred_labels = preds_dicts['seg_preds']
        pred_labels = torch.argmax(pred_labels, dim=1) + 1
        ndim = pred_labels.ndim
        if ndim == 3:
            pred_labels = pred_labels.unsqueeze(1)
        for count in range(batch_size):
            token = example['metadata'][count]['token']
            pc_grid_ind = example['valid_grid_ind'][count]
            if ndim == 3:
                preds = pred_labels[count][0, pc_grid_ind[:,1], pc_grid_ind[:,2]]
            else:
                preds = pred_labels[count][pc_grid_ind[:, 0], pc_grid_ind[:, 1], pc_grid_ind[:, 2]]
            yield {token: preds}


class UpsampleShelhamer(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels, scale_factor: Union[int, Tuple[int, int]]):
        """
        Upsampling layer that is an approximation to bilinear upsampling and can be implemented as a deconvolution
        layer in TensorRT.
        :param n_channels: <int>. Number of input and output channels.
        :param scale_factor: <int> or (ht <int>, wt <int>). Scale factor for upsampling.
        """
        if isinstance(scale_factor, tuple):
            padding = (int(scale_factor[0] / 2), int(scale_factor[1] / 2))
            kernel_size = (int(2 * scale_factor[0]) if scale_factor[0] != 1 else 1,
                           int(2 * scale_factor[1]) if scale_factor[1] != 1 else 1)
        else:
            assert scale_factor != 1, 'Scaling both dimensions to 1?'
            padding = int(scale_factor / 2)
            kernel_size = 2 * scale_factor
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=padding,
                         bias=False)
        # # Initialize the weights
        # if isinstance(scale_factor, int):
        #     assert self.kernel_size[0] == self.kernel_size[1]
        # initial_weight = get_upsampling_weight(self.in_channels, self.out_channels, self.kernel_size)
        # self.weight.detach().copy_(initial_weight)
        #
        # # Turn off gradient updating
        # self.weight.requires_grad = False


@SEG_HEAD.register_module
class DeconvConvHead(SingleConvHead):
    """
    Use deconv to upsample
    """

    def __init__(
            self,
            kernel,
            num_classes=16,
            in_channels_voxel=16,
            in_channels=512,
            up_scale=8,
            weight=1,
            loss=None,
            height=1,
    ):
        if height == 1:
            deconv_channels = in_channels // up_scale
        elif height > 1:
            deconv_channels = height
        else:
            raise NotImplementedError
        super(DeconvConvHead, self).__init__(kernel, num_classes * height, deconv_channels + in_channels_voxel * height, weight,
                                             loss)

        self.deconv = UpsampleShelhamer(in_channels, deconv_channels, up_scale)

    def forward(self, x1, x2):
        ndim = x1.ndim
        shape = x1.shape
        x2 = self.deconv(x2)
        if ndim == 5:
            x2 = x2.view((shape[0], -1) + shape[2:]).contiguous()
            x1 = torch.cat([x1, x2], dim=1)
            x1 = x1.view((shape[0], -1) + shape[3:]).contiguous()
            x1 = self.conv(x1)
            x1 = x1.view((shape[0], -1) + shape[2:]).contiguous()
        else:
            x1 = torch.cat([x1, x2], dim=1)
            x1 = self.conv(x1)
        return {'seg_preds': x1}
