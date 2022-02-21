# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, WeightedFastFocalLoss
from det3d.models.utils import Sequential
from ..registry import BBOX_HEADS
from ..utils.norm import RSNorm
import copy 
import numpy as np
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")
from .center_head import CenterHead
from torch.nn import functional as F

class RangeStratified(nn.Module):
    """
        Range stratified convolution and normalization
        :param kernel: tuple
        :param nheads: int. number of heads. 1 for single group.
        :param ngoups: int
        :param inchannels: int.
        :param outchannels: int.
        :para act: string. 'ReLU' or 'Mish'. 'Mish' does not work.
        """
    def __init__(self, kernel, nheads, ngroups, inchannels, outchannels, act='ReLU'):
        super(RangeStratified, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels * ngroups * nheads, outchannels * ngroups * nheads, kernel, groups=ngroups * nheads),
            nn.GroupNorm(ngroups * nheads, outchannels * ngroups * nheads),
            activation[act],
        )

        self.padding_az = kernel[0] // 2
        self.padding_r = kernel[1] // 2
        self.ngroups = ngroups
    def forward(self, x):
        x = F.pad(x, (0, 0, self.padding_az, self.padding_az))
        step = x.shape[-1] // self.ngroups
        if self.padding_r > 0:
            x = F.pad(x, (self.padding_r, self.padding_r, 0, 0))
            x = torch.cat([x[:,:, :, (step*i):(step*(i+1)+2*self.padding_r)] for i in range(self.ngroups)], 1)
        else:
            x = torch.cat([x[:,:, :, step*i:step*(i+1)] for i in range(self.ngroups)], 1)
        x = self.conv(x)
        step = x.shape[1] // self.ngroups
        x = torch.cat([x[:, step * i:step * (i + 1), :, :] for i in range(self.ngroups)], -1)
        return x

#does not work
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))

activation = {'ReLU': nn.ReLU(inplace=True), 'Mish': Mish()}

@BBOX_HEADS.register_module
class CenterHeadSingle(CenterHead):
    """
    Centerpoint Single-group det heads.
    :param voxel_shape: string. 'cuboid' or 'cylinder'
    :param act: string. 'ReLU' or 'Mish'. 'ReLU' is better.
    """
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
        voxel_shape='cuboid',
        act='ReLU',
    ):
        super(CenterHeadSingle, self).__init__(in_channels, tasks, dataset, weight, code_weights,
                                           common_heads, logger, init_bias, share_conv_channel,
                                           num_hm_conv, dcn_head, voxel_shape)
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset
        self.num_heads = len(num_classes)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.voxel_shape = voxel_shape
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.common_heads = common_heads
        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False
        self.heads = copy.deepcopy(common_heads)
        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            RSNorm(1, 4, share_conv_channel),
            activation[act]
        )
        self.tasks = None
        print("Use HM Bias: ", init_bias)

        head_conv = 64
        final_kernel = 3
        #improve runtime by merging heads
        for head in common_heads:
            classes, num_conv = common_heads[head]

            fc = Sequential()
            if 'reg' in head:
                fc.add(RangeStratified((3,3), 1, 8, share_conv_channel, head_conv, act)),
                fc.add(nn.Conv2d(head_conv, classes,
                                 kernel_size=1, bias=True))
            elif '_' in head:
                n = len(head.split('_'))
                for i in range(num_conv - 1):
                    fc.add(nn.Conv2d(share_conv_channel, head_conv,
                                     kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True, groups=n))

                    fc.add(nn.GroupNorm(head_conv, head_conv))
                    fc.add(activation[act])

                fc.add(nn.Conv2d(head_conv, classes * n,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True, groups=n))
            else:
                for i in range(num_conv - 1):
                    fc.add(nn.Conv2d(share_conv_channel, head_conv,
                                     kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))

                    fc.add(nn.GroupNorm(head_conv, head_conv))
                    fc.add(activation[act])

                fc.add(nn.Conv2d(head_conv, classes,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
            self.__setattr__(head, fc)

        self.hm = Sequential()
        self.heads.update(dict(hm=(sum(num_classes), num_hm_conv)))
        for _ in range(num_hm_conv - 1):
            self.hm.add(nn.Conv2d(share_conv_channel, head_conv,
                                       kernel_size=final_kernel, stride=1,
                                       padding=final_kernel // 2, bias=True))
            self.hm.add(nn.GroupNorm(head_conv, head_conv))
            self.hm.add(activation[act])
        self.hm.add(nn.Conv2d(head_conv, sum(num_classes),
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True))
        logger.info("Finish CenterHead Initialization")
    def forward(self, x):
        ret_dict = dict()
        x = self.shared_conv(x)
        for head in self.heads:
            if '_' in head:
                names = head.split('_')
                tmp = self.__getattr__(head)(x)
                dim = tmp.shape[1] // len(names)
                for j, nm in enumerate(names):
                    ret_dict[nm] = tmp[:, j * dim: (j + 1) * dim, ...]
            elif 'heightdim' in head:
                tmp = self.__getattr__(head)(x)
                ret_dict['height'] = tmp[:, :1, ...]
                ret_dict['dim'] = tmp[:, 1:, ...]
            else:
                ret_dict[head] = self.__getattr__(head)(x)

        return {'det_preds': [ret_dict]}


@BBOX_HEADS.register_module
class CenterHeadSinglePos(CenterHeadSingle):
    """
    Centerpoint Single-group det heads with range stratified and feature undistortion
    :param voxel_shape: string. 'cuboid' or 'cylinder'
    :param voxel_generator: dict.
    :param out_size_fator: int. stride of RPN
    """
    def __init__(
            self,
            in_channels=[128, ],
            tasks=[],
            dataset='nuscenes',
            weight=0.25,
            code_weights=[],
            common_heads=dict(),
            logger=None,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            dcn_head=False,
            voxel_shape='cuboid',
            voxel_generator=None,
            out_size_factor=4,
    ):
        super(CenterHeadSinglePos, self).__init__(in_channels, tasks, dataset, weight, code_weights,
                                              common_heads, logger, init_bias, share_conv_channel,
                                              num_hm_conv, dcn_head, voxel_shape)
        head_conv = 64
        # position decoding
        with torch.no_grad():
            pc_range = voxel_generator['range']
            voxel_size = voxel_generator['voxel_size']
            nsectors = voxel_generator['nsectors']
            min_az, max_az = pc_range[1], pc_range[4]
            interval = (max_az - min_az) / nsectors
            ref_pc_range = pc_range.copy()
            ref_pc_range[4] = min_az + interval
            r_size = round((ref_pc_range[3] - ref_pc_range[0]) / voxel_size[0] / out_size_factor)
            a_size = round((ref_pc_range[4] - ref_pc_range[1]) / voxel_size[1] / out_size_factor)
            grid_a, grid_r = torch.meshgrid(torch.arange(a_size, device=torch.cuda.current_device()),
                                            torch.arange(r_size, device=torch.cuda.current_device()))
            grid_a = grid_a * out_size_factor * voxel_size[1] + ref_pc_range[1]
            grid_r = grid_r * out_size_factor * voxel_size[0] + ref_pc_range[0]
            cos = torch.cos(grid_a)
            sin = torch.sin(grid_a)
            self.pos_encoding = torch.cat(
                [(grid_r * cos).unsqueeze(0), (grid_r * sin).unsqueeze(0), grid_r.unsqueeze(0), cos.unsqueeze(0),
                 sin.unsqueeze(0)]).unsqueeze(0)

        # undistortion weight and bias
        self.calibration_weight = Sequential(
            nn.Conv2d(5, head_conv, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(head_conv, head_conv, kernel_size=1),
            nn.Tanh(),
        )
        self.calibration_bias = Sequential(
            nn.Conv2d(5, head_conv, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(head_conv, head_conv, kernel_size=1),
        )

    def forward(self, x, **kwargs):
        ret_dict = dict()
        x = self.shared_conv(x)
        cal_weight = self.calibration_weight(self.pos_encoding)
        cal_bias = self.calibration_bias(self.pos_encoding)

        calibrated = x * cal_weight + cal_bias

        for head in self.heads:
            if '_' in head:
                names = head.split('_')
                tmp = self.__getattr__(head)(x)
                dim = tmp.shape[1] // len(names)
                for j, nm in enumerate(names):
                    ret_dict[nm] = tmp[:, j * dim: (j + 1) * dim, ...]
            elif 'heightdim' in head:
                tmp = self.__getattr__(head)(x)
                ret_dict['height'] = tmp[:, :1, ...]
                ret_dict['dim'] = tmp[:, 1:, ...]
            elif 'hm' in head:
                ret_dict[head] = self.__getattr__(head)(calibrated)
            else:
                ret_dict[head] = self.__getattr__(head)(x)
        return {'det_preds': [ret_dict]}