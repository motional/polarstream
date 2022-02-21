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
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.models.utils import Sequential
from ..registry import BBOX_HEADS
import copy 
import numpy as np
from detectron2 import layers
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                    
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@BBOX_HEADS.register_module
class CenterHead(nn.Module):
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
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset
        self.voxel_shape = voxel_shape
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

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
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
        self.xs = self.ys = None
        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return {'det_preds': ret_dicts}

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts['det_preds']):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)   
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                       
            else:
                raise NotImplementedError()

            ret = {}
 
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'det_loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def double_flip_decode(self, preds_dict, meta_list):
        batch_size = preds_dict['hm'].shape[0]
        assert batch_size % 4 == 0, print(batch_size)
        batch_size //= 4
        for k in preds_dict.keys():
            # transform the prediction map back to their original coordinate befor flipping
            # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
            # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is
            # X and Y flip pointcloud(x=-x, y=-y).
            # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
            # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
            # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
            _, H, W, C = preds_dict[k].shape
            preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
            preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1])
            preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
            preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])
        preds_dict['hm'] = torch.sigmoid(preds_dict['hm']).mean(dim=1)
        preds_dict['dim'] = torch.exp(preds_dict['dim']).mean(dim=1)
        preds_dict['height'] = preds_dict['height'].mean(dim=1)
        # y = -y reg_y = 1-reg_y
        preds_dict['reg'][:, 1, ..., 1] = 1 - preds_dict['reg'][:, 1, ..., 1]
        preds_dict['reg'][:, 2, ..., 0] = 1 - preds_dict['reg'][:, 2, ..., 0]

        preds_dict['reg'][:, 3, ..., 0] = 1 - preds_dict['reg'][:, 3, ..., 0]
        preds_dict['reg'][:, 3, ..., 1] = 1 - preds_dict['reg'][:, 3, ..., 1]
        preds_dict['reg'] = preds_dict['reg'].mean(dim=1)
        
        batch_rots = preds_dict['rot'][..., 0:1]
        batch_rotc = preds_dict['rot'][..., 1:2]

        # first yflip
        # y = -y theta = pi -theta
        # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
        # batch_rots[:, 1] the same
        batch_rotc[:, 1] *= -1

        # then xflip x = -x theta = 2pi - theta
        # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
        # batch_rots[:, 2] the same
        batch_rots[:, 2] *= -1

        # double flip
        batch_rots[:, 3] *= -1
        batch_rotc[:, 3] *= -1

        batch_rotc = batch_rotc.mean(dim=1)
        batch_rots = batch_rots.mean(dim=1)
        
        preds_dict['rot'] = torch.cat([batch_rots, batch_rotc], -1)
        if 'vel' in preds_dict:
            preds_dict['vel'][:, 1, ..., 1] *= -1
            preds_dict['vel'][:, 2, ..., 0] *= -1
            preds_dict['vel'][:, 3] *= -1
            preds_dict['vel'] = preds_dict['vel'].mean(dim=1)

        meta_list = meta_list[:4 * int(batch_size):4]
        return meta_list

    @torch.no_grad()
    def decode(self, preds_dict, double_flip, test_cfg, ref_pc_range):
        if not double_flip:
            preds_dict['hm'] = torch.sigmoid(preds_dict['hm'])
            preds_dict['dim'] = torch.exp(preds_dict['dim'])
        batch_rot = torch.atan2(preds_dict['rot'][..., 0:1], preds_dict['rot'][..., 1:2])
            
        batch, H, W, num_cls = preds_dict['hm'].size()

        batch_reg = preds_dict['reg'].reshape(batch, H * W, 2)
        batch_hei = preds_dict['height'].reshape(batch, H * W, 1)

        batch_rot = batch_rot.reshape(batch, H * W, 1)

        batch_dim = preds_dict['dim'].reshape(batch, H * W, 3)
        batch_hm = preds_dict['hm'].reshape(batch, H * W, num_cls)

        if (self.ys is None) or (self.ys.shape[0] != batch):
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm.device).float()
            if self.voxel_shape == 'cylinder':
                rhos = xs.view(batch, -1, 1) * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
                azs = ys.view(batch, -1, 1) * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
                self.xs, self.ys = rhos * torch.cos(azs), rhos * torch.sin(azs)
            else:
                self.xs, self.ys = xs, ys

        if self.voxel_shape == 'cuboid':
            xs = self.xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = self.ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + ref_pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + ref_pc_range[1]
        else:
            xs = self.xs + batch_reg[:, :, 0:1]
            ys = self.ys + batch_reg[:, :, 1:2]
            if test_cfg.get('rectify', False):
                azs = torch.atan2(ys, xs)
                batch_rot += azs

        if 'vel' in preds_dict:
            batch_vel = preds_dict['vel']
            batch_vel = batch_vel.reshape(batch, H * W, 2)
            if (self.voxel_shape == 'cylinder') and test_cfg.get('rectify', False):
                vr = torch.norm(batch_vel, dim = -1)
                va = torch.atan2(batch_vel[:, :, 1], batch_vel[:, :, 0])
                va += azs.squeeze(-1)
                batch_vel[:, :, 0], batch_vel[:, :, 1] = vr * torch.cos(va), vr * torch.sin(va)
            batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
        else:
            batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
        return batch_box_preds, batch_hm

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts['det_preds'][0]['hm'].dtype,
                device=preds_dicts['det_preds'][0]['hm'].device,
            )
        for task_id, preds_dict in enumerate(preds_dicts['det_preds']):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            meta_list = example["metadata"]
            if double_flip:
                meta_list = self.double_flip_decode(preds_dict, meta_list)
            batch_box_preds, batch_hm = self.decode(preds_dict, double_flip, test_cfg, example['pc_range'][0])

            metas.append(meta_list)

            prev_dets = kwargs.get('prev_dets', None)
            prev_det = None if prev_dets is None else prev_dets[task_id]
            sec_id = kwargs.get('sec_id', 0)
            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, prev_det, sec_id))

        if test_cfg.get('stateful_nms', False) or test_cfg.get('panoptic', False):
            return rets
        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []

        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 
    
    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, prev_det, sec_id):
        batch_size = len(batch_hm)

        stateful_nms = test_cfg.get('stateful_nms', False)
        panoptic = test_cfg.get('panoptic', False)
        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            
            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            if prev_det is not None:
                if stateful_nms:
                    if self.voxel_shape == 'cuboid':
                        angle = 2 * np.pi / test_cfg.interval * sec_id
                    else:
                        angle = test_cfg.interval * sec_id
                    rot_sin = np.sin(-angle)
                    rot_cos = np.cos(angle)
                    rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=torch.float,
                                             device=box_preds.device)
                    box_preds[:, :2] = box_preds[:, :2] @ rot_mat_T
                    box_preds[:, -1] -= angle
                    if box_preds.shape[1] > 7:
                        box_preds[:, 6:8] = box_preds[:, 6:8] @ rot_mat_T
                    box_preds = torch.cat((prev_det[i]["box3d_lidar"], box_preds))
                    scores = torch.cat((prev_det[i]["scores"], scores))
                    labels = torch.cat((prev_det[i]["label_preds"], labels))
                    if panoptic:
                        sec_ids = labels.new_zeros((len(labels,)), dtype=int)
                        sec_ids[:len(prev_det[i]["box3d_lidar"])] = sec_id - 1
                        sec_ids[len(prev_det[i]["box3d_lidar"]):] = sec_id
                        instances = labels.new_zeros((len(labels,)), dtype=int)
                        instances[:len(prev_det[i]["box3d_lidar"])] = prev_det[i]["instances"]
                        if len(instances):
                            offset = len(prev_det[i]["box3d_lidar"]) + 1
                
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            nms_post_max_size = test_cfg.nms.nms_post_max_size
            if stateful_nms:
                nms_post_max_size = nms_post_max_size * (sec_id + 1)
            if test_cfg.get('per_class_nms', False):
                boxes_for_nms = boxes_for_nms[:, [0, 1, 3, 4, -1]]
                boxes_for_nms[:, -1] = boxes_for_nms[:, -1] / np.pi * 180
                selected = layers.batched_nms_rotated(boxes_for_nms, scores, labels, test_cfg.nms.nms_iou_threshold)
                selected = selected[:nms_post_max_size]
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores, 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=nms_post_max_size)
                # boxes_for_nms = boxes_for_nms[:, [0, 1, 3, 4, -1]]
                # boxes_for_nms[:, -1] = boxes_for_nms[:, -1] / np.pi * 180
                # selected = layers.nms_rotated(boxes_for_nms, scores, test_cfg.nms.nms_iou_threshold)
                # selected = selected[:nms_post_max_size]
            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]               

            if (not stateful_nms) and (sec_id > 0):
                if self.voxel_shape == 'cuboid':
                    angle = 2 * np.pi / test_cfg.interval * sec_id
                else:
                    angle = test_cfg.interval * sec_id
                rot_sin = np.sin(-angle)
                rot_cos = np.cos(angle)
                rot_mat_T = torch.tensor([[rot_cos, -rot_sin],[rot_sin, rot_cos]], dtype=torch.float, device=selected_boxes.device)
                selected_boxes[:, :2] = selected_boxes[:, :2] @ rot_mat_T
                selected_boxes[:, -1] -= angle
                if selected_boxes.shape[1] > 7:
                    selected_boxes[:, 6:8] = selected_boxes[:, 6:8] @ rot_mat_T

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }
            if panoptic:
                if sec_id == 0:
                    instances = torch.arange(len(selected_boxes), device=selected_boxes.device)
                elif stateful_nms:
                    instances = instances[selected]
                    if len(instances):
                        sec_ids = sec_ids[selected]
                        mask = sec_ids == sec_id
                        tmp_id = torch.arange(mask.sum(), device=selected_boxes.device)
                        instances[mask] = tmp_id + offset
                else:
                    if len(prev_det[i]["box3d_lidar"]):
                        prediction_dict['box3d_lidar'] = torch.cat((prev_det[i]["box3d_lidar"], selected_boxes))
                        prediction_dict['scores'] = torch.cat((prev_det[i]['scores'], selected_scores))
                        prediction_dict['label_preds'] = torch.cat((prev_det[i]['label_preds'], selected_labels))
                        offset = len(prev_det[i]['instances'])
                        instances = torch.arange(len(selected_boxes), device=selected_boxes.device) + offset
                        instances = torch.cat((prev_det[i]['instances'], instances))
                    else:
                        instances = torch.arange(len(selected_boxes), device=selected_boxes.device)
                
                prediction_dict.update({'instances': instances})

            prediction_dicts.append(prediction_dict)

        return prediction_dicts

