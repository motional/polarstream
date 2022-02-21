import torch.nn as nn
import torch
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint
from collections import defaultdict
import numpy as np
def merge_list(inputs):
    """
    merge losses from different sector.
    :param inputs: list
    return ret: list
    """
    ret = inputs[0]
    l = len(inputs)
    for i in range(len(ret)):
        for inp in inputs[1:]:
            ret[i] += inp[i]
        ret[i] /= l
    return ret

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        seg_head=None,
        part_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        nsectors=1,
    ):
        super(SingleStageDetector, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_bbox_head(bbox_head) if bbox_head is not None else None
        self.seg_head = builder.build_seg_head(seg_head) if seg_head is not None else None
        self.part_head = builder.build_part_head(part_head) if part_head is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

    def merge_sectors(self, sectors, batch_size):
        """
        merge results from multiple sectors.
        :param sectors: list
        :param batch_size: int
        return ret: dict
        """
        ret = dict()
        for k in sectors[0].keys():
            res = list(map(lambda x: x[k], sectors))
            if ('loss' in k) or (k in []):
                ret[k] = merge_list(res)
            elif k == 'det':
                ret[k] = self.merge_dets(res)
            elif k == 'key_points_index':
                tmp = [[] for _ in range(batch_size)]
                for sec in res:
                    for i in range(batch_size):
                        tmp[i].append(sec[i])
                for i in range(batch_size):
                    tmp[i] = np.concatenate(tmp[i])
                ret[k] = tmp

            elif k in ['seg', 'ins']:
                tmp = [defaultdict(list) for _ in range(batch_size)]
                for sec in res:
                    for i in range(batch_size):
                        for t, pl in next(sec).items():
                            tmp[i][t].append(pl)

                for i in range(batch_size):
                    for t in tmp[i].keys():
                        tmp[i][t] = torch.cat(tmp[i][t])
                ret[k] = tmp

        if 'key_points_index' in ret:
            for i in range(batch_size):
                for t in ret['seg'][i]:
                    tmp = ret['seg'][i][t].new_zeros(ret['seg'][i][t].shape)
                    tmp[ret['key_points_index'][i]] = ret['seg'][i][t]
                    ret['seg'][i][t] = tmp
                if 'ins' in ret:
                    for t in ret['ins'][i]:
                        tmp = ret['ins'][i][t].new_zeros(ret['ins'][i][t].shape)
                        tmp[ret['key_points_index'][i]] = ret['ins'][i][t]
                        ret['ins'][i][t] = tmp
        return ret

    def merge_dets(self, inputs):
        """
        merge detection results from multiple sectors.
        :param inputs: list of length nsectors
        return ret: list of length batch size
        """
        if self.test_cfg.get('stateful_nms', False) or self.test_cfg.get('panoptic', False):
            inputs = inputs[-1]
            ret = []
            num_samples = len(inputs[0])

            for i in range(num_samples):
                tmp = {}
                for k in inputs[0][i].keys():
                    if k in ["box3d_lidar", "scores"]:
                        tmp[k] = torch.cat([inp[i][k] for inp in inputs])
                    elif k in ["label_preds"]:
                        flag = 0
                        for j, num_class in enumerate(self.bbox_head.num_classes):
                            inputs[j][i][k] += flag
                            flag += num_class
                        tmp[k] = torch.cat([inp[i][k] for inp in inputs])
                ret.append(tmp)
        else:
            ret = []
            for i in range(len(inputs[0])):
                tmp = {}
                for k in inputs[0][0].keys():  # inputs[sector][sample]['boxes3d_lidar']
                    if k == 'metadata':
                        tmp[k] = inputs[0][i][k]
                    else:
                        res = list(map(lambda x: x[i][k], inputs))
                        tmp[k] = torch.cat(res)
                ret.append(tmp)
        return ret