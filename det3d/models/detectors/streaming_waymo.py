from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .point_pillars import PointPillars
from copy import deepcopy 
import numpy as np
import torch, time
from torch import nn
from collections import defaultdict
import random
@DETECTORS.register_module
class PointPillarsLSTM(PointPillars):
    """
    Waymo's method according to my first implementation. Due to the ambiguities in the paper, two implementations are possible. This one hurt the performance in all cases.
    """
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head=None,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(PointPillarsLSTM, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.lstm = nn.LSTM(reader.num_filters[-1], reader.num_filters[-1])

    def extract_feat_static(self, data, lstm_out=None):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )

        x1 = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if lstm_out is not None:
            x1 += lstm_out
        next_context = []
        if self.with_neck:
            x2, next_context = self.neck(x1, data['prev_context'])

        return x1, x2, next_context

    def extract_feat_dynamic(self, data, lstm_out=None):
        input_features, unq = self.reader(data)

        x1 = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"], self.reader.point_density,
        )
        if lstm_out is not None:
            x1 += lstm_out.view((x1.shape[0], -1, 1, 1))
        next_context = []
        if self.with_neck:
            x2, next_context = self.neck(x1, data['prev_context'])
            # x2 = self.neck(x1)
        return x1, x2, next_context

    def forward(self, example, return_loss=True, **kwargs):
        ret = []
        stateful_nms = self.test_cfg.get('stateful_nms', False)
        prev_context = []
        lstm_out = None
        for i, ex in enumerate(example):
            kwargs.update({'prev_context': prev_context,
                           'sec_id': i})
            if (i > 0) and stateful_nms and ('det' in ret[-1]):
                kwargs.update({'prev_dets': ret[-1]['det']})

            ret_dict, pooled = self.forward_one_sector(ex, return_loss, lstm_out, **kwargs)

            if i < len(example) - 1:
                prev_context = ret_dict.pop('next_context', [])
                lstm_out, _ = self.lstm(pooled)
            ret.append(ret_dict)

            if 'key_points_index' in ex:
                ret[-1]['key_points_index'] = ex['key_points_index']

        ret = self.merge_sectors(ret, len(ex['metadata']))

        if stateful_nms and ('det' in ret):
            for det, meta in zip(ret['det'], ex['metadata']):
                det['metadata'] = meta

        return ret

    def forward_one_sector(self, example, return_loss=True, lstm_out=None, **kwargs):
        preds = {}
        # hard voxelization
        if 'voxels' in example:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

            batch_size = len(num_voxels)

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_static
        # dynamic voxelization
        else:
            num_points_per_sample = example["num_points"]
            batch_size = len(num_points_per_sample)

            data = dict(
                points=example['points'],
                grid_ind=example['grid_ind'],
                num_points=num_points_per_sample,
                batch_size=batch_size,
                voxel_size=example['voxel_size'][0],
                pc_range=example['pc_range'][0],
                grid_size=example['grid_size'][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_dynamic

        x1, x2, next_context = extract_feat(data, lstm_out)

        if self.bbox_head:
            preds.update(self.bbox_head(x2))

        if self.seg_head:
            preds.update(self.seg_head(x1, x2))

        ret_dict = {}
        if return_loss:
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            if self.part_head:
                self.part_head(preds)
                ret_dict.update(self.part_head.loss(example, preds))
        else:
            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg, **kwargs)

            if self.seg_head:
                ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

        if len(next_context):
            ret_dict.update({'next_context': next_context})

        return ret_dict, x1.mean(dim=(-2, -1)).unsqueeze(0)


@DETECTORS.register_module
class PointPillarsLSTMV1(PointPillarsLSTM):
    """
    My second implementation of Waymo's method and it is used in the paper. It worked when there are more than 8 sectors in a scene.
    """

    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head=None,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(PointPillarsLSTMV1, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.lstm = nn.LSTM(neck.ds_num_filters[-1], neck.ds_num_filters[-1])

    def extract_feat_dynamic(self, data, lstm_out=None):
        input_features, unq = self.reader(data)

        x1 = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"],
        )

        if self.with_neck:
            x2, new_lstm_out = self.neck(x1, lstm_out)
        return x1, x2, new_lstm_out

    def forward_one_sector(self, example, return_loss=True, lstm_out=None, **kwargs):
        preds = {}
        # hard voxelization
        if 'voxels' in example:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

            batch_size = len(num_voxels)

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_static
        # dynamic voxelization
        else:
            num_points_per_sample = example["num_points"]
            batch_size = len(num_points_per_sample)

            data = dict(
                points=example['points'],
                grid_ind=example['grid_ind'],
                num_points=num_points_per_sample,
                batch_size=batch_size,
                voxel_size=example['voxel_size'][0],
                pc_range=example['pc_range'][0],
                grid_size=example['grid_size'][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_dynamic

        x1, x2, new_lstm_out = extract_feat(data, lstm_out)

        if self.bbox_head:
            preds.update(self.bbox_head(x2))

        if self.seg_head:
            preds.update(self.seg_head(x1, x2))

        ret_dict = {}
        if return_loss:
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            if self.part_head:
                self.part_head(preds)
                ret_dict.update(self.part_head.loss(example, preds))
        else:
            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg, **kwargs)

            if self.seg_head:
                if self.test_cfg.get('panoptic', False):
                    sec_id = kwargs.get('sec_id', 0)
                    # ret_dict = self.seg_head.debug_panoptic(example, preds, self.test_cfg, ret_dict,
                    #                                         voxel_shape=self.bbox_head.voxel_shape,
                    #                                         class_names=self.bbox_head.class_names, sec_id=sec_id)
                    ret_dict = self.seg_head.predict_panoptic(example, preds, self.test_cfg, ret_dict,
                                                              voxel_shape=self.bbox_head.voxel_shape,
                                                              class_names=self.bbox_head.class_names, sec_id=sec_id)
                else:
                    ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

        return ret_dict, new_lstm_out


@DETECTORS.register_module
class PointPillarsNoLSTM(PointPillarsLSTM):
    """
    A trial that adds pooled features without LSTM. Turned out LSTMV1 was better.
    """
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head=None,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(PointPillarsNoLSTM, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )

    def forward(self, example, return_loss=True, **kwargs):
        ret = []
        stateful_nms = self.test_cfg.get('stateful_nms', False)
        prev_context = []
        lstm_out = None
        for i, ex in enumerate(example):
            kwargs.update({'prev_context': prev_context,
                           'sec_id': i})
            if (i > 0) and stateful_nms and ('det' in ret[-1]):
                kwargs.update({'prev_dets': ret[-1]['det']})

            ret_dict, pooled = self.forward_one_sector(ex, return_loss, lstm_out, **kwargs)

            if i < len(example) - 1:
                prev_context = ret_dict.pop('next_context', [])
                lstm_out = pooled
            ret.append(ret_dict)

            if 'key_points_index' in ex:
                ret[-1]['key_points_index'] = ex['key_points_index']

        ret = self.merge_sectors(ret, len(ex['metadata']))

        if stateful_nms and ('det' in ret):
            for det, meta in zip(ret['det'], ex['metadata']):
                det['metadata'] = meta

        return ret
