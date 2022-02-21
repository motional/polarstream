from .point_pillars import PointPillars
from ..registry import DETECTORS
import torch
from torch import nn


@DETECTORS.register_module
class PolarStream(PointPillars):
    """
    PolarStream model for simultaneous object detection, semantic segmentation and panoptic segmentation.
    Compatible with polar and cartesian pillars, full-sweep & streaming, w/o context padding, w/ trailing-edge padding
    param reader: dict. config
    param backbone: dict. config
    param nect: dict. config
    param bbox_head: dict. config
    param seg_head: dict. config
    param part_head: dict. deprecated. default as None.
    param train_cfg: dict. config
    param test_cfg: dict. config
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
        super(PolarStream, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat_static(self, data):
        """
        extrac features using static voxelization.
        param data: dict
        return features on canvas, after RPN and features to pad next sector
        """
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )

        x1 = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        next_context = []
        if self.with_neck:
            x2, next_context = self.neck(x1, data['prev_context'])

        return x1, x2, next_context

    def extract_feat_dynamic(self, data):
        """
                extract features using dynamic voxelization.
                param data: dict
                return features on canvas, after RPN and features to pad next sector
                """
        input_features, unq = self.reader(data)

        x1 = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"],
        )

        next_context = []
        if self.with_neck:
            x2, next_context = self.neck(x1, data['prev_context'])
        return x1, x2, next_context

    def forward(self, example, return_loss=True, **kwargs):
        """
        Forward function to handle either full-sweep or streaming.
        param example: dict if full-sweep and list if streaming.
        param return_loss: bool. True if training
        """
        if isinstance(example, dict):
            return self.forward_one_sector(example, return_loss, **kwargs)
        # streaming
        elif isinstance(example, list):
            ret = []
            stateful_nms = self.test_cfg.get('stateful_nms', False)
            panoptic = self.test_cfg.get('panoptic', False)
            prev_context = []
            for i, ex in enumerate(example):
                kwargs.update({'prev_context': prev_context,
                               'sec_id': i})
                if (i > 0) and (stateful_nms or panoptic) and ('det' in ret[-1]):
                    kwargs.update({'prev_dets': ret[-1]['det']})

                ret.append(self.forward_one_sector(ex, return_loss, **kwargs))

                if i < len(example) - 1:
                    prev_context = ret[-1].pop('next_context', [])

                if 'key_points_index' in ex:
                    ret[-1]['key_points_index'] = ex['key_points_index']

            ret = self.merge_sectors(ret, len(ex['metadata']))
            if (stateful_nms or panoptic) and ('det' in ret):
                for det, meta in zip(ret['det'], ex['metadata']):
                    det['metadata'] = meta

            return ret

    def forward_one_sector(self, example, return_loss=True, **kwargs):
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

        x1, x2, next_context = extract_feat(data)

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
        else:
            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg, **kwargs)
            if self.seg_head:
                if self.test_cfg.get('panoptic', False):
                    # stateful panoptic fusion
                    sec_id = kwargs.get('sec_id', 0)
                    ret_dict = self.seg_head.predict_panoptic(example, preds, self.test_cfg, ret_dict,
                                                              voxel_shape=self.bbox_head.voxel_shape,
                                                              class_names=self.bbox_head.class_names, sec_id=sec_id)
                else:
                    ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

        if len(next_context):
            ret_dict.update({'next_context': next_context})

        return ret_dict


@DETECTORS.register_module
class PolarStreamBDCP(PolarStream):
    """
    PolarStream w/ bidirection padding
    param reader: dict. config
    param backbone: dict. config
    param nect: dict. config
    param bbox_head: dict. config
    param seg_head: dict. config
    param part_head: dict. deprecated. default as None.
    param train_cfg: dict. config
    param test_cfg: dict. config
    param nsectors: int.
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
            nsectors=1,
    ):
        neck.update({'nsectors': nsectors})
        super(PolarStreamBDCP, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.nsectors = nsectors
        self.grids = []
        self.Hs = []
        self.Ws = []
        self.az = None
        self.r = None
        self.center = None

    @torch.no_grad()
    def get_grids(self, cur_sweep):
        """
        Get the meshgrid coordiates of features for warping. It is precomputed.
        param cur_sweep: list[tensor]
        """
        for i in range(len(cur_sweep)):
            H, W = cur_sweep[i].shape[-2], cur_sweep[i].shape[-1]
            H *= self.nsectors
            self.Hs.append(H)
            self.Ws.append(W)
            grid_az, grid_r = torch.meshgrid(torch.arange(H, device=torch.cuda.current_device()),
                                             torch.arange(W, device=torch.cuda.current_device()))
            grid_az = (self.test_cfg.pc_range[4] - self.test_cfg.pc_range[1]) / H * grid_az + self.test_cfg.pc_range[1]
            grid_r = (self.test_cfg.pc_range[3] - self.test_cfg.pc_range[0]) / W * grid_r + \
                     self.test_cfg.pc_range[0]

            grid_x, grid_y = grid_r * torch.cos(grid_az), grid_r * torch.sin(grid_az)
            grid = torch.stack([grid_x, grid_y], -1)
            self.grids.append(grid)
    @torch.no_grad()
    def get_center(self):
        """
        Get the centers of the coordinates because grid sample requires nomalizing to -1 ~ 1
        """
        self.center = [(self.test_cfg.pc_range[3] + self.test_cfg.pc_range[0]) / 2,
                       (self.test_cfg.pc_range[4] + self.test_cfg.pc_range[1]) / 2]
        self.az = (self.test_cfg.pc_range[4] - self.test_cfg.pc_range[1]) / 2
        self.r = (self.test_cfg.pc_range[3] - self.test_cfg.pc_range[0]) / 2

    def extract_feat_dynamic(self, data, mode='feature_only'):
        """
        extract features using dynamic voxelization.
                param data: dict
                param mode: string. 'feature_only' for features with previous frame because no annotations for them. 'training' or 'eval' for features with current frame.
                return features on canvas, after RPN (and features for bidictional padding)
        """
        input_features, unq = self.reader(data)

        x1 = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"],
        )

        if mode == 'feature_only':
            x2, cur_sweep = self.neck(x1, nsectors=data['nsectors'], mode=mode)
            return x1, x2, cur_sweep
        else:
            x2, _ = self.neck(x1, prev_sweep=data['prev_sweep'], nsectors=data['nsectors'], mode=mode)
            return x1, x2
    def forward(self, example, return_loss=True, **kwargs):
        """
        Forward function to handle two sweeps.
        param example: list[dict] of length 2 for two sweeps
        param return_loss: bool.
        """
        ret = []
        prev_sweep = None
        for i, ex in enumerate(example):
            kwargs.update({'prev_sweep': prev_sweep})
            if i < len(example) - 1:
                mode = 'feature_only'
                prev_sweep = self.forward_one_sweep(ex, mode, return_loss, **kwargs)
            else:
                mode = 'train' if return_loss else 'eval'
                ret = self.forward_one_sweep(ex, mode, return_loss, **kwargs)
        if not return_loss:
            bs = len(ex['metadata']) // self.nsectors
            ret = self.merge_sectors(ret, bs)
            if 'det' in ret and (self.test_cfg.get('panoptic', False) or self.test_cfg.get('stateful_nms', False)):
                for det, meta in zip(ret['det'], ex['metadata'][:bs]):
                    det['metadata'] = meta

        return ret

    def forward_one_sweep(self, example, mode='feature_only', return_loss=True, **kwargs):
        """
        Forward one sweep.
        param example: dict
        param mode: string. 'feature_only', 'train' or 'val'
        return loss or predictions
        """
        preds = {}

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
            prev_context=[],
        )

        nsectors = self.nsectors
        bs = batch_size // nsectors
        data.update({'nsectors': nsectors})
        if mode == 'feature_only':
            with torch.no_grad():
                transform_matrix = example['transform_matrix'][:bs]
                if nsectors == 1:
                    x1, x2, cur_sweep = self.extract_feat_dynamic(data, mode=mode)
                    if len(self.grids) == 0:
                        self.get_grids(cur_sweep)
                    if self.center is None:
                        self.get_center()
                    for i in range(len(cur_sweep)):
                        grid = torch.einsum('bjk,mnk->bmnj', transform_matrix, self.grids[i])
                        grid_rho = torch.norm(grid, dim=-1).unsqueeze(-1)
                        grid_az = torch.atan2(grid[:, :, :, 1], grid[:, :, :, 0]).unsqueeze(-1)
                        grid = torch.cat([grid_rho, grid_az], -1)
                        grid[:, :, :, 0] -= self.center[0]
                        grid[:, :, :, 0] /= self.r
                        grid[:, :, :, 1] -= self.center[1]
                        grid[:, :, :, 1] /= self.az
                        cur_sweep[i] = nn.functional.grid_sample(cur_sweep[i], grid)
                        if i == 0:
                            x1 = nn.functional.grid_sample(x1, grid)
                        elif i == (len(cur_sweep) // 2):
                            x2 = nn.functional.grid_sample(x2, grid)
                else:
                    x1, x2, cur_sweep = self.extract_feat_dynamic(data, mode=mode)
                    if len(self.grids) == 0:
                        self.get_grids(cur_sweep)
                    if self.center is None:
                        self.get_center()

                    for i in range(len(cur_sweep)):
                        cur_sweep[i] = cur_sweep[i].reshape(
                            (nsectors, bs, -1, cur_sweep[i].shape[-2], cur_sweep[i].shape[-1]))
                        cur_sweep[i] = torch.cat([cur_sweep[i][j] for j in range(nsectors)], -2)
                        grid = torch.einsum('bjk,mnk->bmnj', transform_matrix, self.grids[i])

                        grid_rho = torch.norm(grid, dim=-1, keepdim=True)
                        grid_az = torch.atan2(grid[:, :, :, 1], grid[:, :, :, 0]).unsqueeze(-1)

                        grid_rho -= self.center[0]
                        grid_rho /= self.r

                        grid_az -= self.center[1]
                        grid_az /= self.az
                        grid = torch.cat([grid_rho, grid_az], -1)
                        cur_sweep[i] = nn.functional.grid_sample(cur_sweep[i], grid)

                        if i == 0:
                            x1 = x1.reshape(
                                (nsectors, bs, -1, x1.shape[-2], x1.shape[-1]))
                            x1 = torch.cat([x1[j] for j in range(nsectors)], -2)
                            x1 = nn.functional.grid_sample(x1, grid)
                            step = x1.shape[-2] // nsectors
                            x1 = torch.cat([x1[:, :, j * step:(j + 1) * step] for j in range(nsectors)], 0)
                            x1 = x1.reshape((batch_size, -1, x1.shape[-2], x1.shape[-1]))
                        elif i == (len(cur_sweep) // 2):
                            x2 = x2.reshape(
                                (nsectors, bs, -1, x2.shape[-2], x2.shape[-1]))
                            x2 = torch.cat([x2[j] for j in range(nsectors)], -2)
                            x2 = nn.functional.grid_sample(x2, grid)
                            step = x2.shape[-2] // nsectors
                            x2 = torch.cat([x2[:, :, j * step:(j + 1) * step] for j in range(nsectors)], 0)
                            x2 = x2.reshape((batch_size, -1, x2.shape[-2], x2.shape[-1]))
                return cur_sweep


        elif mode in ['train', 'eval']:
            data['prev_sweep'] = kwargs['prev_sweep']
            if nsectors == 1:
                x1, x2 = self.extract_feat_dynamic(data, mode=mode)
            else:
                input_features, unq = self.reader(data)

                x1 = self.backbone(
                    input_features, unq, data["batch_size"], data["grid_size"],
                )
                x2 = []
                prev_context = []
                for j in range(nsectors):
                    tmp, prev_context = self.neck(x1[j * bs: (j + 1) * bs], prev_context=prev_context,
                                                  prev_sweep=kwargs['prev_sweep'], sec_id=j, nsectors=nsectors,
                                                  mode=mode)
                    x2.append(tmp)

                x2 = torch.cat(x2, 0)

        if self.bbox_head:
            preds.update(self.bbox_head(x2))

        if self.seg_head:
            preds.update(self.seg_head(x1, x2))

        if return_loss:
            ret_dict = {}
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            return ret_dict

        else:
            rets = [{} for _ in range(self.nsectors)]

            bs = len(example['metadata']) // self.nsectors
            cum = 0
            panoptic = self.test_cfg.get('panoptic', False)
            for i in range(self.nsectors):
                ex = {}
                pr = {}
                for k, v in example.items():
                    if k in ['metadata', 'pc_range']:
                        ex[k] = v[i * bs:(i + 1) * bs]
                    elif k in ['hm', 'anno_box', 'ind', 'mask', 'cat']:
                        new_ex = []
                        for t in example[k]:
                            new_ex.append(t[i * bs:(i + 1) * bs])
                        ex[k] = new_ex
                if self.bbox_head:
                    pr['det_preds'] = []
                    for t in preds['det_preds']:
                        new_pr = {}
                        for k, v in t.items():
                            new_pr[k] = v[i * bs:(i + 1) * bs]
                        pr['det_preds'].append(new_pr)
                    kwargs.update({'sec_id': i})
                    rets[i].update({'det': self.bbox_head.predict(ex, pr, self.test_cfg, **kwargs)})
                    if panoptic or self.test_cfg.get('stateful_nms', False):
                        if i < (self.nsectors - 1):
                            kwargs.update({'prev_dets': rets[i]['det']})
                        else:
                            kwargs.update({'prev_dets': None})
                if self.seg_head:
                    pr['seg_preds'] = preds['seg_preds'][i * bs:(i + 1) * bs]
                    for k in ['num_points', 'valid_grid_ind']:
                        ex[k] = example[k][i * bs:(i + 1) * bs]
                    if self.nsectors > 1:
                        rets[i]['key_points_index'] = example['key_points_index'][i * bs:(i + 1) * bs]
                    if panoptic:
                        ex['points'] = example['points'][cum: (cum + sum(ex['num_points']))]
                        cum += sum(ex['num_points'])
                        rets[i].update(self.seg_head.predict_panoptic(ex, pr, self.test_cfg, rets[i],
                                                                      voxel_shape=self.bbox_head.voxel_shape,
                                                                      class_names=self.bbox_head.class_names, sec_id=i))
                    else:

                        rets[i].update({'seg': self.seg_head.predict(ex, pr, self.test_cfg)})

            return rets