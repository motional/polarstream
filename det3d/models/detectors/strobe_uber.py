from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy
import numpy as np
import torch, time
from torch import nn
from collections import defaultdict
from .point_pillars import PointPillars
import torchgeometry as tgm

@DETECTORS.register_module
class STROBE(PointPillars):
    """
    My implementation of Uber's method. But following STROBEV1 is more like what is described in the paper. This version worked better than STROBEV1 so I chose this version in the paper.
    This version did not use the updated features from previous sectors from current frame. This one only used previous frames.
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
        super(STROBE, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.nsectors = nsectors
        self.times = defaultdict(list)
        self.i = 0
        self.grids = []
        self.Hs = []
        self.Ws = []
        self.min_corner = None
        self.max_corner = None
        self.center = None
        self.sector_grids = []

    @torch.no_grad()
    def get_grids(self, cur_sweep):
        for i in range(len(cur_sweep)):
            H, W = cur_sweep[i].shape[-2], cur_sweep[i].shape[-1]
            if self.nsectors >= 32:
                W *= 2
                H *= 8
            elif self.nsectors >= 16:
                W *= 2
                H *= 4
            elif self.nsectors >= 4:
                W *= 2
                H *= 2
            elif self.nsectors >= 2:
                H *= 2
            self.Hs.append(H)
            self.Ws.append(W)
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=torch.cuda.current_device()),
                                            torch.arange(W, device=torch.cuda.current_device()))
            grid_y = (self.test_cfg.pc_range[4] - self.test_cfg.pc_range[1]) / H * grid_y + self.test_cfg.pc_range[1]
            grid_x = (self.test_cfg.pc_range[3] - self.test_cfg.pc_range[0]) / W * grid_x + \
                     self.test_cfg.pc_range[0]
            # grid = torch.stack([grid_x, grid_y], 0).view((2, -1))
            grid = torch.stack([grid_x, grid_y], -1)
            self.grids.append(grid)

    @torch.no_grad()
    def get_center(self, cur_sweep):
        if self.nsectors == 1:
            self.min_corner = np.array([self.test_cfg.pc_range[0], self.test_cfg.pc_range[1]])
            self.max_corner = np.array([self.test_cfg.pc_range[3], self.test_cfg.pc_range[4]])
        elif self.nsectors >= 32:
            self.min_corner = np.array([self.test_cfg.pc_range[0], self.test_cfg.pc_range[1]/4])
            self.max_corner = np.array([0, 0])
        elif self.nsectors >= 16:
            self.min_corner = np.array([self.test_cfg.pc_range[0], self.test_cfg.pc_range[1]/2])
            self.max_corner = np.array([0, 0])
        elif self.nsectors >= 4:
            self.min_corner = np.array([self.test_cfg.pc_range[0], self.test_cfg.pc_range[1]])
            self.max_corner = np.array([0, 0])
        elif self.nsectors >= 2:
            self.min_corner = np.array([self.test_cfg.pc_range[0], self.test_cfg.pc_range[1]])
            self.max_corner = np.array([self.test_cfg.pc_range[3], 0])
        self.center = (self.min_corner + self.max_corner) / 2
  
        for j in range(len(cur_sweep)):
            H, W = cur_sweep[j].shape[-2], cur_sweep[j].shape[-1]
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=torch.cuda.current_device()),
                                            torch.arange(W, device=torch.cuda.current_device()))
            grid_y = (self.max_corner[1] - self.min_corner[1]) / H * grid_y + self.min_corner[1]
            grid_x = (self.max_corner[0] - self.min_corner[0]) / W * grid_x + self.min_corner[0]

            grid = torch.stack([grid_x, grid_y], -1)
            grid_sector = []
            for i in range(self.nsectors):
                angle = 2 * np.pi / self.nsectors * i

                rot_sin = np.sin(angle)
                rot_cos = np.cos(angle)
                rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], device=torch.cuda.current_device(), dtype=torch.float32)

                cur_grid = torch.einsum('ij,mnj->mni', rot_mat_T, grid)
                cur_grid[:, :, 0] /= (self.test_cfg.pc_range[3] - self.test_cfg.pc_range[0]) / 2
                cur_grid[:, :, 1] /= (self.test_cfg.pc_range[4] - self.test_cfg.pc_range[1]) / 2
                grid_sector.append(cur_grid)
            self.sector_grids.append(grid_sector)
 
    def extract_feat(self, data):
        input_features, unq = self.reader(data)

        x1 = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"],
        )

        cur_sweep = []
        if self.with_neck:
            x2, cur_sweep = self.neck(x1, data['prev_sweep'])

        return x1, x2, cur_sweep

    def forward(self, example, return_loss=True, **kwargs):
        ret = []
        prev_sweep = None
        for i, ex in enumerate(example):
            kwargs.update({'prev_sweep': prev_sweep})
            if i < len(example) - 1:
                mode = 'feature_only'
                prev_sweep = self.forward_one_sweep(ex, mode, **kwargs)
            else:
                mode = 'train' if return_loss else 'eval'
                ret = self.forward_one_sweep(ex, mode, **kwargs)

        if not return_loss:
            bs = len(ex['metadata']) // self.nsectors
            ret = self.merge_sectors(ret, bs)
            if 'det' in ret and (self.test_cfg.get('panoptic', False) or self.test_cfg.get('stateful_nms', False)):
                for det, meta in zip(ret['det'], ex['metadata'][:bs]):
                    det['metadata'] = meta
        return ret


    def forward_one_sweep(self, example, mode='feature_only', **kwargs):
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
            prev_sweep=kwargs.get('prev_sweep', None)
        )
        
        if mode == 'feature_only':
            with torch.no_grad():
                _, _, cur_sweep = self.extract_feat(data)
                bs = cur_sweep[0].shape[0] // self.nsectors
                transform_matrix = example['transform_matrix'][:bs]
                if len(self.grids) == 0:
                    self.get_grids(cur_sweep)
                if self.min_corner is None:
                    self.get_center(cur_sweep)
            
                for i in range(len(cur_sweep)):
                    if self.nsectors == 1:
                        # grid = (transform_matrix @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                        #     (0, 2, 3, 1))
                        grid = torch.einsum('bjk,mnk->bmnj', transform_matrix, self.grids[i])
                        grid[:, :, :, 0] -= self.center[0]
                        grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                        grid[:, :, :, 1] -= self.center[1]
                        grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2
                        cur_sweep[i] = nn.functional.grid_sample(cur_sweep[i], grid)
                    else:
                        memory = cur_sweep[i].new_zeros((bs, cur_sweep[i].shape[1], self.Hs[i], self.Ws[i]))
                        # grid = transform_matrix @ self.grids[i] #(bs,2,-1)
                        grid = torch.einsum('bjk,mnk->bmnj', transform_matrix, self.grids[i])
                        for j in range(self.nsectors):
                            cur_grid = grid.clone()
                            angle = 2 * np.pi / self.nsectors * j
                            rot_sin = np.sin(angle)
                            rot_cos = np.cos(angle)
                            rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                                     device=torch.cuda.current_device(), dtype=torch.float32)#.unsqueeze(0).repeat((bs, 1, 1))
                            cur_grid = torch.einsum('jk,bmnk->bmnj', rot_mat_T, cur_grid)
                            # cur_grid = (rot_mat_T @ cur_grid).view((bs, 2, self.Hs[i], self.Ws[i])).permute((0, 2, 3, 1))
                            cur_grid[:, :, :, 0] -= self.center[0]
                            cur_grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                            cur_grid[:, :, :, 1] -= self.center[1]
                            cur_grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2
                            tmp = nn.functional.grid_sample(cur_sweep[i][j * bs: (j + 1) * bs], cur_grid)
                            mask = (tmp.abs() > 0).to(int)
                            memory = memory * (1 - mask) + tmp * mask
                    #     memory = memory + nn.functional.grid_sample(cur_sweep[i][j * bs: (j + 1) * bs], cur_grid)
                    # memory = memory / self.nsectors
                        grid_sectors = self.sector_grids[i]
                        next_sweeps = []
                        for j in range(self.nsectors):
                            next_sweeps.append(nn.functional.grid_sample(memory, grid_sectors[j].repeat((bs, 1, 1, 1))))
                        cur_sweep[i] = torch.cat(next_sweeps)
                
                return cur_sweep
        if mode in ['train', 'eval']:
            x1, x2, _ = self.extract_feat(data)
            if self.bbox_head:
                preds.update(self.bbox_head(x2))

            if self.seg_head:
                preds.update(self.seg_head(x1, x2))
        
        if mode == 'train':
            ret_dict = {}
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            if self.part_head:
                self.part_head(preds)
                ret_dict.update(self.part_head.loss(example, preds))
            return ret_dict
        else:
            rets = [{} for _ in range(self.nsectors)]
            panoptic = self.test_cfg.get('panoptic', False)
            bs = len(example['metadata']) // self.nsectors
            cum = 0
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
                        # only works for single group
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

@DETECTORS.register_module
class STROBEV2(STROBE):
    """
    My second implementation of Uber's method. Looks more like what is described in the paper but worked worse than STROBE above.
    This version used the updated features from previous sector of current sweep.
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
        super(STROBEV2, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained, nsectors
        )
        self.num_filters = neck.ds_num_filters[:-1]
        self.memory = [None for _ in self.num_filters]

    def forward_one_sweep(self, example, mode='feature_only', **kwargs):
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
            prev_sweep=kwargs.get('prev_sweep', None)
        )

        bs = batch_size // self.nsectors
        if mode == 'feature_only':
            with torch.no_grad():
                transform_matrix = example['transform_matrix'][:bs]
                if self.nsectors == 1:
                    _, _, cur_sweep = self.extract_feat(data)
                    if len(self.grids) == 0:
                        self.get_grids(cur_sweep)
                    if self.min_corner is None:
                        self.get_center(cur_sweep)
                    for i in range(len(cur_sweep)):
                        grid = (transform_matrix @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                            (0, 2, 3, 1))
                        grid[:, :, :, 0] -= self.center[0]
                        grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                        grid[:, :, :, 1] -= self.center[1]
                        grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2
                        cur_sweep[i] = nn.functional.grid_sample(cur_sweep[i], grid)
                else:
                    input_features, unq = self.reader(data)

                    x1 = self.backbone(
                        input_features, unq, data["batch_size"], data["grid_size"],
                    )
                    for j in range(self.nsectors):
                        _, cur_sweep = self.neck(x1[j * bs: (j + 1) * bs], data['prev_sweep'])
                        if len(self.grids) == 0:
                            self.get_grids(cur_sweep)
                        if self.min_corner is None:
                            self.get_center(cur_sweep)
                    
                        for i in range(len(cur_sweep)):
                            if self.memory[i] is None:
                                self.memory[i] = data['points'].new_zeros(
                                    (bs, cur_sweep[i].shape[1], self.Hs[i], self.Ws[i]))
                        
                            # put jth sector into memory
                            angle = 2 * np.pi / self.nsectors * j
                            rot_sin = np.sin(angle)
                            rot_cos = np.cos(angle)
                            rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                                     device=torch.cuda.current_device(), dtype=torch.float32).unsqueeze(
                                0).repeat((bs, 1, 1))
                            grid = (rot_mat_T @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                                (0, 2, 3, 1))
                            grid[:, :, :, 0] -= self.center[0]
                            grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                            grid[:, :, :, 1] -= self.center[1]
                            grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2

                        tmp = nn.functional.grid_sample(cur_sweep[i], grid)
                        mask = (tmp.abs() > 0).to(int)
                        self.memory[i] = self.memory[i] * (1 - mask) + tmp * mask
                        
                        # get j+1 th sector from memory
                        grid_sectors = self.sector_grids[i][(j+1)%self.nsectors]
                        # if starting a new sweep, transform the features by ego pose
                        if j == (self.nsectors-1):
                            grid = (transform_matrix @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                                (0, 2, 3, 1))
                            grid[:, :, :, 0] -= (self.test_cfg.pc_range[3] + self.test_cfg.pc_range[0]) / 2
                            grid[:, :, :, 0] /= (self.test_cfg.pc_range[3] - self.test_cfg.pc_range[0]) / 2
                            grid[:, :, :, 1] -= (self.test_cfg.pc_range[4] + self.test_cfg.pc_range[1]) / 2
                            grid[:, :, :, 1] /= (self.test_cfg.pc_range[4] - self.test_cfg.pc_range[1]) / 2
                            self.memory[i] = nn.functional.grid_sample(self.memory[i], grid)
                        cur_sweep[i] = nn.functional.grid_sample(self.memory[i], grid_sectors.repeat((bs, 1, 1, 1)))
                        
                    data['prev_sweep'] = cur_sweep

                return cur_sweep
        if mode in ['train', 'eval']:
            if self.nsectors == 1:
                x1, x2, _ = self.extract_feat(data)
            else:
                input_features, unq = self.reader(data)

                x1 = self.backbone(
                    input_features, unq, data["batch_size"], data["grid_size"],
                )
                x2 = []
                for j in range(self.nsectors):
                    tmp, cur_sweep = self.neck(x1[j * bs: (j + 1) * bs], data['prev_sweep'])
                    x2.append(tmp)

                    for i in range(len(cur_sweep)):
                        # put jth sector into memory
                        with torch.no_grad():
                            angle = 2 * np.pi / self.nsectors * j
                            rot_sin = np.sin(angle)
                            rot_cos = np.cos(angle)
                            rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                                     device=torch.cuda.current_device(), dtype=torch.float32).unsqueeze(
                                0).repeat((bs, 1, 1))
                            grid = (rot_mat_T @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                                (0, 2, 3, 1))
                            grid[:, :, :, 0] -= self.center[0]
                            grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                            grid[:, :, :, 1] -= self.center[1]
                            grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2

                        tmp = nn.functional.grid_sample(cur_sweep[i], grid)
                        with torch.no_grad():
                            mask = (tmp.abs() > 0).to(int)
                        self.memory[i] = self.memory[i] * (1-mask) + tmp * mask

                        # get j+1 th sector from memory
                        if j < (self.nsectors - 1):
                            grid_sectors = self.sector_grids[i][j]
                            cur_sweep[i] = nn.functional.grid_sample(self.memory[i], grid_sectors.repeat((bs, 1, 1, 1)))

                    data['prev_sweep'] = cur_sweep
                x2 = torch.cat(x2)
                self.memory = [None for _ in self.num_filters]
   
            if self.bbox_head:
                preds.update(self.bbox_head(x2))

            if self.seg_head:
                preds.update(self.seg_head(x1, x2))

        if mode == 'train':
            ret_dict = {}
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            if self.part_head:
                self.part_head(preds)
                ret_dict.update(self.part_head.loss(example, preds))
            return ret_dict
        else:
            rets = [{} for _ in range(self.nsectors)]

            bs = len(example['metadata']) // self.nsectors
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
                if self.seg_head:
                    rets[i].update({'seg': self.seg_head.predict(ex, preds, self.test_cfg)})
            return rets

@DETECTORS.register_module
class STROBEV3(STROBEV2):
    """
    this version used homography warp for full sweep instead of grid sampler. Did not work better than STROBE above.
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
        super(STROBEV3, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained, nsectors
        )

    def forward_one_sweep(self, example, mode='feature_only', **kwargs):

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
            prev_sweep=kwargs.get('prev_sweep', None)
        )

        bs = batch_size // self.nsectors
        if mode == 'feature_only':
            with torch.no_grad():
                transform_matrix = example['transform_matrix'][:bs]
                if self.nsectors == 1:
                    _, _, cur_sweep = self.extract_feat(data)
                    if len(self.grids) == 0:
                        self.get_grids(cur_sweep)
                    if self.min_corner is None:
                        self.get_center(cur_sweep)
                    for i in range(len(cur_sweep)):
                        cur_sweep[i] = torch.flip(cur_sweep[i], [-2])
                        cur_sweep[i] = tgm.homography_warp(cur_sweep[i], transform_matrix,
                                                       (cur_sweep[i].shape[-2], cur_sweep[i].shape[-1]))
                        cur_sweep[i] = torch.flip(cur_sweep[i], [-2])
                else:

                    input_features, unq = self.reader(data)

                    x1 = self.backbone(
                        input_features, unq, data["batch_size"], data["grid_size"],
                    )
                    for j in range(self.nsectors):
                        _, cur_sweep = self.neck(x1[j * bs: (j + 1) * bs], data['prev_sweep'])
                        if len(self.grids) == 0:
                            self.get_grids(cur_sweep)
                        if self.min_corner is None:
                            self.get_center(cur_sweep)

                        for i in range(len(cur_sweep)):
                            if self.memory[i] is None:
                                self.memory[i] = data['points'].new_zeros(
                                    (bs, cur_sweep[i].shape[1], self.Hs[i], self.Ws[i]))

                            # put jth sector into memory
                            angle = 2 * np.pi / self.nsectors * j
                            rot_sin = np.sin(angle)
                            rot_cos = np.cos(angle)
                            rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                                     device=torch.cuda.current_device(), dtype=torch.float32).unsqueeze(
                                0).repeat((bs, 1, 1))
                            grid = (rot_mat_T @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                                (0, 2, 3, 1))
                            grid[:, :, :, 0] -= self.center[0]
                            grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                            grid[:, :, :, 1] -= self.center[1]
                            grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2

                            tmp = nn.functional.grid_sample(cur_sweep[i], grid)
                            mask = (tmp.abs() > 0).to(int)
                            self.memory[i] = self.memory[i] * (1 - mask) + tmp * mask

                            # get j+1 th sector from memory
                            grid_sectors = self.sector_grids[i][(j + 1) % self.nsectors]
                            # if starting a new sweep, transform the features by ego pose
                            if j == (self.nsectors - 1):
                                cur_sweep[i] = torch.flip(cur_sweep[i], [-2])
                                cur_sweep[i] = tgm.homography_warp(cur_sweep[i], transform_matrix,
                                                                (cur_sweep[i].shape[-2], cur_sweep[i].shape[-1]))
                                cur_sweep[i] = torch.flip(cur_sweep[i], [-2])
                            cur_sweep[i] = nn.functional.grid_sample(self.memory[i], grid_sectors.repeat((bs, 1, 1, 1)))

                        data['prev_sweep'] = cur_sweep

                return cur_sweep
        if mode in ['train', 'eval']:
            if self.nsectors == 1:
                x1, x2, _ = self.extract_feat(data)
            else:
                input_features, unq = self.reader(data)

                x1 = self.backbone(
                    input_features, unq, data["batch_size"], data["grid_size"],
                )
                x2 = []
                for j in range(self.nsectors):
                    tmp, cur_sweep = self.neck(x1[j * bs: (j + 1) * bs], data['prev_sweep'])
                    x2.append(tmp)

                    for i in range(len(cur_sweep)):
                        # put jth sector into memory
                        with torch.no_grad():
                            angle = 2 * np.pi / self.nsectors * j
                            rot_sin = np.sin(angle)
                            rot_cos = np.cos(angle)
                            rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                                     device=torch.cuda.current_device(), dtype=torch.float32).unsqueeze(
                                0).repeat((bs, 1, 1))
                            grid = (rot_mat_T @ self.grids[i]).view((bs, 2, self.Hs[i], self.Ws[i])).permute(
                                (0, 2, 3, 1))
                            grid[:, :, :, 0] -= self.center[0]
                            grid[:, :, :, 0] /= (self.max_corner[0] - self.min_corner[0]) / 2
                            grid[:, :, :, 1] -= self.center[1]
                            grid[:, :, :, 1] /= (self.max_corner[1] - self.min_corner[1]) / 2

                        tmp = nn.functional.grid_sample(cur_sweep[i], grid)
                        with torch.no_grad():
                            mask = (tmp.abs() > 0).to(int)
                        self.memory[i] = self.memory[i] * (1 - mask) + tmp * mask

                        # get j+1 th sector from memory
                        if j < (self.nsectors - 1):
                            grid_sectors = self.sector_grids[i][j]
                            cur_sweep[i] = nn.functional.grid_sample(self.memory[i],
                                                                     grid_sectors.repeat((bs, 1, 1, 1)))

                    data['prev_sweep'] = cur_sweep
                x2 = torch.cat(x2)
                self.memory = [None for _ in self.num_filters]

            if self.bbox_head:
                preds.update(self.bbox_head(x2))

            if self.seg_head:
                preds.update(self.seg_head(x1, x2))

        if mode == 'train':
            ret_dict = {}
            if self.bbox_head:
                ret_dict.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                ret_dict.update(self.seg_head.loss(example, preds))
            if self.part_head:
                self.part_head(preds)
                ret_dict.update(self.part_head.loss(example, preds))
            return ret_dict
        else:
            # torch.cuda.synchronize()
            # start = time.time()
            rets = [{} for _ in range(self.nsectors)]

            bs = len(example['metadata']) // self.nsectors
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
                    # rets[i].update({'det': self.bbox_head.debug_gt(ex, pr, self.test_cfg, **kwargs)})
                if self.seg_head:
                    rets[i].update({'seg': self.seg_head.predict(ex, preds, self.test_cfg)})
            return rets