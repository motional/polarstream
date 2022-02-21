from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.times = []

    def extract_feat_hard(self, data):

        input_features = self.reader(data["features"], data["num_voxels"])

        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def extract_feat_dynamic(self, data):
        input_features, unq = self.reader(data)
        x, voxel_feature = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"]
        )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
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
            extract_feat = self.extract_feat_hard
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

        if self.seg_head:
            x, voxel_feature = extract_feat(data)
            preds.update(self.seg_head(voxel_feature['conv1'].dense(), x))


        else:
            x, _ = extract_feat(data)
        if self.bbox_head:
            preds.update(self.bbox_head(x))
        if return_loss:
            loss = {}
            if self.bbox_head:
                loss.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                loss.update(self.seg_head.loss(example, preds))
            return loss
        else:
            ret_dict = {}

            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg)
            if self.seg_head:
                ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

            return ret_dict

    def forward_two_stage(self, example, return_loss=True, **kwargs):
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
        )

        x, voxel_feature = self.extract_feat(data)

        bev_feature = x
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None

