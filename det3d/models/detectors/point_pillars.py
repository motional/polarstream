from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
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
        super(PointPillars, self).__init__(
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

        x2 = self.neck(x1)

        return x1, x2

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

        x2 = self.neck(x1)
        return x1, x2

    def forward(self, example, return_loss=True, **kwargs):
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

        x1, x2 = extract_feat(data)
        preds = {}
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

        x = self.extract_feat(data)
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
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None


