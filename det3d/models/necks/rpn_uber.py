import torch
from torch import nn
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.registry import NECKS
from .rpn import RPN

@NECKS.register_module
class RPNUber(RPN):
    def __init__(
            self,
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            us_layer_strides,
            us_num_filters,
            num_input_features,
            norm_cfg=None,
            name="rpn",
            logger=None,
            **kwargs
    ):
        super(RPNUber, self).__init__(layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides,
                                      us_num_filters, num_input_features, norm_cfg, name, logger, **kwargs)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]

        for i in range(1, len(self._layer_nums)):
            memory = Sequential(
                nn.Conv2d(
                    in_filters[i] * 2,
                    in_filters[i] * 2,
                    3,
                    padding=1,
                ),
                nn.GroupNorm(
                    in_filters[i],
                    in_filters[i] * 2,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_filters[i] * 2,
                    in_filters[i],
                    1,
                ),
                nn.GroupNorm(
                    in_filters[i],
                    in_filters[i],
                ),
                nn.ReLU(),
            )
            self.__setattr__('memory' + str(i), memory)

    def forward(self, x, prev_sweeps):
        ups = []
        cur_sweep = []
        for i in range(len(self.blocks)):
            if i > 0:
                if prev_sweeps is not None:
                    x = torch.cat([x, prev_sweeps[i - 1]], 1)
                    x = self.__getattr__('memory' + str(i))(x)
                cur_sweep.append(x)
            x = self.blocks[i](x)

            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x, cur_sweep
