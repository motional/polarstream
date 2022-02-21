import torch
from ..registry import NECKS
from .rpn import RPN

@NECKS.register_module
class RPNWaymo(RPN):
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
        super(RPNWaymo, self).__init__(layer_nums,ds_layer_strides,ds_num_filters,us_layer_strides,
                                       us_num_filters,num_input_features,norm_cfg,name,logger,**kwargs)

    def forward(self, x, lstm_out):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i == len(self.blocks) - 1:
                new_lstm_out = x.mean(dim=(-2, -1)).unsqueeze(0)
                if lstm_out is not None:
                    x += lstm_out.view((x.shape[0], -1, 1, 1))
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x, new_lstm_out