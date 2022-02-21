import torch

from torch import nn
from torch.nn import functional as F
from det3d.models.utils import Sequential
from ..registry import NECKS
from ..utils import build_norm_layer
from .rpn import RPN

class ConvContext(nn.Module):
    """
    Convolution compatible with zero-padding and trailing-edge padding.
    param inplanes: int. in channels
    param outplanes: int. out channels
    param kernel: tuple or int. kernel size
    param stride: tuple or int.
    param padding: int
    param bias: bool
    param norm_cfg: dict
    """
    def __init__(self, inplanes, outplanes, kernel, stride, padding, bias, norm_cfg):
        super(ConvContext, self).__init__()
        self.block = Sequential(
            nn.Conv2d(inplanes, outplanes, kernel, stride=stride, bias=bias),
            build_norm_layer(norm_cfg, outplanes)[1],
            nn.ReLU(),
        )
        self.padding = padding
    def forward(self, input):
        """
        param input: dict
        return dict
        """
        x = input['input']
        input['cur_context'].append(x[:, :, -self.padding:, :])
        if len(input['prev_context']) == 0:
            x = F.pad(x, (1, 1, 1, 1))
        else:
            prev_context = input['prev_context'].pop(0)
            x = torch.cat([prev_context, x], 2)
            x = F.pad(x, (1, 1, 0, self.padding))
        input['input'] = self.block(x)
        return input


@NECKS.register_module
class RPNTECP(RPN):
    """
    RPN with trailing-edge padding
    """
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
        super(RPNTECP, self).__init__(layer_nums,ds_layer_strides,ds_num_filters,us_layer_strides,
                                       us_num_filters,num_input_features,norm_cfg,name,logger,**kwargs)
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            ConvContext(inplanes, planes, 3, stride, 1, False, self._norm_cfg)
        )

        for _ in range(num_blocks):
            block.add(ConvContext(planes, planes, 3, 1, 1, False, self._norm_cfg))

        return block, planes

    def forward(self, x, prev_context=[], sec_id=0):
        """
        param x: tensor
        param prev_context: list[tensor]. same length as #conv here
        param sec_id: int. current sector id
        return output tensor, features to pad next sector as list[tensor]
        """
        ups = []
        data_bundle = {'input': x, 'prev_context': prev_context, 'cur_context': []}
        for i in range(len(self.blocks)):
            data_bundle = self.blocks[i](data_bundle)
            x = data_bundle['input']
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x, data_bundle['cur_context']


class ConvBDCP(ConvContext):
    """
    Convolution with bidirectional padding
    param inplanes: int. in channels
    param outplanes: int. out channels
    param kernel: tuple or int. kernel size
    param stride: tuple or int.
    param padding: int
    param bias: bool
    param norm_cfg: dict
    param nsectors: #sectors in a scene
    """
    def __init__(self, inplanes, outplanes, kernel, stride, padding, bias, norm_cfg, nsectors):
        super(ConvBDCP, self).__init__(inplanes, outplanes, kernel, stride, padding, bias, norm_cfg)
        self.nsectors = nsectors
  
    def forward(self, input):
        x = input['input']
        mode = input['mode']
        input['cur_context'].append(x)
        if mode == 'feature_only':
            
            nsectors = input['nsectors']
            if nsectors == 1:
                x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')
            else:
                x = x.reshape([nsectors, x.shape[0]//nsectors, x.shape[1], x.shape[2], x.shape[3]])
                tmp = torch.cat((x[:-1, :, :, -self.padding:, :], x[1:]), -2) # [7, 4, 128, 32, 512]
                x = torch.cat((tmp, F.pad(x[-1:], (0, 0, 0, self.padding))), 0)
                tmp = torch.cat((x[:-1], x[1:, :, :, :self.padding:, :]), -2)
                x = torch.cat((F.pad(x[:1], (0, 0, self.padding, 0)), tmp), 0)
                x = x.reshape((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
            x = F.pad(x, (self.padding, self.padding, 0, 0))
        else:
            layer_id = input['layer_id']
            prev_sweep = input['prev_sweep'][layer_id]
            input['layer_id'] = (layer_id + 1) % len(input['prev_sweep'])
            full_az = prev_sweep.shape[-2]
            az = x.shape[-2]
            nsectors = full_az // az
            # full sweep
            if nsectors == 1:
                x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')
            # streaming
            else:
                sec_id = input['sec_id']

                if sec_id == 0:
                    if self.nsectors == nsectors:
                        x = torch.cat([prev_sweep[:, :, -self.padding:, :], x, prev_sweep[:, :, (sec_id+1)*az:((sec_id+1)*az+self.padding):, :]], 2)
                    else:
                        x = F.pad(x, (0, 0, self.padding, 0))
                        x = torch.cat([x, prev_sweep[:, :, (sec_id+1)*az:((sec_id+1)*az+self.padding):, :]], 2)
                elif sec_id == nsectors - 1:
                    prev_context = input['prev_context'].pop(0)
                    if self.nsectors == nsectors:
                        x = torch.cat([prev_context[:, :, -self.padding:, :], x, prev_sweep[:, :, :self.padding, :]], 2)
                    else:
                        x = F.pad(x, (0, 0, 0, self.padding))
                        x = torch.cat([prev_context[:, :, -self.padding:, :], x], 2)
                else:
                    prev_context = input['prev_context'].pop(0)
                    x = torch.cat([prev_context[:, :, -self.padding:, :], x, prev_sweep[:, :, (sec_id+1)*az:((sec_id+1)*az+self.padding):, :]], 2)
            x = F.pad(x, (self.padding, self.padding, 0, 0))   

        input['input'] = self.block(x)
        return input

@NECKS.register_module
class RPNBDCP(RPNTECP):
    """
    RPN with bidirectional padding.
    """
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
        self.nsectors = kwargs.get('nsectors', 1)
        super(RPNBDCP, self).__init__(layer_nums,ds_layer_strides,ds_num_filters,us_layer_strides,
                                       us_num_filters,num_input_features,norm_cfg,name,logger,**kwargs)
        
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            ConvBDCP(inplanes, planes, 3, stride, 1, False, self._norm_cfg, self.nsectors)
        )

        for _ in range(num_blocks):
            block.add(ConvBDCP(planes, planes, 3, 1, 1, False, self._norm_cfg, self.nsectors))

        return block, planes
    
    def forward(self, x, prev_sweep=[], prev_context=[], sec_id=0, nsectors=1, mode='feature_only'):
        """
                param x: tensor
                param prev_context: list[tensor]. same length as #conv here
                param sec_id: int. current sector id
                param mode: string. 'feature only', 'training' or 'eval'
                return output tensor, features to pad next sector as list[tensor]
                """
        ups = []
        data_bundle = {'input': x, 'prev_context': prev_context, 'cur_context': [], 'prev_sweep': prev_sweep,
                       'mode': mode, 'sec_id': sec_id, 'layer_id': 0, 'nsectors': nsectors}
        for i in range(len(self.blocks)):
            data_bundle = self.blocks[i](data_bundle)
            x = data_bundle['input']
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x, data_bundle['cur_context']

