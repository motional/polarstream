import collections

from det3d.utils import build_from_cfg
from ..registry import PIPELINES
import time
import numpy as np
from collections import defaultdict
@PIPELINES.register_module
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Empty':
                    continue 
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")
        self.times = defaultdict(list)
        self.avg_times = dict()

    def __call__(self, res, info):
        for t in self.transforms:
            res, info = t(res, info)
            if res is None:
                return None
        return res, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

