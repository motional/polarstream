from det3d.utils import build_from_cfg
from torch import nn

from .registry import (
    BACKBONES,
    DETECTORS,
    BBOX_HEADS,
    SEG_HEAD,
    LOSSES,
    NECKS,
    READERS,
    SECOND_STAGE,
    ROI_HEAD
)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_second_stage_module(cfg):
    return build(cfg, SECOND_STAGE)

def build_roi_head(cfg):
    return build(cfg, ROI_HEAD)


def build_reader(cfg):
    return build(cfg, READERS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)

def build_bbox_head(cfg):
    return build(cfg, BBOX_HEADS)

def build_seg_head(cfg):
    return build(cfg, SEG_HEAD)

def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
