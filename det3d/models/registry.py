from det3d.utils import Registry

READERS = Registry("reader")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
BBOX_HEADS = Registry("bbox_heads")
SEG_HEAD = Registry("seg_heads")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")