# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.6, CUDA 10.1, and CUDNN 7.5.
- The training is conducted on 8 V100 GPUs with. 
- Testing times are measured on a V100 GPU with batch size 1. 
 
## nuScenes 3D Detection 

**We provide training / validation configurations, pretrained models for all models in the paper**

### PolarStream with PointPillars backbone
| Model                 | det FPS          | seg FPS | panoptic FPS |Test MAP  | Test NDS  |  Test mIoU| Test freq_weigted mIoU|Validation PQ  | Validation SQ  | Validation RQ |Link          |
|-----------------------|------------------|---------|--------------|----------|-----------|-----------|-----------------------|---------------|----------------|---------------|---------------|
| [polarstream_det_n_seg_1_sector](configs/nusc/pp/polarstream/polarstream_det_n_seg_1_sector.py) | 26.3 | 33.9 |22.3|52.9 | 61.2 |  73.4| 87.4 |68.7 |85.3|79.9|[URL](https://drive.google.com/file/d/1JVUGjDvg0dOhWdY1wTld0sKgt15as5pJ/view?usp=sharing) |
| [polarstream_det_n_seg_4_sector_bidirectional](configs/nusc/pp/polarstream/polarstream_det_n_seg_1_sector.py) | 47.2 | 59.2 |44.3 | 52.9 | 61.2 |  73.1 | 87.5| 69.6 | 85.5 | 80.8 |[URL](https://drive.google.com/file/d/1P3X2rXY6RvMRILvLiU4a3P-8-9-dHvgn/view?usp=sharing) |

### Reimplementation of [STROBE](https://arxiv.org/abs/2011.06425) and [Han et. al.](https://arxiv.org/abs/2005.01864)
| Model                 | Link          |
|-----------------------|---------------|
| [han_1_sector](configs/nusc/pp/han_method/han_1_sector.py) | [URL](https://drive.google.com/file/d/1_uNb1GlN0oLvoJonvPDoS9uE6oBxIYRl/view?usp=sharing) |
| [han_4_sector](configs/nusc/pp/han_method/han_4_sectors.py) |[URL](https://drive.google.com/file/d/1Mgw0_Ed41EVsLI-xQweDZbH7F0LtXAWi/view?usp=sharing) |
| [strobe_1_sector](configs/nusc/pp/strobe/strobe_1_sector.py) | [URL](https://drive.google.com/file/d/1cUiMrXiKaO_MO2Y4STqVRWnefbBH0ueS/view?usp=sharing) |
| [strobe_4_sector](configs/nusc/pp/strobe/strobe_4_sector.py) |[URL](https://drive.google.com/file/d/1lrkvIenmS3MZysXBwwIn1Rwu6D7OnDde/view?usp=sharing) |