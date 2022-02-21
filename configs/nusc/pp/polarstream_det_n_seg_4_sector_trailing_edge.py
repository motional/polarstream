import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
nsweeps = 10
super_tasks = ['det','seg']
# use relative velocity and orientation
rectify=True

voxel_generator = dict(
    range=[0.3, -3.1488, -5.0, 50.476, 3.1488, 3.0],
    voxel_size=[0.098, 0.0123, 8],
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
    voxel_shape='cylinder',
    return_density=True,
    dynamic=True,
    nsectors=4,
)
tasks = [
    dict(num_class=10, class_names=["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle","pedestrian", "traffic_cone"])
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)


# model settings
bbox_head = None
seg_head = None
part_head = None
if 'det' in super_tasks:
    bbox_head = dict(
        type="CenterHeadSinglePos",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.5,
        code_weights=[1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'rot_vel': (2, 2), 'height': (1, 2), 'dim':(3, 2)}, # (output_channel, num_conv)
        voxel_shape=voxel_generator['voxel_shape'],
        voxel_generator=voxel_generator,
    )
if 'seg' in super_tasks:
    seg_head = dict(
        type="SingleConvHead",
        num_classes=16,
        in_channels=512,
        loss=dict(
            type="SegLoss",
            ignore=-1,
        ),
        weight=2,
    )
    
model = dict(
    type="PolarStream",
    pretrained=None,
    reader=dict(
        type="DynamicPFNet",
        num_filters=[64, 128],
        num_input_features=7,
        xyz_cluster=True,
        raz_cluster=True,
        xy_center=True,
        ra_center=True,
        voxel_size=voxel_generator['voxel_size'],
        pc_range=voxel_generator['range']
    ),
    backbone=dict(type="DynamicPPScatter", ds_factor=1),
    neck=dict(
        type="RPNTECP",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[128, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=128,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=bbox_head,
    seg_head=seg_head,
    part_head=part_head,
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    voxel_shape=voxel_generator['voxel_shape'],
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    stateful_nms=True,  # True if using stateful nms else False
    per_class_nms=True,  # True if using per_class_nms else multi-class nms
    rectify=rectify,  # True if using relative velocity and orientation
    panoptic=False,  # True if stateful panoptic fusion else global fusion
    interval=(voxel_generator['range'][4] - voxel_generator['range'][1]) / voxel_generator.get('nsectors', 1), # azimuth range for each sector
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.1,
    pc_range=voxel_generator['range'],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_generator['voxel_size'],
)

# dataset settings
dataset_type = "NuScenesDataset"
data_root = "data/nuScenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/nuScenes/dbinfos_train_%dsweeps_withvelo_seg.pkl" %nsweeps,
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
    voxel_shape=voxel_generator['voxel_shape'],
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    voxel_shape=voxel_generator['voxel_shape'],
    class_names=class_names,
)



train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"],super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"],super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]

train_anno = "data/nuScenes/infos_train_%02dsweeps_withvelo_filter_True.pkl" %nsweeps
val_anno = "data/nuScenes/infos_val_%02dsweeps_withvelo_filter_True.pkl" %nsweeps
test_anno = "data/nuScenes/infos_val_%02dsweeps_withvelo.pkl" %nsweeps

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        super_tasks=super_tasks,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        mode=val_preprocessor['mode'],
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        super_tasks=super_tasks,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        super_tasks=super_tasks,
        mode='test',
        version='v1.0-test',
        transform_type='feature',
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.0075, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1), ('val', 1)]

