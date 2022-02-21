import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
super_tasks = ['seg']
nsweeps = 10
rectify=True
voxel_generator = dict(
    #range=[0.3, -3.1488, -5.0, 50.476, 3.1488, 3.0],
    #voxel_size=[0.098/2, 0.0123/2, 0.2],
    range=[0.3, -3.1488, -5.0, 50.86, 3.1488, 3.0],
    voxel_size=[0.079, 0.00984, 0.2],
    max_points_in_voxel=30,
    max_voxel_num=[120000, 180000],
    dynamic=True,
    voxel_shape='cylinder',
    nsectors=1,
)
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
bbox_head = None
seg_head = None
if 'det' in super_tasks:
    bbox_head = dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        share_conv_channel=64,
        dcn_head=False,
        voxel_shape=voxel_generator['voxel_shape'],
    )
seg_weight = 2
if 'det' in super_tasks:
    seg_weight = 10
if 'seg' in super_tasks:
    seg_head = dict(
        type="DeconvConvHead",
        kernel=3,
        num_classes=16,
        in_channels=512,
        in_channels_voxel=16,
        up_scale=8,
        loss=dict(
            type="SegLoss",
            ignore=-1,
        ),
        weight=seg_weight,
        height=40,
    )
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="DynamicVoxelEncoderV1",
        num_input_features=7,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=7, ds_factor=8,extra_sp_shape=[0,0,0],
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=bbox_head,
    seg_head=seg_head,
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
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    rectify=rectify,
    score_threshold=0.1,
    pc_range=voxel_generator['range'],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_generator['voxel_size']
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
    global_rot_noise=[-0.87, 0.87],
    global_scale_noise=[0.7, 1.3],
    global_translate_std=0.5,
    shuffle_points=True,
    db_sampler=db_sampler,
    class_names=class_names,
    voxel_shape=voxel_generator['voxel_shape'],
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    voxel_shape=voxel_generator['voxel_shape'],
)



train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"], super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"], super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]

train_anno = "data/nuScenes/infos_train_%02dsweeps_withvelo_filter_True.pkl" %nsweeps
val_anno = "data/nuScenes/infos_val_%02dsweeps_withvelo_filter_True.pkl" %nsweeps
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
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
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.0025, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
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
workflow = [('train', 1)]
