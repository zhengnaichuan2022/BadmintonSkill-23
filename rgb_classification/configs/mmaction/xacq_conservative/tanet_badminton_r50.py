# Auto-generated Beiyu video benchmark config (xacq_conservative/tanet_badminton_r50)
custom_imports = dict(
    imports=['mmaction_ext.beiyu_dataset', 'mmaction_ext.transforms'],
    allow_failed_imports=False,
)

_base_ = ['/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/models/tanet_r50.py', '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/default_runtime.py', '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/schedules/sgd_tsm_50e.py']

# 
dataset_type = 'RawframeDataset'
data_root = '${BADMINTONSKILL23_ROOT}/video/cache/frames'
ann_file_train = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/train_frames.txt'
ann_file_val = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/val_frames.txt'
ann_file_test = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/test_frames.txt'

file_client_args = dict(io_backend='disk')

# 
model = dict(
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=11, dropout_ratio=0.6)
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

# 
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30, 40, 45],
        gamma=0.1)
]

# 
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001),
    clip_grad=dict(max_norm=40, norm_type=2))

# 
auto_scale_lr = dict(enable=False, base_batch_size=128)

load_from = '${BADMINTONSKILL23_ROOT}/video/checkpoints/kinetics400/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb_20220919-a34346bc.pth'
