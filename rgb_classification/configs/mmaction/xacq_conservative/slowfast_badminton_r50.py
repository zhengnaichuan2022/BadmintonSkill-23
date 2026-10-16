# Auto-generated Beiyu video benchmark config (xacq_conservative/slowfast_badminton_r50)
custom_imports = dict(
    imports=['mmaction_ext.beiyu_dataset', 'mmaction_ext.transforms'],
    allow_failed_imports=False,
)

_base_ = [
    '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/models/slowfast_r50.py', 
    '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/default_runtime.py'
]

# 
dataset_type = 'RawframeDataset'
data_root = '${BADMINTONSKILL23_ROOT}/video/cache/frames'
ann_file_train = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/train_frames.txt'
ann_file_val = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/val_frames.txt'
ann_file_test = '${BADMINTONSKILL23_ROOT}/video/annotations/xacq_conservative/test_frames.txt'
max_epoch = 50

file_client_args = dict(io_backend='disk')

# 
model = dict(
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=11)
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
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

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epoch, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=66,
        eta_min=0,
        by_epoch=True,
        begin=34,
        end=100)
]

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

load_from = '${BADMINTONSKILL23_ROOT}/video/checkpoints/kinetics400/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'
