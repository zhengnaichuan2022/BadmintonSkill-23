# Auto-generated Beiyu video benchmark config (xsub/i3d_badminton_r50)
custom_imports = dict(
    imports=['mmaction_ext.beiyu_dataset', 'mmaction_ext.transforms'],
    allow_failed_imports=False,
)

_base_ = ['/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/models/i3d_r50.py', '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/schedules/sgd_100e.py', '/home/ubuntu/users/dyc/code/mmaction2/configs/_base_/default_runtime.py']

# 
dataset_type = 'RawframeDataset'
data_root = '${BADMINTONSKILL23_ROOT}/video/cache/frames'
ann_file_train = '${BADMINTONSKILL23_ROOT}/video/annotations/xsub/train_frames.txt'
ann_file_val = '${BADMINTONSKILL23_ROOT}/video/annotations/xsub/val_frames.txt'
ann_file_test = '${BADMINTONSKILL23_ROOT}/video/annotations/xsub/test_frames.txt'

file_client_args = dict(io_backend='disk')

# 
model = dict(
    backbone=dict(pretrained=None, pretrained2d=False),
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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
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
        milestones=[20, 40],
        gamma=0.1)
]

# 
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# 
auto_scale_lr = dict(enable=False, base_batch_size=128)

load_from = '${BADMINTONSKILL23_ROOT}/video/checkpoints/kinetics400/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth'
