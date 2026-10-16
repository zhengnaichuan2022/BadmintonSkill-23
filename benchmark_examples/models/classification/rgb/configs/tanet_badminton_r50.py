# BadmintonSkill-23 RGB classification — tanet_badminton_r50 (XSUB example)
# Paths: see benchmark/paths_example.py — override BADMINTONSKILL23_ROOT as needed.

import os
import sys

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_RGB_ROOT = os.path.dirname(_CONFIG_DIR)
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CONFIG_DIR)))
sys.path.insert(0, _BENCH_ROOT)
sys.path.insert(0, _RGB_ROOT)

from paths_example import CHECKPOINTS_ROOT, RGB_ANNOTATIONS, RGB_FRAMES_ROOT, rgb_work

custom_imports = dict(
    imports=['badmintonskill23_ext.badmintonskill23_dataset', 'badmintonskill23_ext.transforms'],
    allow_failed_imports=False,
)

_base_ = [
    os.path.join(_RGB_ROOT, 'mmaction2_configs/_base_/models/tanet_r50.py'),
    os.path.join(_RGB_ROOT, 'mmaction2_configs/_base_/default_runtime.py'),
    os.path.join(_RGB_ROOT, 'mmaction2_configs/_base_/schedules/sgd_100e.py'),]

dataset_type = 'RawframeDataset'
protocol = 'xsub'
data_root = str(RGB_FRAMES_ROOT)
ann_file_train = str(RGB_ANNOTATIONS / protocol / 'train_frames.txt')
ann_file_val = str(RGB_ANNOTATIONS / protocol / 'val_frames.txt')
ann_file_test = str(RGB_ANNOTATIONS / protocol / 'test_frames.txt')
max_epoch = 50
file_client_args = dict(io_backend='disk')

model = dict(
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=11),
)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MultiScaleCrop', input_size=224, scales=(1, 0.875, 0.75, 0.66),
         random_crop=False, max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]

val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, ann_file=ann_file_train,
                 data_prefix=dict(img=data_root), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file_val,
                 data_prefix=dict(img=data_root), pipeline=val_pipeline, test_mode=True))

test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file_test,
                 data_prefix=dict(img=data_root), pipeline=test_pipeline, test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[40, 80], gamma=0.1),
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=128)
load_from = str(CHECKPOINTS_ROOT / 'kinetics400/tanet_imagenet-pretrained-r50_8xb8-1x1x8-100e_kinetics400-rgb_20220906-0f4e34d2.pth')
work_dir = str(rgb_work(protocol, 'tanet_badminton_r50'))
