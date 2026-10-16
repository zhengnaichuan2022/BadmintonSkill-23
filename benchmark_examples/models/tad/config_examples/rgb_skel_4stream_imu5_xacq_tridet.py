# BadmintonSkill-23 TAD config example — paths are relative placeholders.
# Override via environment: BADMINTONSKILL23_ROOT, BADMINTONSKILL23_EXP_ROOT
# Dataset class: BadmintonSkill23SlidingDataset (see models/tad/opentad_datasets/)

# Auto-generated: RGB R2+1D + Skeleton 4-stream + IMU5 (Inception 256-d) | xacq | tridet
# Task group: 5_4stream  in_channels=2304
# train schedule: max_epoch=60 val_start=45 batch=1 sgp_mlp_dim=1024 lr=5e-5

_base_ = ["../../_base_/models/tridet.py"]

dataset_type = "BadmintonSkill23SlidingDataset"
annotation_path = "example/BadmintonSkill23/tad/annotations/xacq/anno.json"
class_map = "example/BadmintonSkill23/tad/annotations/category_idx.txt"
block_list = "example/BadmintonSkill23/tad/annotations/missing_features_blocklist.txt"
data_path = [
    "example/BadmintonSkill23/tad/features/rgb/r2plus1d_xacq_stride4/",
    "example/BadmintonSkill23/tad/features/skeleton/protogcn_j_xacq_stride4/",
    "example/BadmintonSkill23/tad/features/skeleton/protogcn_b_xacq_stride4/",
    "example/BadmintonSkill23/tad/features/skeleton/protogcn_jm_xacq_stride4/",
    "example/BadmintonSkill23/tad/features/skeleton/protogcn_bm_xacq_stride4/",
    "example/BadmintonSkill23/tad/features/imu/inception5_xacq_stride4/",
]
window_size = 256
feature_stride = 4
sample_stride = 1
offset_frames = 0

_load = dict(type="LoadFeats", feat_format=["npy", "npy", "npy", "npy", "npy", "npy"])
_common = dict(
    type=dataset_type,
    ann_file=annotation_path,
    block_list=block_list,
    class_map=class_map,
    data_path=data_path,
    filter_gt=False,
    feature_stride=feature_stride,
    sample_stride=sample_stride,
    window_size=window_size,
    offset_frames=offset_frames,
)

dataset = dict(
    train=dict(
        **_common,
        subset_name="training",
        window_overlap_ratio=0.25,
        pipeline=[
            _load,
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        **_common,
        subset_name="validation",
        window_overlap_ratio=0.25,
        pipeline=[
            _load,
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        **_common,
        subset_name="testing",
        test_mode=True,
        window_overlap_ratio=0.5,
        pipeline=[
            _load,
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

model = dict(
    projection=dict(in_channels=2304, out_channels=512, sgp_mlp_dim=1024),
    rpn_head=dict(num_classes=11),
)

solver = dict(train=dict(batch_size=1, num_workers=2), val=dict(batch_size=1, num_workers=1), test=dict(batch_size=1, num_workers=1), clip_grad_norm=1, ema=True)
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.05, paramwise=True)
scheduler = dict(type='LinearWarmupCosineAnnealingLR', warmup_epoch=8, max_epoch=60)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,
)

workflow = dict(logging_interval=20, checkpoint_interval=1, val_loss_interval=1, val_eval_interval=1, val_start_epoch=45)
work_dir = "example/experiments/tad/benchmark/xacq/rgb_skel_4stream_imu5_xacq_tridet"
