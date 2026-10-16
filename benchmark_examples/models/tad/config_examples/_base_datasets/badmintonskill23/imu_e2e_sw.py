# BadmintonSkill23 IMU end-to-end TAD — sliding window on raw IMU timeline

dataset_type = "BadmintonSkill23IMUSlidingDataset"
class_map = "data/badmintonskill23/annotations/category_idx.txt"
data_path = ""  # unused; IMU CSV paths live in anno.json
block_list = None

window_size = 256
feature_stride = 4
sample_stride = 1
offset_frames = 0

_imu_train_pipeline = [
    dict(type="LoadIMURaw", normalize="window"),
    dict(type="IMUSlidingWindowTrunc", with_mask=True),
    dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
    dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
    dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
]

_imu_eval_pipeline = [
    dict(type="LoadIMURaw", normalize="window"),
    dict(type="IMUSlidingWindowTrunc", with_mask=True),
    dict(type="ConvertToTensor", keys=["feats"]),
    dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
    dict(type="Collect", inputs="feats", keys=["masks"]),
]

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file="data/badmintonskill23/annotations/imu/xsub/anno.json",
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=feature_stride,
        sample_stride=sample_stride,
        window_size=window_size,
        window_overlap_ratio=0.25,
        offset_frames=offset_frames,
        pipeline=_imu_train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file="data/badmintonskill23/annotations/imu/xsub/anno.json",
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=feature_stride,
        sample_stride=sample_stride,
        window_size=window_size,
        window_overlap_ratio=0.25,
        offset_frames=offset_frames,
        pipeline=_imu_train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file="data/badmintonskill23/annotations/imu/xsub/anno.json",
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=feature_stride,
        sample_stride=sample_stride,
        window_size=window_size,
        window_overlap_ratio=0.5,
        offset_frames=offset_frames,
        pipeline=_imu_eval_pipeline,
    ),
)

evaluation = dict(
    type="mAP",
    subset="testing",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename="data/badmintonskill23/annotations/imu/xsub/anno.json",
)
