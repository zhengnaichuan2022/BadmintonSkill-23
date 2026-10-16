# Auto-generated: Skeleton j+b concat | xsub | gtad
# Task group: 2_jb  in_channels=768
# train schedule: max_epoch=45 val_start=26 batch=2 milestones=[28, 38]

_base_ = ["../../_base_/models/gtad.py"]

dataset_type = "BeiyuSlidingDataset"
annotation_path = "data/beiyu/annotations/xsub/anno.json"
class_map = "data/beiyu/annotations/category_idx.txt"
block_list = "data/beiyu/annotations/missing_features_blocklist.txt"
data_path = [
    "data/beiyu/features/skeleton/protogcn_j_xsub_stride4/",
    "data/beiyu/features/skeleton/protogcn_b_xsub_stride4/",
]
window_size = 256
feature_stride = 4
sample_stride = 1
offset_frames = 0

_load = dict(type="LoadFeats", feat_format=["npy", "npy"])
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
    projection=dict(in_channels=768, num_convs=2, conv_cfg=dict(groups=32)),
    roi_head=dict(proposal_head=dict(num_classes=11)),
)

solver = dict(train=dict(batch_size=2, num_workers=2), val=dict(batch_size=1, num_workers=1), test=dict(batch_size=1, num_workers=1), clip_grad_norm=0)
optimizer = dict(type='Adam', lr=8e-4, weight_decay=1e-4)
scheduler = dict(type='MultiStepLR', max_epoch=45, milestones=[28, 38], gamma=0.1)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    external_cls=dict(
        type='StandardClassifier',
        path='data/beiyu/classifiers/skel_jb_xsub.json',
        topk=11,
        apply_softmax=False,
    ),
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

workflow = dict(logging_interval=20, checkpoint_interval=1, val_loss_interval=1, val_eval_interval=1, val_start_epoch=26)
work_dir = "exps/beiyu/benchmark/xsub/skel_jb_xsub_gtad"
