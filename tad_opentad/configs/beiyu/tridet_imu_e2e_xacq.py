_base_ = [
    "../_base_/datasets/beiyu/imu_e2e_sw.py",
    "../_base_/models/tridet_imu_e2e.py",
]

annotation_path = "data/beiyu/annotations/imu/xacq/anno.json"
dataset = dict(
    train=dict(ann_file=annotation_path),
    val=dict(ann_file=annotation_path),
    test=dict(ann_file=annotation_path),
)
evaluation = dict(ground_truth_filename=annotation_path)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=1e-4, weight_decay=0.05),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=40)

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
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=20,
)

work_dir = "exps/beiyu/imu_e2e/xacq/tridet_inception"
