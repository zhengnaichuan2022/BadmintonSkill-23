_base_ = [
    "../../_base_/datasets/beiyu/features_rgb_sw.py",
    "../../_base_/models/tridet.py",
]

annotation_path = "data/beiyu/annotations/xsub/anno.json"
dataset = dict(
    train=dict(ann_file=annotation_path),
    val=dict(ann_file=annotation_path),
    test=dict(ann_file=annotation_path),
)
evaluation = dict(ground_truth_filename=annotation_path)

model = dict(
    projection=dict(in_channels=2048),
    rpn_head=dict(num_classes=11),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=35)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=25,
)

work_dir = "exps/beiyu/xsub/tridet_rgb_i3d"
