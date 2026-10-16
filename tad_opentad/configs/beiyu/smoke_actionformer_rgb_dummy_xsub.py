_base_ = [
    "../../_base_/datasets/beiyu/features_rgb_sw.py",
    "../../_base_/models/actionformer.py",
]

# Smoke test with dummy features (run: extract_rgb.py --backend dummy)
data_path = "data/beiyu/features/rgb/dummy_stride4/"

annotation_path = "data/beiyu/annotations/xsub/anno.json"
dataset = dict(
    train=dict(ann_file=annotation_path, data_path=data_path),
    val=dict(ann_file=annotation_path, data_path=data_path),
    test=dict(ann_file=annotation_path, data_path=data_path),
)
evaluation = dict(ground_truth_filename=annotation_path)

model = dict(
    projection=dict(in_channels=512, input_pdrop=0.2),
    rpn_head=dict(num_classes=11),
)

solver = dict(train=dict(batch_size=2, num_workers=0), val=dict(batch_size=1, num_workers=0), test=dict(batch_size=1, num_workers=0), clip_grad_norm=1, ema=False)
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=2)
workflow = dict(logging_interval=5, checkpoint_interval=1, val_loss_interval=1, val_eval_interval=1, val_start_epoch=1)
work_dir = "exps/beiyu/xsub/smoke_actionformer_rgb_dummy"
