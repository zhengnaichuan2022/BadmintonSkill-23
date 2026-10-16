_base_ = [
    "../../_base_/datasets/beiyu/features_rgb_sw.py",
    "../../_base_/models/gtad.py",
]

annotation_path = "data/beiyu/annotations/xsub/anno.json"
dataset = dict(
    train=dict(ann_file=annotation_path),
    val=dict(ann_file=annotation_path),
    test=dict(ann_file=annotation_path),
)
evaluation = dict(ground_truth_filename=annotation_path)

model = dict(
    projection=dict(in_channels=2048, num_convs=2, conv_cfg=dict(groups=32)),
    roi_head=dict(proposal_head=dict(num_classes=11)),
)

solver = dict(train=dict(batch_size=2, num_workers=2), val=dict(batch_size=1, num_workers=1), test=dict(batch_size=1, num_workers=1), clip_grad_norm=0)
optimizer = dict(type="Adam", lr=1e-3, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", max_epoch=35, milestones=[20, 30], gamma=0.1)
workflow = dict(logging_interval=20, checkpoint_interval=1, val_loss_interval=1, val_eval_interval=1, val_start_epoch=20)
work_dir = "exps/beiyu/xsub/gtad_rgb_i3d"
