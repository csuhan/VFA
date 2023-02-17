_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[85000, 100000])
evaluation = dict(interval=10000)
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=110000)
optimizer = dict(lr=0.005)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60)))