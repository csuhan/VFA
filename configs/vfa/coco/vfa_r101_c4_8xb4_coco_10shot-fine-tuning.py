_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=10,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=10, num_base_shots=10))
evaluation = dict(interval=10000)
checkpoint_config = dict(interval=10000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[10000])
runner = dict(max_iters=10000)
# load_from = 'path of base training model'
load_from = 'work_dirs/vfa_r101_c4_8xb4_coco_base-training/latest.pth'
# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80)),
    with_refine=True,
    frozen_parameters=[
    'backbone', 'shared_head',  'aggregation_layer', 'rpn_head.rpn_conv',
])

# iter 10000
# OrderedDict([('BASE_CLASSES bbox_mAP', 0.309), ('BASE_CLASSES bbox_mAP_50', 0.509), ('BASE_CLASSES bbox_mAP_75', 0.33), ('BASE_CLASSES bbox_mAP_s', 0.165), ('BASE_CLASSES bbox_mAP_m', 0.351), ('BASE_CLASSES bbox_mAP_l', 0.442), ('BASE_CLASSES bbox_mAP_copypaste', '0.309 0.509 0.330 0.165 0.351 0.442'), ('NOVEL_CLASSES bbox_mAP', 0.168), ('NOVEL_CLASSES bbox_mAP_50', 0.363), ('NOVEL_CLASSES bbox_mAP_75', 0.136), ('NOVEL_CLASSES bbox_mAP_s', 0.074), ('NOVEL_CLASSES bbox_mAP_m', 0.176), ('NOVEL_CLASSES bbox_mAP_l', 0.253), ('NOVEL_CLASSES bbox_mAP_copypaste', '0.168 0.363 0.136 0.074 0.176 0.253'), ('bbox_mAP', 0.273), ('bbox_mAP_50', 0.473), ('bbox_mAP_75', 0.282), ('bbox_mAP_s', 0.142), ('bbox_mAP_m', 0.308), ('bbox_mAP_l', 0.395), ('bbox_mAP_copypaste', '0.273 0.473 0.282 0.142 0.308 0.395')])