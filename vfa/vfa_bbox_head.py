# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS

from mmfewshot.detection.models.roi_heads.bbox_heads.meta_bbox_head import MetaBBoxHead


@HEADS.register_module()
class VFABBoxHead(MetaBBoxHead):

    @auto_fp16()
    def forward(self, x_agg, x_query):
        if self.with_avg_pool:
            if x_agg.numel() > 0:
                x_agg = self.avg_pool(x_agg)
                x_agg = x_agg.view(x_agg.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x_agg = torch.mean(x_agg, dim=(-1, -2))
            if x_query.numel() > 0:
                x_query = self.avg_pool(x_query)
                x_query = x_query.view(x_query.size(0), -1)
            else:
                x_query = torch.mean(x_query, dim=(-1, -2))
        cls_score = self.fc_cls(x_agg) if self.with_cls else None
        bbox_pred = self.fc_reg(x_query) if self.with_reg else None
        return cls_score, bbox_pred
