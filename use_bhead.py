from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
import torch
#import method 
bhead=BBoxHead(with_avg_pool = False,
                 with_cls= True,
                 with_reg= True,
                 roi_feat_size= 7,
                 in_channels= 256,
                 num_classes= 80,
                 bbox_coder= dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 predict_box_type= 'hbox',
                 reg_class_agnostic= False,
                 reg_decoded_bbox= False,
                 reg_predictor_cfg= dict(type='Linear'),
                 cls_predictor_cfg= dict(type='Linear'),
                 loss_cls= dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox= dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg= None)

roi=torch.rand((1000,4)) #roi boxes
cls_score=torch.rand((1000,320))
bbox_pred=torch.rand((1000,320))
img_meta={'img_shape': (600,600)}
rescale=False
rcnn_test_cfg= rcnn=dict(
            score_thr=0.05,#0.05
            nms=dict(type='nms', iou_threshold=0.5),#char to debug
            max_per_img=100,
            mask_thr_binary=0.5)

bhead._predict_by_feat_single(roi,cls_score,bbox_pred,img_meta,rescale,rcnn_test_cfg)


#build from dict method 