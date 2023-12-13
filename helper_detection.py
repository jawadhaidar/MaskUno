import torch 
from mmcv.ops.nms import NMSop
from mmengine.visualization import*
from  mmdet.models.task_modules.coders import*
import torchvision.transforms as T
import cv2
from mmdet.registry import MODELS, TASK_UTILS
from helper_functions import*
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmengine.config import ConfigDict

def all2uno(clss,deltas,roi_boxes_selected_2,roi_boxes_ids,scores):
    print(f'cls shape {clss.shape}')
    classID=0 #TODO: bit weird to keep the 0 and not the 1 
    keep= clss==classID
    clss=torch.ones_like(clss[keep])
    
    print(f'deltas,roi_boxes_selected_2,roi_boxes_ids,scores {deltas.shape,roi_boxes_selected_2.shape,roi_boxes_ids.shape,scores.shape}')
    print(f'clss {clss.shape}')
    deltas,roi_boxes_selected_2,roi_boxes_ids,scores=deltas[keep,:],roi_boxes_selected_2[keep,:],roi_boxes_ids[keep],scores[keep]
    return clss,deltas,roi_boxes_selected_2,roi_boxes_ids,scores

def detectUno(detuno,det_score,activation_det,img,img_size_org,img_size_trans,classID):
    #get rois_bbxs

    bbox_coder= dict(
    type='DeltaXYWHBBoxCoder',
    clip_border=True,
    target_means=[0., 0., 0., 0.],
    target_stds=[0.1, 0.1, 0.2, 0.2])
    bbox_coder = TASK_UTILS.build(bbox_coder)

    #aplly thr one


    #get ids_selected_nms 
    roi_boxes_ids=torch.load("/home/jawad/codes/MaskUno/roi_save/roi_ids.pt")
    
    roi_boxes_selected_1,roi_selected_1=first_selection(activation_det)
    #save the first box to compare with their
    # torch.save(roi_boxes_selected_1[0,:],"/home/jawad/codes/MaskUno/compare/myboxid1.pt")
    #stage two selection 
    roi_boxes_selected_2=torch.zeros(roi_boxes_ids.shape[0],4).to(device="cuda") # 4 or 5 sape or len
    roi_selected_2=torch.zeros(roi_boxes_ids.shape[0],1024).to(device="cuda")
    # print(f'roi_boxes_selected {roi_boxes_selected_2.shape} roi_selected {roi_selected_2.shape}')
    for counter,id in enumerate(roi_boxes_ids):
        roi_boxes_selected_2[counter,:]=roi_boxes_selected_1[id,:]

        roi_selected_2[counter,:]=roi_selected_1[id,:]
    
    # torch.save(roi_boxes_selected_2[0,:],"/home/jawad/codes/MaskUno/compare/myboxid2.pt")
    # print(img.shape)
    # img_sh=torch.load("/home/jawad/codes/MaskUno/roi_save/image_shape.pt")
    img_sh=torch.load("/home/jawad/codes/MaskUno/roi_save/img_shape_trans.pt")[:2]

    # print(f'loaded image shape {img_sh}')
    # transform = T.Resize(size = img_sh[:2])

    # apply the transform on the input image
    # img = transform(img)
    res = cv2.resize(img, dsize=(img_sh[1],img_sh[0]), interpolation=cv2.INTER_CUBIC)
    # print(f'res shape {res.shape}')
    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(roi_boxes_selected_2, edge_colors='g')
    # vis.get_image()
    # vis.show()
        # pbbxs labels m

    #inference box predictiob head ---> deltas 
    deltas=detuno(roi_selected_2) #might give scores as well
    # print(f'shape of deltas {deltas.shape}') # torch.Size([2, 4])
    ou=det_score(roi_selected_2)#SxC
    #try to remoove the background score
    # print(f' ou shape {ou.shape}') # 
  
    # print(f'ou.shape {ou.shape}')
    # clss1= torch.argmax(ou,dim=1) #TODO: might need softmax first
    #apply softmax
    softmax=torch.nn.Softmax(dim=1)
    ou=softmax(ou)
    scores,clss=torch.max(ou,dim=1) #the issue in classes #TODO: 0 is the class 1 is the background
    # clss=torch.load("/home/jawad/codes/MaskUno/roi_save/clss.pt")
    print(f'clss shapee {clss.shape}')
    # if clss1.shape!=clss.shape:
    #     print("entered here")
    #     print(f'clss1 my {clss1.shape} clss {clss.shape}')
    #     clss=clss1
    clss,deltas,roi_boxes_selected_2,roi_boxes_ids,scores=all2uno(clss,deltas,roi_boxes_selected_2,roi_boxes_ids,scores)
    print(f'clss after keep : {clss}')
    deltas_selected=torch.zeros(roi_boxes_ids.shape[0],4).to(device="cuda")
    for counter,id in enumerate(roi_boxes_ids): #clls before sort 0 1 2 3 4 5 6  7 8
        start=0#((clss[counter])*4)  #1 : 4:8 #TODO:  check if 0
        # print(start)
        deltas_selected[counter,:]=deltas[counter,start:start+4]
    #add deltas to selected_roi_boxes 
    # boxes=roi_boxes_selected_2 + deltas_selected #not addition
    # print(f' roi_boxes_selected_2 {roi_boxes_selected_2[0,:]}+ deltas_selected {deltas_selected[0,:]} = {boxes[0,:]}')
    #TODO: check if we need to reformat
    boxes=bbox_coder.decode(roi_boxes_selected_2, deltas_selected, max_shape=img_sh)
    # print(f'boxes {boxes[0,:]}')
    #(800, 1067)

    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(boxes, edge_colors='g')
    # vis.get_image()
    # vis.show()

    boxes_sorted=boxes
    clss_sorted=clss
    boxes=resize_boxes(boxes_sorted,img_size_trans, img_size_org) # trans: torch.Size([800, 1088])
    

    return boxes,clss_sorted,scores



def resize_boxes(boxes, original_size, new_size):
    
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    #print(f"inside util {boxes}")
    xmin, ymin, xmax, ymax =boxes.unbind(1)#boxes[0],boxes[1] ,boxes[2] ,boxes[3]  #boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
   
    return torch.stack((xmin, ymin, xmax, ymax), dim=1) #torch.tensor([xmin, ymin, xmax, ymax])#torch.stack((xmin, ymin, xmax, ymax), dim=1)


def detectall(det_model,det_score,activation_det,img,img_size_org,img_size_trans):
    #get rois_bbxs

    bbox_coder= dict(
    type='DeltaXYWHBBoxCoder',
    clip_border=True,
    target_means=[0., 0., 0., 0.],
    target_stds=[0.1, 0.1, 0.2, 0.2])
    bbox_coder = TASK_UTILS.build(bbox_coder)

    #aplly thr one
    roi_boxes=torch.load("/home/jawad/codes/MaskUno/roi_save/roi.pt")
    roi_boxes=roi_boxes[:,1:]

    #get ids_selected_nms 
    roi_boxes_ids=torch.load("/home/jawad/codes/MaskUno/roi_save/roi_ids.pt")
    
    roi_boxes_selected_1,roi_selected_1=first_selection(activation_det)
    # img_sh=torch.load("/home/jawad/codes/MaskUno/roi_save/image_shape.pt")
    img_sh=torch.load("/home/jawad/codes/MaskUno/roi_save/img_shape_trans.pt")[:2]
    #img_shape_trans not always equal img_sh !!!
    print(f'img shapee loaded {img_sh}')
    res = cv2.resize(img, dsize=(img_sh[1],img_sh[0]), interpolation=cv2.INTER_CUBIC)

    # #all proposlas
    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(roi_boxes, edge_colors='y')
    # vis.get_image()
    # vis.show()
    # #first selection
    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(roi_boxes_selected_1, edge_colors='y')
    # vis.get_image()
    # vis.show()
    #save the first box to compare with their
    # torch.save(roi_boxes_selected_1[0,:],"/home/jawad/codes/MaskUno/compare/myboxid1.pt")
    #stage two selection 
    roi_boxes_selected_2=torch.zeros(roi_boxes_ids.shape[0],4).to(device="cuda") # 4 or 5 sape or len
    roi_selected_2=torch.zeros(roi_boxes_ids.shape[0],1024).to(device="cuda")
    # print(f'roi_boxes_selected {roi_boxes_selected_2.shape} roi_selected {roi_selected_2.shape}')
    for counter,id in enumerate(roi_boxes_ids):
        roi_boxes_selected_2[counter,:]=roi_boxes_selected_1[id,:]

        roi_selected_2[counter,:]=roi_selected_1[id,:]
    
    # torch.save(roi_boxes_selected_2[0,:],"/home/jawad/codes/MaskUno/compare/myboxid2.pt")
    print(img.shape)
 
    print(f'res shape {res.shape}')
    #proposals 
    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(roi_boxes_selected_2, edge_colors='y')
    # vis.get_image()
    # vis.show()
    # pbbxs labels m

    #inference box predictiob head ---> deltas 
    deltas=det_model(roi_selected_2) #might give scores as well
    ou=det_score(roi_selected_2)#SxC
    #try to remoove the background score
    ou=ou[:,:-1]
    # print(f'ou.shape {ou.shape}')
    # clss= torch.argmax(ou,dim=1) #TODO: might need softmax first
    #apply softmax
    softmax=torch.nn.Softmax(dim=1)
    ou=softmax(ou)
    scores,clss1=torch.max(ou,dim=1) #the issue in classes
    clss=torch.load("/home/jawad/codes/MaskUno/roi_save/clss.pt")
    # if clss1.shape!=clss.shape:
    #     print("entered here")
    #     print(f'clss1 my {clss1.shape} clss {clss.shape}')
    #     clss=clss1
    # print(f'scores in all {scores}') 
    deltas_selected=torch.zeros(roi_boxes_ids.shape[0],4).to(device="cuda")#4
    for counter,id in enumerate(roi_boxes_ids): #clls before sort 0 1 2 3 4 5 6  7 8
        start=((clss[counter])*4)  #1 : 4:8 #TODO: the problemmm is that label is  sorted acc to score
        # print(start)
        # print(f'deltas_selected {deltas_selected.shape} deltas {deltas.shape}')
        deltas_selected[counter,:]=deltas[counter,start:start+4]
    #add deltas to selected_roi_boxes 
    # boxes=roi_boxes_selected_2 + deltas_selected #not addition
    # print(f' roi_boxes_selected_2 {roi_boxes_selected_2[0,:]}+ deltas_selected {deltas_selected[0,:]} = {boxes[0,:]}')
    boxes=bbox_coder.decode(roi_boxes_selected_2, deltas_selected, max_shape=img_sh)
    # print(f'boxes {boxes[0,:]}')
    #(800, 1067)
    #after decode before scale 
    # vis = Visualizer()
    # vis.set_image(res)
    # vis.draw_bboxes(boxes, edge_colors='b')
    # vis.get_image()
    # vis.show()
    # print(f'scores {scores.shape}boxes {boxes.shape} clss {clss.shape} ')
    #sort boxes
    # sbc=zip(scores,boxes,clss)
    # print(sbc)
    # sbc=sorted(sbc)
    # print(sbc)
    # count=0
    # boxes_sorted=torch.zeros_like(boxes)
    # clss_sorted=torch.zeros_like(clss)
    # for score,box,cls in sbc:
    #     print(score,box)
    #     boxes_sorted[count,:]=box
    #     clss_sorted[count]=cls
    #     count+=1
    # # print(boxes_sorted)

    # print(boxes_sorted.shape)
    #scale bounding bxs
    # print(f"boxes shap {boxes.shape}")
    # print(f'boxes before scale {boxes}')
    boxes_sorted=boxes
    clss_sorted=clss
    boxes=resize_boxes(boxes_sorted,img_size_trans, img_size_org) # trans: torch.Size([800, 1088])
    

    return boxes,clss_sorted,scores

#(0.5, 0, 0, -1) {iou_threshold,offset,score_threshold,max_num}

def first_selection(activation_det):
    labels=torch.arange(80)
    roi_boxes=torch.load("/home/jawad/codes/MaskUno/roi_save/roi.pt")
    roi_boxes=roi_boxes[:,1:]


    #get roi using activation
    roi=activation_det['roi_head.bbox_head.fc_reg'][0]
    #first selection process
    #0-79 --> 0
    #80-159-->1
    id1s=torch.load("/home/jawad/codes/MaskUno/roi_save/first_inds.pt") #0:79,999
    # print(f'id1s before//80 {id1s}')
    id1s=id1s//80 #0:999
    # print(id1s)

    roi_boxes_selected=torch.zeros(id1s.shape[0],4).to(device="cuda") # 4 or 5 sape or len
    roi_selected=torch.zeros(id1s.shape[0],1024).to(device="cuda")
    # print(f'roi_boxes_selected {roi_boxes_selected.shape} roi_selected {roi_selected.shape}')
    for counter,id in enumerate(id1s):
        roi_boxes_selected[counter,:]=roi_boxes[id,:]

        roi_selected[counter,:]=roi[id,:]

    return roi_boxes_selected,roi_selected


#apply id1 and id2 to out boxes (before nms) to see if you get the same output
def labels_test():
    ids1=torch.load("/home/jawad/codes/MaskUno/roi_save/first_inds.pt") 
    ids2=torch.load("/home/jawad/codes/MaskUno/roi_save/roi_ids.pt")
    scores=torch.zeros((1000,80))
    num_classes=80
    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)
    print(labels.shape)
    # labels = labels.reshape(-1)
    # print(labels[80:80+80])
    # print(labels.shape)
    labels_selected=torch.zeros(ids1.shape[0]).to(device="cuda")
    for counter,id in enumerate(ids1):
        labels_selected[counter]=labels[id]
        
    labels_selected2=torch.zeros(ids2.shape[0]).to(device="cuda")
    for counter,id in enumerate(ids2):
        labels_selected2[counter]=labels_selected[id]

    classes=torch.load("/home/jawad/codes/MaskUno/roi_save/clss.pt")

    return labels_selected2,classes

def detectall_selectafter(det_model,activation_det):
    roi_boxes=torch.load("/home/jawad/codes/MaskUno/roi_save/roi.pt")
    img_shape=torch.load("/home/jawad/codes/MaskUno/roi_save/image_shape.pt")

    #get roi using activation
    # roi=activation_det['roi_head.bbox_head.fc_reg'][0]
    # bbox_pred=det_model(roi)

    num_classes = 80#1 if self.reg_class_agnostic else self.num_classes
    roi_boxes = roi_boxes.repeat_interleave(num_classes, dim=0)
    # roi_boxes=torch.load('/home/jawad/codes/MaskUno/compare/roi_boxes_repeated.pt')
    bbox_pred = bbox_pred.view(-1, bbox_coder.encode_size)
    # print(f'bbox_pred shape before coder {bbox_pred.shape}')
    # bbox_pred=torch.load('/home/jawad/codes/MaskUno/compare/bbox_pred_view.pt')
    b = bbox_coder.decode(
        roi_boxes[..., 1:], bbox_pred, max_shape=img_shape)
    print(b.shape)
    print(f'  deltas_selected {bbox_pred[1,:]} + roi_boxes {roi_boxes[1,1:]} = {b[1,:]}')

    return b

#import method 
# bhead=BBoxHead(with_avg_pool = False,
#                  with_cls= True,
#                  with_reg= True,
#                  roi_feat_size= 7,
#                  in_channels= 256,
#                  num_classes= 80,
#                  bbox_coder= dict(
#                      type='DeltaXYWHBBoxCoder',
#                      clip_border=True,
#                      target_means=[0., 0., 0., 0.],
#                      target_stds=[0.1, 0.1, 0.2, 0.2]),
#                  predict_box_type= 'hbox',
#                  reg_class_agnostic= False,
#                  reg_decoded_bbox= False,
#                  reg_predictor_cfg= dict(type='Linear'),
#                  cls_predictor_cfg= dict(type='Linear'),
#                  loss_cls= dict(
#                      type='CrossEntropyLoss',
#                      use_sigmoid=False,
#                      loss_weight=1.0),
#                  loss_bbox= dict(
#                      type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
#                  init_cfg= None)

# roi=torch.rand((1000,4)) #roi boxes
# cls_score=torch.rand((1000,320))
# bbox_pred=torch.rand((1000,320))
# img_meta={'img_shape': (600,600)}
# rescale=False
# rcnn_test_cfg= rcnn=dict(
#             score_thr=0.05,#0.05
#             nms=dict(type='nms', iou_threshold=0.5),#char to debug
#             max_per_img=100,
#             mask_thr_binary=0.5)

# bhead._predict_by_feat_single(roi,cls_score,bbox_pred,img_meta,rescale,rcnn_test_cfg)

def detectall_noids(det_model,det_score,activation_det,mask_head,mask_predictor,img_org_size):
    #load roi boxes and feat
    roi_boxes=torch.load("/home/jawad/codes/MaskUno/roi_save/roi.pt")
    roi=activation_det['roi_head.bbox_head.fc_reg'][0]
    img_sh=torch.load("/home/jawad/codes/MaskUno/roi_save/img_shape_trans.pt")[:2]#check this later
    # img_size_trans=torch.load("/home/jawad/codes/MaskUno/roi_save/image_shape.pt")

    img_meta={'img_shape': img_sh}
    rescale=False

    #i will clean later
    bhead=BBoxHead(with_avg_pool = False,
                 with_cls= True,
                 with_reg= True,
                 roi_feat_size= 7,
                 in_channels= 256,
                 num_classes= 1, #all 80 one 1
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
    rcnn_test_cfg=ConfigDict(
            score_thr=0.05,#0.05
            nms=dict(type='nms', iou_threshold=0.5),#char to debug
            max_per_img=100,
            mask_thr_binary=0.5)
 
    #predict bxs
    deltas=det_model(roi) #might give scores as well
    #predict scores,clss
    scores=det_score(roi)

    #aplly nms 
    print(f'inside detectnoids roi_boxes {roi_boxes.shape} deltas {deltas.shape}')
    out=bhead._predict_by_feat_single(roi_boxes,scores,deltas,img_meta,rescale,rcnn_test_cfg)
    boxes=out.bboxes
    labels=out.labels
    scores=out.scores 

    if boxes.shape[0]==0:
        print(f"skip teh mask inference since no bbxs {boxes.shape[0]}")
        image_masks=torch.tensor([]) #TOD0:
    else:
        #mask predictions
        print("get masks")
        masks=mask_from_enhanced_boxes(boxes,mask_head,mask_predictor).squeeze(1) #scale after transform
        print(f'masks.shape before sig {masks.shape}')
        sigmoid=nn.Sigmoid()
        masks=sigmoid(masks)
        boxes=resize_boxes(boxes,img_sh, img_org_size) # trans: torch.Size([800, 1088])
        print(f'labels predicted {labels}')
        #return pboxes,pclss,pscores
        labels=torch.ones_like(labels)
        print(f'labels after {labels}')
        #project masks on images
        image_masks=project_back_on_image(boxes,masks,img_org_size=img_org_size[:2])
    return boxes,scores,labels,image_masks

def mask_from_enhanced_boxes(boxes,mask_head,mask_predictor):
    #build mask extractor
    #mask build extractor
    m=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])

    mask_roi_extractor = MODELS.build(m)
    #add col of zeros
    boxes_roi=torch.zeros(boxes.shape[0],5)
    boxes_roi[:,1:]=boxes
    #get feat
    feats=torch.load("/home/jawad/codes/MaskUno/featextracttest/feat.pt")
    #extract mask feats
    mask_feat=mask_roi_extractor(feats,boxes_roi)
    #pass o to mask head 
    # mask_feat1=mask_head(mask_feat)

    for module in mask_head:
        mask_feat = module(mask_feat)
    #mask then to uno mask head prediction 
    masks=mask_predictor(mask_feat)
    #what about sigmoid?

    #scale up 
    return masks
    #vis
def project_back_on_image(pbbxs,pmasks,img_org_size):
    ''''''
    mask_img_list=[]
    for id1,mask in enumerate(pmasks):

        print([int(np.round(num.detach().cpu().numpy())) for num in pbbxs[id1]])
        instance_img=paste_mask_in_image_my(mask, [int(np.round(num.detach().cpu().numpy())) for num in pbbxs[id1]], img_org_size[-2], img_org_size[-1]) #TODO: o_im_s inverse
        #paste_mask_in_image
        # print(f'instance_img shape {instance_img.shape}')
        # plt.imshow(instance_img.squeeze(0).detach().cpu().numpy())
        # plt.show()
        mask_img_list.append(instance_img.unsqueeze(0)) #TODO: changing threshold gives good acc
    #combine | cat
    if len(mask_img_list)==0:
        masks=torch.zeros(size=(1,img_org_size[-2],img_org_size[-1]))
        print(f'mask shape when there is no mask {masks.shape}') 
    else:
        masks=torch.cat(mask_img_list, 0)  
        print(f'mask shape normal case {masks.shape}') 


    return masks 
def paste_mask_in_image_my(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    print(f'box in mask {box}')
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    # plt.imshow(mask.squeeze(0).detach().cpu().numpy())
    # plt.show()
    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    print(f'size of mask interp: {mask.shape}')
    #draw 
    # print("inside")
    # plt.imshow(mask.squeeze(0).detach().cpu().numpy())
    # plt.show()
    # print(f'mask.dtype {mask.dtype}')
    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    print(f'size of im_mask: {im_mask.shape}')
    # plt.imshow(im_mask.squeeze(0).detach().cpu().numpy())
    # plt.show()
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    print(x_0,x_1,y_0,y_1)
    print((y_0 - box[1]),(y_1 - box[1]),(x_0 - box[0]),(x_1 - box[0]))
    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
    ]

    # print("after")
    # plt.imshow(im_mask.squeeze(0).detach().cpu().numpy())
    # plt.show()

    return im_mask
def inference_5000():
    pass 

#TODO: check why img_shape changes sometimes from trans 
if __name__ == "__main__":
    # bbox_coder=DeltaXYWHBBoxCoder()

    bbox_coder= dict(
        type='DeltaXYWHBBoxCoder',
        clip_border=True,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2])
    bbox_coder = TASK_UTILS.build(bbox_coder)
    
    roi_boxes=torch.load('/home/jawad/codes/MaskUno/compare/roi_boxes_repeated.pt')
    img_shape=torch.load("/home/jawad/codes/MaskUno/compare/image_shape.pt")
    bbox_pred=torch.load('/home/jawad/codes/MaskUno/compare/bbox_pred_view.pt')

    b = bbox_coder.decode(
        roi_boxes[..., 1:], bbox_pred, max_shape=img_shape)
    print(b)
    print(f'  deltas_selected {bbox_pred[0,:]} + roi_boxes {roi_boxes[0,1:]} = {b[0,:]}')

    roi=torch.load('/home/jawad/codes/MaskUno/compare/roi_boxes_repeated.pt')
    bbox_pred=torch.load('/home/jawad/codes/MaskUno/compare/bbox_pred_view.pt')

    bboxes = bbox_coder.decode(
    roi[..., 1:], bbox_pred, max_shape=img_shape)
    print(f'bbox_pred {bbox_pred[0:4,:] } + roi {roi[0:4,1:]} = bboxes {bboxes[0:4,:]}')
