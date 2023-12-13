import mmdet 
import mmdet.apis
import torchinfo
import torch 
import torch
from model import*
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms 
from data_class import*
# from helper_functions import*
import cv2 as cv
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import sys
from mmengine.visualization import*
from helper_detection import* 
from holistic import*

# sys.path.append(r'/home/jawad/codes/references/detection')
# print(sys.path)
#from engine import train_one_epoch, evaluate 
def drawfn(img,selected_box,selected_score):
    selected_box=selected_box.detach().cpu().numpy()
    for index,box in enumerate(selected_box):
        # print(box)
        xmin,ymin,xmax,ymax=int(np.round(box[0],2)),int(np.round(box[1],2)),int(np.round(box[2],2)),int(np.round(box[3],2))
        # print(xmin,ymin,xmax,ymax)
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 4)
        #put score
        cv.putText(img,f'score {"{:.2f}".format(selected_score[index])}',(xmin+5,ymin+10),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

    cv.imshow("img",img)
    cv.waitKey(10000)
    cv.destroyAllWindows()

device="cuda"

def collate_fn(batch):
    return tuple(zip(*batch))

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = input #.detach() #output.detach()
    return hook
    
# confg_file="/home/jawad/mmdetection/myconfigs/htc.py"
# checkpoint_file="/home/jawad/Downloads/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"

# confg_file="/home/jawad/mmdetection/myconfigs/detectors.py"
# checkpoint_file="/home/jawad/mmdetection/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth"
# confg_file="/home/jawad/codes/MaskUno/detectors_v2.py"
# checkpoint_file="/home/jawad/Downloads/detectors_htc_r50_1x_coco-329b1453.pth"
# confg_file="/home/jawad/codes/MaskUno/mask_config_uno.py"
# checkpoint_file="/home/jawad/codes/epoch_11.pth" 

confg_file="/home/jawad/codes/MaskUno/mask_config_all.py"
checkpoint_file="/home/jawad/mmdetection/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"

model=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")

# model.roi_head.register_forward_hook(get_activation('roi_head'))
activation_value_detection='roi_head.bbox_head.fc_reg'
model.roi_head.bbox_head.fc_reg.register_forward_hook(get_activation('roi_head.bbox_head.fc_reg'))

model.roi_head.mask_head.upsample.register_forward_hook(get_activation('roi_head.mask_head.upsample'))
#for image size
model.backbone.register_forward_hook(get_activation('backbone.layer1.0'))
#backbone.layer1.0
#get the uno model 
confg_file="/home/jawad/codes/MaskUno/mask_config_uno.py" #change num classes into one
checkpoint_file="/home/jawad/codes/epoch_11_unocat.pth"    
# checkpoint_file='/home/jawad/codes/MaskUno/epoch_22_unocow.pth'
# checkpoint_file='/home/jawad/codes/MaskUno/epoch_18_bird.pth'
# checkpoint_file='MaskUno/epoch_17_cat0.1pos.pth'
# checkpoint_file="/home/jawad/codes/MaskUno/epoch_5_dog.pth"
# checkpoint_file="/home/jawad/codes/MaskUno/epoch_1_cat_5000.pth"
detectionuno=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
# get the reg part 
detuno=detectionuno.roi_head.bbox_head.fc_reg #just for exp
detuno_score=detectionuno.roi_head.bbox_head.fc_cls

det=model.roi_head.bbox_head.fc_reg #just for exp
det_score=model.roi_head.bbox_head.fc_cls
#get mask head
mask_head=model.roi_head.mask_head.convs
#get mask uno maks prediction head
modelClass=nn.Sequential(detectionuno.roi_head.mask_head.upsample,
                         detectionuno.roi_head.mask_head.relu,
                         detectionuno.roi_head.mask_head.conv_logits                               
                    )
model.eval()
for name, param in model.named_parameters():
    print(name)

print(model)

#load data 
# img=torch.ones(size=(1000,900,3)).to(device="cuda")
# # print(model((img,img)))
# out=mmdet.apis.inference_detector(model,img.detach().cpu().numpy())
# print(out)
# features=activation['roi_head.mask_head.2.convs.3.conv']
# print(features.shape)

#dataloader 
mytransform=transforms.Compose([
    # transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float)
])


dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name='all')
data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
#holistic postprocess
hol=Holistic()
for i, batched_sample in enumerate(data_loader):
    images,targets=batched_sample
    # images_before=list(image.to(device=device)  for image in images)

    # images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
    # targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] 
    for img in images:
        img_org_size=img.shape
        #img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
        print(f'the shape of img {img.shape}')
        # print(model.data_preprocessor(images))
        out=mmdet.apis.inference_detector(model,img)
        #image size after trans
        img_trans=activation["backbone.layer1.0"][0]
        img_trans_size=img_trans.shape[-2:]
        print(f'the shape of the image img_trans_size {img_trans_size}')
       
        features=activation['roi_head.mask_head.upsample'][0] #([14, 256, 14, 14])

        box_features=activation[activation_value_detection][0]
        classID=15
        pbbxs=out.pred_instances.bboxes
        labels=out.pred_instances.labels
        masks=out.pred_instances.masks
        scores=out.pred_instances.scores #3scores are ordered here
        keep=labels==classID

        # print(f' Before keep pbbxs labels masks scores : {pbbxs.shape} {labels.shape} {masks.shape} {scores.shape} ')
        # print(f'sahpe of hook predicted bxs shape {bxs_h.shape}') #N,Cx4 #torch.Size([1000, 320])
        r=torch.load("/home/jawad/codes/MaskUno/roi_save/roi.pt") #torch.Size([1000, 5])
        # print(f'read from saved roi_boxes {r.shape}')
        #add the deltas to rois 
        # bbx_after_add= r + bxs_h[:,:5] #we should choose the right range 
        # print(f' sahpe of addddddddition {bbx_after_add.shape} value {bbx_after_add}')
        roi_ids=torch.load("/home/jawad/codes/MaskUno/roi_save/roi_ids.pt")
        # print(f'loaded roi_ids {roi_ids.shape}')
        img_shape_trans=torch.load("/home/jawad/codes/MaskUno/roi_save/img_shape_trans.pt")[:2]
       
        print(f'shape of box_features {box_features.shape}')
        print(f'img_trans_size act {img_trans_size} vs img_shape_trans {img_shape_trans}')
        # boxes,clss_sorted,scores=detectUno(detuno,detuno_score,activation,img,img_org_size,img_shape_trans,classID)
        # boxes,clss_sorted,scores=detectall(det,det_score,activation,img,img_org_size,img_shape_trans) #out is not ordered acc to scores but to index oreder
        # keep1= clss_sorted==classID
        # print(keep2)
        # detectall_selectafter(det,activation)
        pbbxs_my,scores_my,labels_my,masks=detectall_noids(detuno,detuno_score,activation,mask_head,modelClass,img_org_size)
        #mask
       
        # for mask in masks:
        #     mask=mask>0.5
        #     plt.imshow(mask.detach().cpu())
        #     plt.show()
            

        print(f'redicted masks shape {masks.shape}')
        
        print(f'labels_my {labels_my} labels {labels}')
        print(f'scores_my {scores_my} scores {scores}  ')
        # print(f'classes: my {clss_sorted} their {labels}')
        # print(labels_test()) #selected, truth
        # # print(bxs_h)
        # bbxs_keep,probas_keep,classes_keep,keep=predictionThresholdMaskRcnn(pbbxs,labels,scores,0)
        # #print(scores)
        # print(f'After keep pbbxs labels masks scores : {bbxs_keep.shape} {classes_keep.shape} {masks.shape} {probas_keep.shape} ')
        # print(f'img shape last {img.shape}')
        drawfn(img.copy(),pbbxs[keep,:],scores[keep])
        # scores_my=torch.tensor([0.99,0.76,0.99])
        # class_name=["cow","dear","bird"]
        drawfn(img.copy(),pbbxs_my,scores_my)
        #holistic exp
        hol.second_stage(list(pbbxs_my),list(scores_my),list(labels_my))
        print(f'hol.selected_classes {hol.selected_classes}')

        # #from my det
        # vis = Visualizer()

        #         #from detection
        # vis.set_image(img)
        # # vis.draw_bboxes(boxes[keep1,:], edge_colors='r')
        # vis.draw_bboxes(pbbxs[keep,:], edge_colors='g')
        # vis.get_image()
        # vis.show()

        # vis.set_image(img)
        # vis.draw_bboxes(pbbxs_my, edge_colors='r')
        # #vis.draw_bboxes(pbbxs[keep,:], edge_colors='g')
        # vis.get_image()
        # vis.show()




        # pbbxs labels masks scores : torch.Size([100, 4]) torch.Size([100]) torch.Size([100, 480, 640]) torch.Size([100]) 
        # the shape of img (640, 480, 3)
        