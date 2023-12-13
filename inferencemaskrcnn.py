import torch
from model import*
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms 
from data_class import*
from helper_functions import*
import cv2 as cv
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/home/jawad/codes/references/detection')
print(sys.path)
from engine import train_one_epoch, evaluate 

def collate_fn(batch):
    return tuple(zip(*batch))

detectionpipeline=input("normal for normal | mask for mmdet maskrcnn | detectors for detectors: ")
#device
device="cuda"
#load model 

#dataloader 

#load DETR model 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = input#output.detach()#output.detach()#input#output.detach()
    return hook

if detectionpipeline=="normal":
    mytransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    ])
    detectionmodel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device=device)
    detectionmodel.eval()
    #register hook
    for name, param in detectionmodel.named_parameters():
        print(name)

    activation_value='roi_heads.mask_head.3.0' #TODO: object fill automatic
    detectionmodel.roi_heads.mask_head[3][0].register_forward_hook(get_activation('roi_heads.mask_head.3.0'))
    modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_maskrcnnfeatures_opt3_same_thr0_fixedcounter_editgt_30epochs.pth').to(device=device)
    print(detectionmodel)
    torchinfo.summary(detectionmodel)

elif detectionpipeline=="detectors":
    mytransform=transforms.Compose([
    # transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float),
    ])
    confg_file="/home/jawad/mmdetection/myconfigs/detectors.py"
    checkpoint_file="/home/jawad/mmdetection/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth"
    # confg_file="/home/jawad/codes/MaskUno/detectors_v2.py"
    # checkpoint_file="/home/jawad/Downloads/detectors_htc_r50_1x_coco-329b1453.pth"
    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    for name, param in detectionmodel.named_parameters():
        print(name)
        param.requires_grad = False
    activation_value='roi_head.mask_head.2.convs.3.conv'
    detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
    #detectionmodel.eval()
    modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_detectorsv2features_opt3_same_thr0.1_fixedcounter_editgt_30epochs.pth').to(device=device)
    det=[]
    det_score=[]
    mask_head=[]

elif detectionpipeline=="mask" :
    mytransform=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
    ])
    confg_file="/home/jawad/codes/MaskUno/mask_config_all.py"
    checkpoint_file="/home/jawad/mmdetection/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"
    # confg_file="/home/jawad/codes/MaskUno/mask_config_uno.py" #change num classes into one
    # checkpoint_file="/home/jawad/codes/epoch_11.pth"    
    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    # detectionmodel=torch.load("/home/jawad/codes/MaskUno/models/model.pth").to(device=device)
    
    for name, param in detectionmodel.named_parameters():
        print(name)
    #for mask
    activation_value='roi_head.mask_head.upsample'
    detectionmodel.roi_head.mask_head.upsample.register_forward_hook(get_activation('roi_head.mask_head.upsample'))
    #for detection 
    activation_value_detection='roi_head.bbox_head.shared_fcs[1]'
    detectionmodel.roi_head.bbox_head.shared_fcs[1].register_forward_hook(get_activation('roi_head.bbox_head.shared_fcs[1]'))
    detectionmodel.roi_head.bbox_head.fc_reg.register_forward_hook(get_activation('roi_head.bbox_head.fc_reg'))
    detectionmodel.backbone.register_forward_hook(get_activation('backbone.layer1.0'))

    det=detectionmodel.roi_head.bbox_head.fc_reg #just for exp
    det_score=detectionmodel.roi_head.bbox_head.fc_cls
    #get mask head
    mask_head=detectionmodel.roi_head.mask_head.convs

elif detectionpipeline=="htc" :
    print("htc")
    mytransform=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
    ])
    confg_file="/home/jawad/mmdetection/myconfigs/htc.py"
    checkpoint_file="/home/jawad/Downloads/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"
    #uno
    # confg_file="/home/jawad/mmdetection/myconfigs/htc.py"
    # checkpoint_file="/home/jawad/codes/epoch_22.pth"

    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    activation_value='roi_head.mask_head.2.convs.3.conv'

    for name, param in detectionmodel.named_parameters():
        print(name)
    detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
    detectionmodel.eval()
    modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_htcmmdetfeatures_opt3_same_thr0_fixedcounter_editgt_bgr_epochs5.pth').to(device=device)
    det=[]
    det_score=[]
    mask_head=[]


dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name='all') #all refers for annotations now 
data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
#load data 
if detectionpipeline=="normal":
    classID=dataset.catIds[0]
else:
    classID=15 #for cats in mmeddetetion


projection_size=28

draw=True
flag=4

confg_file="/home/jawad/codes/MaskUno/mask_config_uno.py" #change num classes into one
checkpoint_file="/home/jawad/codes/epoch_11_unocat.pth"
model_uno=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")

# print(model_uno)
# torchinfo.summary(model_uno)
#for mask
modelClass=nn.Sequential(model_uno.roi_head.mask_head.upsample,
                         model_uno.roi_head.mask_head.relu,
                         model_uno.roi_head.mask_head.conv_logits                               
                    )


detuno=model_uno.roi_head.bbox_head.fc_reg #just for exp
detuno_score=model_uno.roi_head.bbox_head.fc_cls
# 

#trained model class
# evaluate(modelClass, data_loader, device,flag,detectionpipeline,detectionmodel,activation,activation_value,classID,detuno,detuno_score,det,det_score,mask_head)
idsall=dataset.coco_obj.getImgIds(catIds=[])
idscat=dataset.coco_obj.getImgIds(catIds=[17])
print(idsall)
for i, batched_sample in enumerate(data_loader):
    print(f'i {i}')
    images,targets=batched_sample
    if detectionpipeline=="normal":
        images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
    targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] # each target in the batch is nowin a list 
    #project gtmasks to the size of loss input (16x16)
    # inference(targets,images,detectionmodel,activation,classID,modelClass,activation_value,projection_size,detectionpipeline,device,draw)
    #    out=quantitative(images,detectionmodel,activation,classID,modelClass,activation_value,detectionpipeline,device)
    out=quantitative_new(images,detectionmodel,activation,classID,detectionpipeline,detuno,detuno_score,det,det_score,mask_head,modelClass,device,i,idsall,idscat)
#     print(f'out : {len(out)}')
      
            
