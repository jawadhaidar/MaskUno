import torch
from model import*
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms 
from data_class import*
from helper_functions import*
import torchinfo
import mmdet 
import mmdet.apis
from mmengine.visualization import*
import time



def collate_fn(batch):
    return tuple(zip(*batch))
#device
device="cuda"
#load model 
catmodel=UnoModelMaskRcnnSameMmdet().to(device=device)
#loss function
loss_fun=torch.nn.BCELoss()
for param in catmodel.parameters(): #todo: check always ,depends on model
    param.requires_grad = True
params = [p for p in catmodel.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001) #was 0.001 for all trained epochs
# optimizer=torch.optim.SGD(params,lr=0.001)#was 0.001 for all trained epochs


print(catmodel)
torchinfo.summary(catmodel)





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
    # confg_file="/home/jawad/mmdetection/myconfigs/detectors.py"
    # checkpoint_file="/home/jawad/mmdetection/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth"
    confg_file="/home/jawad/codes/MaskUno/detectors_v2.py"
    checkpoint_file="/home/jawad/Downloads/detectors_htc_r50_1x_coco-329b1453.pth"
    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    for name, param in detectionmodel.named_parameters():
        print(name)
        param.requires_grad = False
    activation_value='roi_head.mask_head.2.convs.3.conv'
    detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
    #detectionmodel.eval()
    modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_detectorsv2features_opt3_same_thr0.1_fixedcounter_editgt_30epochs.pth').to(device=device)

elif detectionpipeline=="mask" :
    mytransform=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
    ])
    # confg_file="/home/jawad/codes/MaskUno/mask_config.py"
    # checkpoint_file="/home/jawad/mmdetection/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"#"/home/jawad/codes/epoch_11.pth"#
    confg_file="/home/jawad/codes/MaskUno/mask_config_uno.py" #change num classes into one
    checkpoint_file="/home/jawad/codes/epoch_11.pth"  
    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    # detectionmodel=torch.load("/home/jawad/codes/MaskUno/models/model.pth").to(device=device)

    activation_value='roi_head.mask_head.upsample'

    detectionmodel.roi_head.mask_head.upsample.register_forward_hook(get_activation('roi_head.mask_head.upsample'))

    # modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_maskrcnnmmdetfeatures_opt3_same_thr0_fixedcounter_editgt_bgr_freeze_epochs5.pth').to(device=device)

elif detectionpipeline=="htc" :
    print("htc")
    mytransform=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float),
    ])
    confg_file="/home/jawad/mmdetection/myconfigs/htc.py"
    checkpoint_file="/home/jawad/Downloads/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"
    detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
    activation_value='roi_head.mask_head.2.convs.3.conv'

    for name, param in detectionmodel.named_parameters():
        print(name)
    detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
    detectionmodel.eval()
    modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_htcmmdetfeatures_opt3_same_thr0_fixedcounter_editgt_bgr_epochs5.pth').to(device=device)




# DETRmodel = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device=device)
# DETRmodel.eval()
# #register hook
# DETRmodel.transformer.decoder.norm.register_forward_hook(get_activation('transformer.decoder.norm'))


dataset=COCODatasetUNOApi(transforms=mytransform,stage='train',class_name='cat')
data_loader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate_fn)
epochs=30
#train function
torchinfo.summary(catmodel)
torchinfo.summary(detectionmodel)
# img=torch.rand(1,500,600,3).to(device="cuda")
# bbxs,labels,masks,scores,features=detection(detectionpipeline,model,img,activation)

trainPerClassMaskrcnn(epochs,data_loader,catmodel,detectionmodel,activation,dataset.catIds[0],loss_fun,optimizer,detectionpipeline,device)