import mmdet 
import mmdet.apis
import torchinfo
import torch 
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
from mmengine.visualization import*


device="cuda"

def collate_fn(batch):
    return tuple(zip(*batch))

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = input #.detach() #output.detach()
    return hook
    


model_detection=torch.load("/home/jawad/codes/MaskUno/models/model.pth")


confg_file="/home/jawad/codes/MaskUno/mask_config.py"
checkpoint_file="/home/jawad/codes/epoch_11.pth"
model_uno=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")

#register activation
activation_value='roi_head.mask_head.upsample'
model_uno.roi_head.mask_head.upsample.register_forward_hook(get_activation('roi_head.mask_head.upsample'))

detectionpipeline="mask"
#do detection 
img=torch.rand(1,500,600,3).to(device="cuda")
detectionmodel=model_uno
bbxs,labels,masks,scores,features=detection(detectionpipeline,detectionmodel,img,activation)


model=nn.Sequential(model_uno.roi_head.mask_head.upsample,
                    model_uno.roi_head.mask_head.conv_logits,
                    model_uno.roi_head.mask_head.relu)

c1=model_uno.roi_head.mask_head.upsample
c2=model_uno.roi_head.mask_head.conv_logits
rl=model_uno.roi_head.mask_head.relu


#they should be the same (but first we should add sigmoid)
print(model(features).shape)
print(masks.shape)