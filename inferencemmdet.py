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
#device
device="cuda"
#load model 
modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_maskrcnnfeatures_opt3_same_thr0_fixedcounter_editgt_30epochs.pth').to(device=device)
#loss function
loss_fun=torch.nn.BCELoss()
params = [p for p in modelClass.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params, lr=0.01)


#dataloader 
mytransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name='cat')
data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
#load DETR model 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

detectionmodel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device=device)
detectionmodel.eval()
#register hook
activation_value='roi_heads.mask_head.mask_fcn4' #TODO: object fill automatic
detectionmodel.roi_heads.mask_head.mask_fcn4.register_forward_hook(get_activation(activation_value))

flag1=2 #SET THE FLAG HERE
# if flag1==1:
#     epochs=1
#     #train(epochs,data_loader,model1,optimizer,device=DEVICE)
#     for epoch in range(epochs):
#         # train for one epoch, printing every 10 iterations
#         #train_one_epoch(model1, optimizer, data_loader, DEVICE, epoch, print_freq=10)
#         print(dataset.catIds)
#         evaluate(modelClass, data_loader, device=device,flag=2,classid=dataset.catIds[0]) #TODO: [0] since it return []
    
# epochs=30

#load data 
classID=dataset.catIds[0]
projection_size=28

dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name='cat')
data_loader=DataLoader(dataset,batch_size=6,shuffle=False,collate_fn=collate_fn)
draw=True
flag=4
# evaluate(modelClass, data_loader, device,flag,classID)
for i, batched_sample in enumerate(data_loader):
    images,targets=batched_sample


    images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
    targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] # each target in the batch is nowin a list 
    #project gtmasks to the size of loss input (16x16)
    inference(targets,images,detectionmodel,activation,classID,modelClass,activation_value,projection_size,device,draw)
    # out=quantitative(images,detectionmodel,activation,classID,modelClass,activation_value,device)
    # print(f'out : {len(out)}')
      
            
