import torch
from model import*
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms 
from data_class import*
from helper_functions import*
import cv2 as cv
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))
#device
device="cuda"
#load model 
catmodel=torch.load('/home/jawad/codes/MaskUno/models/model_complete_cat_onlypredictionh_softstatic_Sigmoid_gt_thiertrans_withoutpadd_putepochs.pth').to(device=device)
#loss function
loss_fun=torch.nn.BCELoss()
params = [p for p in catmodel.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params, lr=0.01)


#dataloader 
mytransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


dataset=COCODatasetUNOApi(transforms=mytransform,stage='train',class_name='cat')
data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
#load DETR model 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

DETRmodel = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device=device)
DETRmodel.eval()
#register hook
DETRmodel.transformer.decoder.norm.register_forward_hook(get_activation('transformer.decoder.norm'))


epochs=30

#load data 
classID=dataset.catIds[0]

sigmoid=nn.Sigmoid()
for i, batched_sample in enumerate(data_loader):
    images,targets=batched_sample
    images_before=list(image.to(device=device)  for image in images)
    #TODO: augment the box here
    #targets=aug_as_u_go(targets,images)
    #print("I am augmenting")
    #TODO: implement in a better way | preprocess , here the
    trans=GeneralizedRCNNTransformMy(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
    images,targets=trans(images,targets)
    images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
    targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] 
    #project to mask size 
    target_out=project_masks_on_boxes_binary(targets,images,flag="roia_same")


    for id,img in enumerate(images): 
        gtMasksProjected=target_out[id]
        gtbbxs=targets[id]['boxes']
        gtclasses=targets[id]['labels']
        #detection 
        detPrediction=DETRmodel(img.unsqueeze(0)) #might need some format changes 
        #TODO: here we should choose pred >0.9
        pbbxs,pprobas,pclasses,keep=predictionThreshold(detPrediction) #all classes
        #get features
        features=activation['transformer.decoder.norm'] 
        print(f'features before keep {features.shape}')
        features=features[keep,:,:]
        print(f'features after keep: {features.shape} ')
        if features.shape[0]==0:
            continue #this will not work if it was the last image in batch
        #select features and gtdata s
        #in this case gtclasses only belong to one class
        #TODO: map id of classes to 1
        pclasses=torch.ones_like(pclasses)*-1
        pclasses[pclasses!=classID]=1
        #rescale predictions
        pbbxs=rescale_bboxes(pbbxs, size=(img.shape[-1],img.shape[-2]),device=device) 

        #TODO: chose only features for the same class
        MaxIousRowIndex,MaxIousColIndex =matcherIndex(pbbxs,pclasses,gtbbxs,gtclasses)#match at the augmente level
        featuresSelected,gtMasksSelected=match(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex)
        featuresSelected=torch.reshape(featuresSelected,(featuresSelected.shape[0],int(np.sqrt(featuresSelected.shape[1])),int(np.sqrt(featuresSelected.shape[1]))))
        print(f'featuresSelected after reshape {featuresSelected.shape}')

        out_one=catmodel(featuresSelected.unsqueeze(1).to(device=device)).to(device=device) #unsqueeze to add channel dim/ 
        print(f'out shape {out_one.shape}')
        out_one=sigmoid(out_one[:,1,:,:])
        target_out_sample=gtMasksSelected.to(device=device) 

        for id2,tar in enumerate(target_out_sample):
            plt.subplot(target_out_sample.shape[0], 2, id2*2+1)
            print(f'the shape of tar.shape {tar.shape}')
            plt.imshow(tar.detach().cpu().numpy())
            plt.subplot(target_out_sample.shape[0], 2, id2*2+1+1)
            plt.imshow(out_one[id2].detach().cpu().numpy())
        plt.show()


      
            
