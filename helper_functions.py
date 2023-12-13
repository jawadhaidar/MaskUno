import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from torchsummary import summary
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2 as cv 

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
from data_class import*
from torch.utils.data import Dataset, DataLoader
#from testModel import*
from torchvision.models.detection.roi_heads import* #to get mask project
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform ,GeneralizedRCNNTransformMy
import cv2 as cv
import mmdet 
import mmdet.apis
import time
from helper_detection import*
from mmengine.visualization import*

'''As in Fast R-CNN, an RoI is considered positive
if it has IoU with a ground-truth box of at least 0.5 and
negative otherwise
The mask loss Lmask is defined only on
positive RoIs. 
The mask target is the intersection between
an RoI and its associated ground-truth mask'''




def get_iou_torch(ground_truth, pred):
    # Coordinates of the area of intersection.
    ix1 = torch.max(ground_truth[0], pred[0])
    iy1 = torch.max(ground_truth[1], pred[1])
    ix2 = torch.min(ground_truth[2], pred[2])
    iy2 = torch.min(ground_truth[3], pred[3])
    
    # Intersection height and width.
    i_height = torch.max(iy2 - iy1 + 1, torch.tensor(0.))
    i_width = torch.max(ix2 - ix1 + 1, torch.tensor(0.))
    
    area_of_intersection = i_height * i_width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou

def IOUMatrix(pbbxs,gtbbxs,pclasses,gtclasses):
    '''calculates IOU between two nested bbxs arrays'''
    IouMatrix=torch.rand((pbbxs.shape[0],gtbbxs.shape[0]))
    SameClassMatrix=torch.zeros((pbbxs.shape[0],gtbbxs.shape[0]))
    ThresholdMatrix=torch.zeros((pbbxs.shape[0],gtbbxs.shape[0]))
    #print(IouMatrix.shape)
    #you can add a matrix for o.5 thr

    for row,bbx in enumerate(pbbxs):
        for col,gtbbx in enumerate(gtbbxs):
            IouMatrix[row][col]=get_iou_torch(bbx, gtbbx)
            if pclasses[row] == gtclasses[col]:
                SameClassMatrix[row][col]=1
            if IouMatrix[row][col]>0.5: #TODO: try varrying this threshold
                ThresholdMatrix[row][col]=1 

    return IouMatrix,SameClassMatrix,ThresholdMatrix

 
def matcherIndex(pbbxs,pclasses,gtbbxs,gtclasses):
    '''
    returns:
    MaxIousRowIndex: the ids of the feature (1 -->100/N)
    MaxIousColIndex: teh id of corresponding gtbbx/gtmask

    '''
    IouMatrix,SameClassMatrix,ThresholdMatrix=IOUMatrix(pbbxs,gtbbxs,pclasses,gtclasses)
    #print(f' IouMatrix  {IouMatrix}')
    IouMatrixClass=IouMatrix*SameClassMatrix*ThresholdMatrix
    #print(f'IouMatrixClass {IouMatrixClass}')
    #get max per col
    MaxIousPerCol,MaxIousRowIndex=torch.max(IouMatrixClass,dim=0)
    MaxIousColIndex=torch.arange(0, MaxIousRowIndex.shape[0], 1) #double check .shape[0 or 1]
    #if the max is zero remove from list
    
    MaxIousRowIndex=MaxIousRowIndex[MaxIousPerCol!=0]
    MaxIousColIndex=MaxIousColIndex[MaxIousPerCol!=0]

    return MaxIousRowIndex,MaxIousColIndex #index of features, index of gt

def match(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex): #TODO: change to per batch
    num_samples=MaxIousRowIndex.shape[0] # check 0 or 1
    #print(f'number of chosen {num_samples} ')
    #features shape torch.Size([100, 1, 256])
    #gtMasksProjected shape [N,14,14]
    features=features.squeeze(1)
    #print(f'featers shape after squeeze {features.shape}')
    featuresSelected=torch.zeros(size=(num_samples,features.shape[1]))
    gtMasksSelected=torch.zeros(size=(num_samples,gtMasksProjected.shape[1],gtMasksProjected.shape[2]))
    counter=0
    for row,col in zip(MaxIousRowIndex,MaxIousColIndex):
        featuresSelected[counter,:]=features[row,:]
        gtMasksSelected[counter,:,:]=gtMasksProjected[col,:,:]
        counter+=1 #TODO: previously you forgot the counter
    #print(f'featers  gt selected shapes {featuresSelected.shape,gtMasksSelected.shape}')
    return featuresSelected,gtMasksSelected

def matchMaskrcnn(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex): #TODO: change to per batch
    num_samples=MaxIousRowIndex.shape[0] # check 0 or 1
    print(f'num_samples {num_samples}')
    print(MaxIousRowIndex.shape)
    #print(f'number of chosen {num_samples} ')
    #features shape  #torch.Size([M, 256, 14, 14])
    #gtMasksProjected shape [N,14,14]
    
    #print(f'featers shape after squeeze {features.shape}')
    featuresSelected=torch.zeros(size=(num_samples,features.shape[1],features.shape[2],features.shape[3]))
    gtMasksSelected=torch.zeros(size=(num_samples,gtMasksProjected.shape[1],gtMasksProjected.shape[2]))
    counter=0
    print(f'gtMasksSelected shape {gtMasksSelected.shape}')
    for row,col in zip(MaxIousRowIndex,MaxIousColIndex):
        print(f'row,col {row,col}')
        print(f'coun {counter}')
        print(f'count {counter}')
        featuresSelected[counter,:,:,:]=features[row,:,:,:]
        gtMasksSelected[counter,:,:]=gtMasksProjected[col,:,:]
        counter+=1
    #print(f'featers  gt selected shapes {featuresSelected.shape,gtMasksSelected.shape}')
    return featuresSelected,gtMasksSelected
def matchMaskrcnnfeatures(features,MaxIousRowIndex,MaxIousColIndex): #TODO: change to per batch
    num_samples=MaxIousRowIndex.shape[0] # check 0 or 1

    
    #print(f'featers shape after squeeze {features.shape}')
    featuresSelected=torch.zeros(size=(num_samples,features.shape[1],features.shape[2],features.shape[3]))
    counter=0
    for row,col in zip(MaxIousRowIndex,MaxIousColIndex):

        featuresSelected[counter,:,:,:]=features[row,:,:,:]
        counter+=1
    #print(f'featers  gt selected shapes {featuresSelected.shape,gtMasksSelected.shape}')
    return featuresSelected




##compare predicted mask with this
def predictionThreshold(outputs):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    max_values, max_indices =probas.max(-1)
    keep=max_values>0.8
    # print(outputs['pred_logits'].shape) #torch.Size([1, 100, 92])
    # print(probas.shape) #torch.Size([100, 91])
    # print(probas.max(-1))
    # print(outputs['pred_boxes'].shape) #torch.Size([1, 100, 4])
  
    
    bbxs_keep=outputs['boxes'][0, keep]
    probas_keep=max_values[keep]
    classes_keep=max_indices[keep]
    return bbxs_keep,probas_keep,classes_keep,keep

# def predictionThresholdMaskRcnn(outputs):
#     scores = outputs['scores']
#     keep=scores>0.9
#     bbxs=outputs['boxes']
#     #print(f'shape of pred bxx  {bbxs.shape}')
#     bbxs_keep=outputs['boxes'][keep,:]
#     probas_keep=scores[keep]
#     classes_keep=outputs['labels'][keep]
#     return bbxs_keep,probas_keep,classes_keep,keep
def predictionThresholdMaskRcnn(bbxs,labels,scores,thr):
    
    keep=scores>thr
    bbxs_keep=bbxs[keep,:]
    probas_keep=scores[keep]
    classes_keep=labels[keep]

    return bbxs_keep,probas_keep,classes_keep,keep

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x,device):
    x_c, y_c, w, h = x.unbind(1) #x[0],x[1],x[2],x[3]  #x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1).to(device=device)

def rescale_bboxes(out_bbox, size,device):
    img_w, img_h = size
    #print(f"the size {size}")
    b = box_cxcywh_to_xyxy(out_bbox,device)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device=device)
    return b

def visualize(img,bbxs):
    arrImg = img.squeeze(0).numpy().transpose(1, 2, 0)
    arrImg = cv.cvtColor(arrImg,cv.COLOR_BGR2RGB )#cv.COLOR_RGB2BGR
    #bbxs=rescale_bboxes(bbxs, size) #resclae outside
  

    for bbx in bbxs:
        pt1=(round(bbx[0].item()),round(bbx[1].item()))
        pt2=(round(bbx[2].item()),round(bbx[3].item()))
        #pt2=(round(bbx[2].item()),round(bbx[3].item()))
        print(pt1,pt2)
        cv.rectangle(arrImg, pt1, pt2, (255,0,0), 2) 
        #print(bbx)

    cv.imshow("img", arrImg) 
    cv.waitKey(5000) 
    cv.destroyAllWindows() 
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]    
def plot_results(pil_img,boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for (xmin, ymin, xmax, ymax), c in zip(boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
      
    plt.axis('off')
    plt.show()

#train function
def train(epochs,train_loader,model,DETRmodel,activation,loss_fun,optimizer,device):
    running_loss_list=[]
    for epoch in tqdm(range(epochs)):
        running_loss=0
        sigmoid=nn.Sigmoid()
        for i, batched_sample in enumerate(train_loader):
            images,targets=batched_sample
            images_before=list(image.to(device=device)  for image in images)
            #TODO: augment the box here
            #targets=aug_as_u_go(targets,images)
            print("I am augmenting")
            #TODO: implement in a better way | preprocess , here the
            trans=GeneralizedRCNNTransformMy(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
            images,targets=trans(images,targets)
            images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
            targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] # each target in the batch is nowin a list 
            #forward pass
            #get the prediction
            boxes=[dict['boxes'] for dict in targets]
            #TODO: add a function that augment the boxes
            #loop for each image box pair
            #preprocess the masks (resize them)
            target_out=project_masks_on_boxes_binary(targets,images,flag="roia_same") #TODO: here only the len of images is needed , fix for readability
            classes=[] # strings

            out=[]
            losses=0
            for id,img in enumerate(images_before): #TODO: this is not batch training try to fix it
                #apply DETR 
                DETRmodel(img.unsqueeze(0))
                #get features
                features=activation['transformer.decoder.norm']
                out_one=model(classes,features).to(device=device)

                #get the target_out corresponding to this img
                target_out_sample=target_out[id].to(device=device)
                #get the predicted out corresponding to this img

                out_one=sigmoid(out_one[:,1,:,:])
                #we need to match out with target
                #losses=losses+loss_fun(out_one,target_out_sample.type(torch.LongTensor).to(device=device)) #TOD:: type was causing the gpu issue
                losses=losses+loss_fun(out_one,target_out_sample) #TOD:: type was causing the gpu issue

            losses.backward()
            #update weights 
            optimizer.step()
            #zero gradients
            optimizer.zero_grad()
            running_loss += losses.item()
            print(f'batch {i} out of {len(train_loader)} with batch loss {losses.item():.20f} ')
        running_loss_list.append(running_loss)
        print(f'epoch {epoch} || loss : {running_loss_list}')
        torch.save(model, r'/home/jawad/codes/OneClassMethod/models/model_complete_cat_onlypredictionh_softstatic_Sigmoid_gt_thiertrans_withoutpadd_putepochs.pth')

def trainPerClass(epochs,train_loader,modelClass,DETRmodel,activation,classID,loss_fun,optimizer,device):
    running_loss_list=[]
    for epoch in tqdm(range(epochs)):
        running_loss=0
        sigmoid=nn.Sigmoid()
        for i, batched_sample in enumerate(train_loader):
            images,targets=batched_sample
            #TODO: augment the box here
            #targets=aug_as_u_go(targets,images)
            #print("I am augmenting")
            #TODO: implement in a better way | preprocess , here the
            trans=GeneralizedRCNNTransformMy(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
            images,targets=trans(images,targets)
            images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
            targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] # each target in the batch is nowin a list 
            #project gtmasks to the size of loss input (16x16)
            #TODO: there is a mistake here| it should be projected accourding to the pbox
            target_out=project_masks_on_boxes_binary(targets,images,projection_size=16,flag="roia_same") #TODO: here only the len of images is needed , fix for readability

          
            losses=0
            for id,img in enumerate(images): #TODO: this is not batch training try to fix it
                gtMasksProjected=target_out[id]
                gtbbxs=targets[id]['boxes']
                gtclasses=targets[id]['labels'] #all 1s in this case
                #apply DETR 
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
                print(f'pclasses before {pclasses}')
                temp=torch.ones_like(pclasses)*-1
                temp[pclasses==classID]=1 #TODO: there was a mistake here != should be ==| mean we were taking all predictions as gtcat
                pclasses=temp
                print(f'pclasses after {pclasses}')
                #rescale predictions
                pbbxs=rescale_bboxes(pbbxs, size=(img.shape[-1],img.shape[-2]),device=device) 
                #visualize(img.detach().cpu(),pbbxs)
                #DEBUG: show the pbbxs
                MaxIousRowIndex,MaxIousColIndex =matcherIndex(pbbxs,pclasses,gtbbxs,gtclasses)#match at the augmente level
                featuresSelected,gtMasksSelected=match(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex)
                print(f'featuresSelected before reshape {featuresSelected.shape}')
                
                featuresSelected=torch.reshape(featuresSelected,(featuresSelected.shape[0],int(np.sqrt(featuresSelected.shape[1])),int(np.sqrt(featuresSelected.shape[1]))))
                print(f'featuresSelected after reshape {featuresSelected.shape}')

                out_one=modelClass(featuresSelected.unsqueeze(1).to(device=device)).to(device=device) #unsqueeze to add channel dim/ 

                #get the target_out corresponding to this img
                target_out_sample=gtMasksSelected.to(device=device) 
                #TODO: reshape features selected with the corr tar 

                    
                    #project
                #get the predicted out corresponding to this img

                out_one=sigmoid(out_one[:,1,:,:])
                #0    | 1    | 2   |3 
                #0 1  |  2 3|  4 5 | 6 7
                #1 2 | 3 4| 5 6
                # for id2,tar in enumerate(target_out_sample):
                #     plt.subplot(target_out_sample.shape[0], 2, id2*2+1)
                #     print(f'the shape of tar.shape {tar.shape}')
                #     plt.imshow(tar.detach().cpu().numpy())
                #     plt.subplot(target_out_sample.shape[0], 2, id2*2+1+1)
                #     plt.imshow(out_one[id2].detach().cpu().numpy())
                # plt.show()
                #we need to match out with target
                #losses=losses+loss_fun(out_one,target_out_sample.type(torch.LongTensor).to(device=device)) #TOD:: type was causing the gpu issue
                losses=losses+loss_fun(out_one,target_out_sample) #TOD:: type was causing the gpu issue
                if torch.isnan(losses):
                    print(f'loss is nan')
            if features.shape[0]==0 or torch.isnan(losses):
                    continue 
            losses.backward()
            #update weights 
            optimizer.step()
            #zero gradients
            optimizer.zero_grad()
            running_loss += losses.item()
            print(f'batch {i} out of {len(train_loader)} with batch loss {losses.item():.20f} ')
        running_loss_list.append(running_loss)
        print(f'epoch {epoch} || loss : {running_loss_list}')
        torch.save(modelClass, r'/home/jawad/codes/MaskUno/models/model_complete_cat_onlypredictionh_softstatic_Sigmoid_gt_thiertrans_withoutpadd_putepochs.pth')

def trainPerClassMaskrcnn(epochs,train_loader,modelClass,Detectionmodel,activation,classID,loss_fun,optimizer,detectionpipeline,device):
    running_loss_list=[]
    softmax=nn.Softmax(dim=1)
    for epoch in tqdm(range(epochs)):
        running_loss=0
        sigmoid=nn.Sigmoid()
        for i, batched_sample in enumerate(train_loader):
            images,targets=batched_sample
            #print(f'images[0] shape {images[0].shape}')

            if detectionpipeline=="normal": #onedifference
                images = list(image.to(device=device)  for image in images) #check later # each sample in the batch is no in a list 
            targets = [{k: v.to(device=device)  for k, v in t.items()} for t in targets] # each target in the batch is nowin a list 
            #project gtmasks to the size of loss input (16x16)
            #TODO: there is a mistake here| it should be projected accourding to the pbox
            # target_out=project_masks_on_boxes_binary(targets,images,projection_size=28,flag="roia_same") #TODO: here only the len of images is needed , fix for readability

          
            losses=torch.tensor([0]).to(device=device)
            for id,img in enumerate(images): #TODO: this is not batch training try to fix it
                #print(f'img shape {img.shape}')
                #print(id)
                # gtMasksProjected=target_out[id]
                gtbbxs=targets[id]['boxes']
                gtclasses=targets[id]['labels'] #all 1s in this case
                #apply DETR 
                print("here")
                bbxs,labels,masks,scores,features=detection(detectionpipeline,Detectionmodel,img,activation)
                #print(f'scores.shape {scores.shape}')
                if scores.shape[0]==0:
                    continue
                #detection threshld #we are keeping them all for now
                pbbxs,pprobas,pclasses,keep=predictionThresholdMaskRcnn(bbxs,labels,scores,0) #all classes
                #get features
                
                # print(f'features before keep {features.shape}')
                # print(f'keep {keep}')
                features=features[keep,:,:]
                # print(f'features after keep: {features.shape} ')
                if features.shape[0]==0:
                    continue #this will not work if it was the last image in batch
                #select features and gtdata s
                #in this case gtclasses only belong to one class
                #TODO: map id of classes to 1
                #print(f'pclasses before {pclasses}')
                # temp=torch.ones_like(pclasses)*-1
                # if detectionpipeline=="normal":
                #     temp[pclasses==classID]=1 #TODO: there was a mistake here != should be ==| mean we were taking all predictions as gtcat
                # else: #TODO: fix this later
                #     temp[pclasses==15]=1 #15 for cat inn mmdetection 
                # pclasses=temp
                pclasses=torch.ones_like(pclasses)
                #print(f'pclasses after {pclasses}')
 
                #DEBUG: show the pbbxs
                MaxIousRowIndex,MaxIousColIndex =matcherIndex(pbbxs,pclasses,gtbbxs,gtclasses)#match at the augmente level
                #TODO: here add a func that projects masks acc to the bbx predictionsselected
                #print(MaxIousRowIndex,MaxIousColIndex)
                if MaxIousRowIndex.shape[0]==0:
                    continue
                gtMasksProjected=project_masks_on_predicted_boxes_binary(targets[id],pbbxs,img,28,MaxIousRowIndex,MaxIousColIndex )
                #print(f'the shape of gtMasksProjected with prediction : {gtMasksProjected.shape}')
                # print(f' the shape of gtMasksProjected {gtMasksProjected.shape} ')
                # featuresSelected,gtMasksSelected=matchMaskrcnn(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex)
                featuresSelected=matchMaskrcnnfeatures(features,MaxIousRowIndex,MaxIousColIndex)
                gtMasksSelected=gtMasksProjected

                #visualize features 
                #check the range
                #print(f'max {}')
                
                
                print(f'featuresSelected.shape {featuresSelected.shape}')
                out_one=modelClass(featuresSelected.to(device=device)).to(device=device) #unsqueeze to add channel dim/ 
                print(f'out_one shape is {out_one.shape}')
                #get the target_out corresponding to this img
                target_out_sample=gtMasksSelected.detach().to(device=device) 
                #TODO: reshape features selected with the corr tar 
                print(f'target_out_sample shape is {target_out_sample.shape}')
                    
                    #project
                #get the predicted out corresponding to this img

                # out_one=sigmoid(out_one[:,1,:,:])
                #out_one=softmax(out_one)[:,1,:,:]
                out_one=sigmoid(out_one).squeeze(1) #ToDO: fix later to make it general
                # print(out_one.shape)
                #0    | 1    | 2   |3 
                #0 1  |  2 3|  4 5 | 6 7
                #1 2 | 3 4| 5 6
                # plt.imshow(img)
                # plt.show()
                # for id2,tar in enumerate(target_out_sample):
                #     plt.subplot(target_out_sample.shape[0], 2, id2*2+1)
                #     print(f'the shape of tar.shape {tar.shape}')
                #     plt.imshow(tar.detach().cpu().numpy())
                #     plt.subplot(target_out_sample.shape[0], 2, id2*2+1+1)
                #     plt.imshow(out_one[id2].detach().cpu().numpy())
                # plt.show()
                #we need to match out with target
                #try softmax for detectors
                #losses=losses+loss_fun(out_one,target_out_sample.type(torch.LongTensor).to(device=device)) #TOD:: type was causing the gpu issue
                losses=losses+loss_fun(out_one,target_out_sample) #TOD:: type was causing the gpu issue
                if torch.isnan(losses):
                    print(f'loss is nan')
            if features.shape[0]==0 or torch.isnan(losses) or MaxIousRowIndex.shape[0]==0 or scores.shape[0]==0:
                    continue 
            losses.backward()
            #update weights 
            optimizer.step()
            #zero gradients
            optimizer.zero_grad()
            running_loss += losses.item()
            print(f'batch {i} out of {len(train_loader)} with batch loss {losses.item():.20f} in epoch {epoch} running loss {running_loss_list}')
        running_loss_list.append(running_loss)
        print(f'epoch {epoch} || loss : {running_loss_list}')
        torch.save(modelClass, r'/home/jawad/codes/MaskUno/models/model_cat_maskrcnnmmdetfeatures_fromunoasdetection.pth')
        time.sleep(10)
def add_gt_masks():
    '''used to add the gt bbxs and their corresponding features to te training'''
    #loss(predicted,gtprojected)
    #gtprojected used the func
    #predicted: 
    #use roia to extract features from the backebone
    #return features,gtprojected
    
def project_masks_on_boxes_binary(targets,images,projection_size,flag): #used as gt to train
    '''
    used to project a mask of size image into a mask of small size (extracts roi then resize)
    this is done on the training samples
    
    '''
    #TO:DO fchange the bbx list into tensor with S,14,14 | predictions S,C,14,14
    #print(f' the num of images {len(images)}')
    out_list=[]
     
    for i in range(len(images)):
        #print(f'the i value {i}')
        gt_masks=targets[i]['masks']
        bbxs=targets[i]['boxes']
           
        mask_RoIAs=[]
      
        for id,gt_mask in enumerate(gt_masks):
            if flag=="roia":
                bbx=bbxs[id]
                #print(f'the bbx is {bbx}')
                #put gt_mask in orderd dict
                #print(f'shape of box {bbx.shape}')
                #print(f'shape of mask {gt_mask.shape}')
                gt_mask_dic=OrderedDict()
                gt_mask_dic['0']=gt_mask.unsqueeze(0).unsqueeze(0).type(torch.float32) #torch.rand(1, 5, 800, 800) #gt_mask
                #print(gt_mask_dic)
                #init the roia objetc
                #RODO: change (28,28)
                RoIA=MultiScaleRoIAlign(featmap_names=['0'], output_size=(16, 16), sampling_ratio=4)
                #apply ROI
                #bbxs= torch.rand(6, 4) * 256; bbxs[:, 2:] += bbxs[:, :2]
                bbx=[bbx.unsqueeze(0)] #change this later
                img_size=[(gt_mask_dic['0'].shape[-2],gt_mask_dic['0'].shape[-1])]
                #TODO: you might need to add a threshold
                #img_size=[(800,800)]

                #print(img_size)
                mask_RoIAs.append(RoIA(gt_mask_dic,bbx,img_size).squeeze(0))
            elif flag=="roia_same":
                #print("I am at roia_same")
                gtmask=gt_mask.unsqueeze(0).unsqueeze(0).type(torch.float32)
                M=projection_size
                bbx=bbxs[id]
                bbx=[bbx.unsqueeze(0)] 
                rois=bbx
                
                projected=roi_align(gtmask, rois, (M, M), 1.)[:, 0] #mychange
                mask_RoIAs.append(projected)
                '''
                print(f'the shape of projected is {projected.shape}')
                plt.imshow(projected[0].detach().cpu().numpy())
                plt.show()
                '''
                

            elif flag=="simple":
                #you should take the part of bbx omly then resize it
                bbx=bbxs[id]
                bbx=bbx.cpu().numpy()
                xmin,ymin,xmax,ymax=round(bbx[0]),round(bbx[1]),round(bbx[2]),round(bbx[3]) #TO DO: you might need to round them and this might cause an issue
                #print(xmin,ymin,xmax,ymax) 
                gt_mask=gt_mask.type(torch.float32).cpu().numpy() #torch.rand(1, 5, 800, 800) #gt_mask
                #print(type(gt_mask))
                #print(gt_mask.shape)
                #TODO: try different interpolation | replace by project_masks_on_boxes
                gt_mask=cv.resize(gt_mask[ymin:ymax,xmin:xmax], (28, 28), interpolation = cv.INTER_NEAREST) #(w,h)
                gt_mask=torch.from_numpy(np.expand_dims(gt_mask, axis=0))
                mask_RoIAs.append(gt_mask)
            elif flag=="dynamic":
                #print("in dynamic")
                #you should take the part of bbx omly then resize it
                bbx=bbxs[id]
                bbx=bbx.cpu().numpy()
                xmin,ymin,xmax,ymax=round(bbx[0]),round(bbx[1]),round(bbx[2]),round(bbx[3]) #TO DO: you might need to round them and this might cause an issue
                #print(xmin,ymin,xmax,ymax) 
                gt_mask=gt_mask.type(torch.float32).cpu().numpy() #torch.rand(1, 5, 800, 800) #gt_mask
                #print(type(gt_mask))
                #print(gt_mask.shape)
                gt_mask=gt_mask[ymin:ymax,xmin:xmax] 
                gt_mask=torch.from_numpy(np.expand_dims(gt_mask, axis=0))
                mask_RoIAs.append(gt_mask)
            
            #print(f'the type of the ROIA in the mas_ROIA list: {type(mask_RoIAs[0])}')
        if flag!="dynamic":
            #print("I should not be here")
            mask_RoIAs=torch.cat(mask_RoIAs, 0)
            #print(f'the shapeof the ROIA in the mas_ROIA : {mask_RoIAs.shape}')
        #in case of dynamic the output will be a list of lists 
        #each internal list is a list of gt_masks per img 
        out_list.append(mask_RoIAs)
    return out_list

def project_masks_on_predicted_boxes_binary(target,pboxs,image,projection_size,MaxIousRowIndex,MaxIousColIndex ): #used as gt to train
    '''
    used to project a mask of size image into a mask of small size (extracts roi then resize)
    this is done on the training samples
    
    '''
    #TO:DO fchange the bbx list into tensor with S,14,14 | predictions S,C,14,14
    #print(f' the num of images {len(images)}')
  
     

    gt_masks=target['masks']
    #TODO: here we should select only gt acc to selected iou
    #select bbxs and gtmasks
    print(MaxIousRowIndex,MaxIousColIndex)
    num_samples=MaxIousRowIndex.shape[0] # check 0 or 1

    selected_pboxes=torch.zeros((num_samples,pboxs.shape[-1]))
    selected_gt_masks=torch.zeros((num_samples,gt_masks.shape[-2],gt_masks.shape[-1]))
    counter=0
    print(f'selected_pboxes {selected_pboxes.shape} selected_gt_masks {selected_gt_masks.shape}')
    for row,col in zip(MaxIousRowIndex,MaxIousColIndex):
        selected_pboxes[counter,:]=pboxs[row,:]
        selected_gt_masks[counter,:,:]=gt_masks[col,:,:]
        counter+=1
        
    mask_RoIAs=[]
    
    for id,gt_mask in enumerate(selected_gt_masks):
        

        #print("I am at roia_same")
        gtmask=gt_mask.unsqueeze(0).unsqueeze(0).type(torch.float32)
        M=projection_size
        bbx=selected_pboxes[id]
        bbx=[bbx.unsqueeze(0)] 
        rois=bbx
        
        projected=roi_align(gtmask, rois, (M, M), 1.)[:, 0] #mychange
        mask_RoIAs.append(projected)
        #visualize
        '''
        print(f'the shape of projected is {projected.shape}')
        plt.imshow(projected[0].detach().cpu().numpy())
        plt.show()
        '''
            

        

    mask_RoIAs=torch.cat(mask_RoIAs, 0)

    return mask_RoIAs

#inference function
def inference(targets,images,detectionmodel,activation,classID,modelClass,activation_value,projection_size,detectionpipeline,device,draw):
    sigmoid=nn.Sigmoid()
    target_out=project_masks_on_boxes_binary(targets,images,projection_size,flag="roia_same")

    for id,img in enumerate(images): 
        print(f'the img shape {img.shape}')
        gtMasksProjected=target_out[id]# projected acc to gt / do project acc to p
        gtbbxs=targets[id]['boxes']
        gtclasses=targets[id]['labels']
        #detection 
        bbxs,labels,masks,scores,features=detection(detectionpipeline,detectionmodel,img,activation)
        #threshold
        pbbxs,pprobas,pclasses,keep=predictionThresholdMaskRcnn(bbxs,labels,scores,0) #all classes
        #get features
        masks_det=masks[keep,:,:]

        #get features
        # features=activation[activation_value][0] #only if act is input
        features=features[keep,:,:]
        if features.shape[0]==0:
            continue #this will not work if it was the last image in batch
        #select features and gtdata s
        #in this case gtclasses only belong to one class
        # print(f'pclasses before {pclasses}')
        # keep1= pclasses==classID
        # print(keep1)
        # pbbxs=pbbxs[keep1,:]
        # pprobas=pprobas[keep1]
        # pclasses=pclasses[keep1]/int(classID)
        # print(f'pclasses is {pclasses}')
        # features=features[keep1,:,:]
        # masks_det=masks[keep1,:,:]


        pclasses=torch.ones_like(pclasses) #for same puzzel

        #DEBUG: show the pbbxs
        MaxIousRowIndex,MaxIousColIndex =matcherIndex(pbbxs,pclasses,gtbbxs,gtclasses)#match at the augmente level
        #TODO during training do not restric yourself to one prediction per gt bounding box
        featuresSelected,gtMasksSelected=matchMaskrcnn(features,gtMasksProjected,MaxIousRowIndex,MaxIousColIndex)# you are assuming that the features have same order as pbbx
        #select pboxs
        num_samples=MaxIousRowIndex.shape[0]
        pboxs_selected=torch.zeros((num_samples,4))
        gtbbxs_selected=torch.zeros((num_samples,4))
        count=0
        for row,col in zip(MaxIousRowIndex,MaxIousColIndex):
            pboxs_selected[count,:]=pbbxs[row,:]
            gtbbxs_selected[count,:]=gtbbxs[col,:]
            count+=1

        #you did not use featuresselected
        print(f' shape of features {features.shape} features selected{featuresSelected.shape} gtMasksProjected {gtMasksProjected.shape} gtMasksSelected {gtMasksSelected.shape} pbxs {bbxs.shape} gtbbxs {gtbbxs.shape}')
        out_one=modelClass(featuresSelected.to(device=device)).to(device=device) #unsqueeze to add channel dim/ 
        out_one=sigmoid(out_one).squeeze(1)#sigmoid(out_one[:,1,:,:])
        print(f'out_one.shape {out_one.shape}')
        #get the target_out corresponding to this img
        target_out_sample=gtMasksSelected.to(device=device)
        print(f'the shape of images {img.shape[:2]}')
        print(f' the shape of outs : {out_one.shape}')
        image_masks=project_back_on_image(pbbxs,out_one,img_org_size=img.shape[:2])
        print(f'image_masks.shape {image_masks.shape}')



        drawfn(img,gtbbxs_selected[0,:],pboxs_selected[0,:])
        vis = Visualizer()
        vis.set_image(img)
        vis.draw_bboxes(pboxs_selected, edge_colors='r')
        #vis.draw_bboxes(pbbxs[keep,:], edge_colors='g')
        vis.get_image()
        vis.show()
         
        if draw:
            for id3,mask in enumerate(image_masks):
                print(f"score {pprobas[id3]}")
                print(f'image num {id}')
                mask=mask.squeeze(0).detach().cpu().numpy()
                mask=mask>0.5
                print(f'the shape of mask in img {mask.shape}')
                plt.imshow(mask)
                plt.show()
            for id2,tar in enumerate(target_out_sample):
                plt.subplot(target_out_sample.shape[0], 2, id2*2+1)
                print(f'the shape of tar.shape {tar.shape}')
                plt.imshow(tar.detach().cpu().numpy())
                plt.subplot(target_out_sample.shape[0], 2, id2*2+1+1)
                plt.imshow(out_one[id2].detach().cpu().numpy())
            plt.show()

def drawfn(img,gtMasksSelected,selected_box):
    selected_box=selected_box.detach().cpu().numpy()
    gtMasksSelected=gtMasksSelected.detach().cpu().numpy()
    xmin,ymin,xmax,ymax=round(selected_box[0]),round(selected_box[1]),round(selected_box[2]),round(selected_box[3])
    xmint,ymint,xmaxt,ymaxt=round(gtMasksSelected[0]),round(gtMasksSelected[1]),round(gtMasksSelected[2]),round(gtMasksSelected[3])

    print(xmin,ymin,xmax,ymax)
    print(xmint,ymint,xmaxt,ymaxt)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 4)
    # cv2.rectangle(img, (xmint, ymint), (xmaxt, ymaxt), (255,0,255), 4)
    # img_cropped=img[ymin:ymax,xmin:xmax]
    # img_cropped_smaller = cv2.resize(img_cropped, (28, 28)) #w,h
    cv.imshow("img",img)
    # cv.imshow("imgcropped",img_cropped)
    # cv.imshow("img_cropped_smaller",img_cropped_smaller)
    # plt.imshow(img_cropped_smaller)
    plt.show()
    cv.waitKey(0)

def project_back_on_image(pbbxs,pmasks,img_org_size):
    ''''''
    mask_img_list=[]
    for id1,mask in enumerate(pmasks):
        print(f'id1 {id1}')
        print(f'pbbxs[id1] {pbbxs[id1]}')
        print(f'pbbxs {pbbxs.shape}')
        print(f'img_org_size {img_org_size}')
        print(f'mask shape inside: {mask.shape}')
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

#detection
#take bbxs and features
#keep high scores plus sought class
#pass through mask block
#project mask to image size
#save bbxs, scores, masks in a dictionary 
#return dictionary
#does engine take care of correspondance

def detection(detectionpipeline,Detectionmodel,img,activation):
        if detectionpipeline=="normal":
            detPrediction=Detectionmodel(img.unsqueeze(0)) #might need some format changes 
            #TODO: here we should choose pred >0.9? try different thrs
            #splitdetection[0]
            bbxs,labels,scores=detPrediction[0]['boxes'],detPrediction[0]['labels'],detPrediction[0]['scores']
            features=activation['roi_heads.mask_head.3.0']
            masks=detPrediction[0]['masks'] #TODO: fix this later
        elif detectionpipeline=="detectors" :

            out=mmdet.apis.inference_detector(Detectionmodel,img)
            bbxs=out.pred_instances.bboxes
            labels=out.pred_instances.labels
            masks=out.pred_instances.masks
            scores=out.pred_instances.scores
            features=activation['roi_head.mask_head.2.convs.3.conv']
        elif detectionpipeline=="mask" :
            print("inside mask")
            # img=torch.ones(375,500,3).to(device="cuda")*30

            out=mmdet.apis.inference_detector(Detectionmodel,img)
            bbxs=out.pred_instances.bboxes
            labels=out.pred_instances.labels
            masks=out.pred_instances.masks
            scores=out.pred_instances.scores
            #print(f'scores {scores}')
            features=activation['roi_head.mask_head.upsample'][0]
        elif detectionpipeline=="htc" :
            out=mmdet.apis.inference_detector(Detectionmodel,img)
            bbxs=out.pred_instances.bboxes
            labels=out.pred_instances.labels
            masks=out.pred_instances.masks
            scores=out.pred_instances.scores
            #print(f'scores {scores}')
            features=activation['roi_head.mask_head.2.convs.3.conv']
        return bbxs,labels,masks,scores,features
def coco80_to_coco91_class(index):  # converts 80-index (val2014) to 91-index (paper) 

    x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90] 

    return x[index]

def quantitative_new(images,Detectionmodel,activation,classID,detectionpipeline,DetUnoModel,DetUnoScoreModel,det,det_score,mask_head,MaskUnoModel,device,idimg,idall,idcats):
    sigmoid=nn.Sigmoid()
    out_dict_list=[]
    for id,img in enumerate(images): 
        img_org_size=img.shape
        #detect all (do this)
        #######detection ALL#############
        pbbxs,pclasses,image_masks,pprobas,features=detection(detectionpipeline,Detectionmodel,img,activation)
        img_trans_size=torch.load("/home/jawad/codes/MaskUno/roi_save/img_shape_trans.pt")[:2] #was before which was wrong
        #change to coco eval format
        for i,cl in enumerate(pclasses):
            pclasses[i]=coco80_to_coco91_class(cl)

        print(f'pclasses after trans to coco91 {pclasses}')
        # print(pprobas)
        # #keep sought class
        # keep1= pclasses==17#classID
        # pbbxs,pprobas,pclasses,image_masks=pbbxs[keep1,:],pprobas[keep1],pclasses[keep1],image_masks[keep1,:,:]

        # print(f'pclasses {pclasses} classID {classID}')
        # # print(f'mask det shape {masks_det.shape}')
        # pbbxs,pprobas,pclasses,image_masks=pbbxs[keep1,:],pprobas[keep1],pclasses[keep1]/int(classID),image_masks[keep1,:,:]
        # features=features[keep1,:,:]
        ######Detect UNO##############################
        # pbbxs,pclasses,pprobas=detectUno(DetUnoModel,DetUnoScoreModel,activation,img,img_org_size,img_trans_size,classID)
        # pbbxs,pprobas,pclasses=detectall_noids(DetUnoModel,DetUnoScoreModel,activation,img_org_size)
        if pbbxs.shape[0]==0:
            #no pridictions
            print("no pridictions") 
            # continue
        

        if 17 in pclasses:
            sought_index=list(pclasses).index(17)
            print(f'sought_index {sought_index}')
            print(pprobas[sought_index])
            
            print("inside 17 in pclasses")
            if pprobas[sought_index]>0:
                pbbxs,pprobas,pclasses,image_masks=detectall_noids(DetUnoModel,DetUnoScoreModel,activation,mask_head,MaskUnoModel,img_org_size)
                pclasses=pclasses*17
                #save image index
                print(idimg)
                if idall[idimg] not in idcats:
                    #save in
                    file1 = open("/home/jawad/codes/MaskUno/imgids.txt", "a")
                    file1.writelines(f'{idimg}\n')
                    file1.close()
  
        # pbbxs,pclasses,scores=detectall(det,det_score,activation,img,img_org_size,img_trans_size) #out is not ordered acc to scores but to index oreder
        # keep1= pclasses==classID
        # pbbxs,pprobas,pclasses,features=pbbxs[keep1,:],pprobas[keep1],pclasses[keep1]/int(classID),features[keep1,:,:]
        ##### Mask UNO #####
        #draw
        # vis = Visualizer()
        # vis.set_image(img)
        # vis.draw_bboxes(pbbxs, edge_colors='r')
        # #vis.draw_bboxes(pbbxs[keep,:], edge_colors='g')
        # vis.get_image()
        # vis.show()
        #pass through mask block
        # out_one=MaskUnoModel(features.to(device=device)).to(device=device) #unsqueeze to add channel dim/ 
        # out_one=sigmoid(out_one)#sigmoid(out_one[:,1,:,:])
        # print(f'out_one {out_one.shape}')
        # # print(f'pbbxs shape {pbbxs_all.shape} pbbxs_uno {pbbxs.shape}')
        # #prepare
        # image_masks=project_back_on_image(pbbxs,out_one,img_org_size=img.shape[:2])#TODO: .shape[-2:] if pytorch
        #save bbxs, scores, masks in a dictionary 
        # print(f'image_masks {image_masks.shape}')
        # print(f'classes {pclasses}')
        dict_img={}
        dict_img['boxes']=pbbxs#pbbxs
        #TODO: take them from prediction of faster rcnn | change labels
        dict_img['labels']=pclasses#*17#torch.ones_like(pclasses)#pclasses
        dict_img['scores']=pprobas
        dict_img['masks']= image_masks#masks_det#image_masks#torch.ones((pbbxs.shape[0],img.shape[0],img.shape[1]))#masks_det#image_masks#image_masks#masks_det#image_masks#masks_det# #instance_img_tensor #TODO: fix later instance_list or instance_img_tensor

        if pbbxs.shape[0]==0:
            print("no predictions in this image")
            # continue
        out_dict_list.append(dict_img)

    return out_dict_list



        


        
       




def quantitative(images,Detectionmodel,activation,classID,modelClass,activation_value,detectionpipeline,device):
    sigmoid=nn.Sigmoid()
    out_dict_list=[]

    for id,img in enumerate(images): 

        #######detection ALL#############
        bbxs,labels,masks,scores,features=detection(detectionpipeline,Detectionmodel,img,activation)
        #get rois_bbxs plus ids_selected_nms -->selected rois
        #get rois_after_boxhead plus ids_selected_nms --->selected box_features
        #inference box predictiob head ---> deltas 
        #add deltas to selected_roi_boxes 
        #scale boxes
        print(f'labels {labels}')
        #detection threshold
        pbbxs,pprobas,pclasses,keep=predictionThresholdMaskRcnn(bbxs,labels,scores,0) #all classes
        print(pbbxs.shape,pprobas.shape,pclasses.shape,keep.shape)
        print(f'keep {keep}')
        print(f'masks.shape {masks.shape}')
        if keep.shape[0]==0:
            print("keep shape zero")
            continue
        #masks=masks.squeeze(1) #only for normal
        masks_det=masks[keep,:,:]
        #get features
        #get features
        # features=activation[activation_value]#[0]#TODO remove thi later  [0]
        print("remove this")
        print(features.shape)
        features=features[keep,:,:]
        # if features.shape[0]==0:
        #     continue #this will not work if it was the last image in batch
        ###keep only sought class##
        print(f'classid  {classID}')
        print(pclasses)
        
        # keep1= pclasses==classID
        # print(keep1)
        # pbbxs=pbbxs[keep1,:]
        # pprobas=pprobas[keep1]
        # pclasses=pclasses[keep1]/int(classID)
        
        # print(f'pclasses is {pclasses}')
        # features=features[keep1,:,:]
        # masks_det=masks_det[keep1,:,:]

        pclasses=torch.ones_like(pclasses)
 
        #pass through mask block
        out_one=modelClass(features.to(device=device)).to(device=device) #unsqueeze to add channel dim/ 
        out_one=sigmoid(out_one)#sigmoid(out_one[:,1,:,:])

        #project masks into image
        print(f'if pytorch .shape[-2:] else .shape[:2]')
        image_masks=project_back_on_image(pbbxs,out_one,img_org_size=img.shape[:2])#TODO: .shape[-2:] if pytorch
        #save bbxs, scores, masks in a dictionary 
        print(f'image_masks {image_masks.shape}')
        dict_img={}
        dict_img['boxes']=pbbxs
        #TODO: take them from prediction of faster rcnn | change labels
        dict_img['labels']=pclasses
        dict_img['scores']=pprobas
        #dict_img['masks']=instance_img_tensor
        print('changed predicted mask into masks_detfrom detection')

        dict_img['masks']= image_masks#image_masks#masks_det#image_masks#masks_det# #instance_img_tensor #TODO: fix later instance_list or instance_img_tensor
        # if len(pbbxs)==0: #TODO: this is wrong you should only add not replace all
        #     #TODO: just a trick
        #     print('all boxes empty!!!!!!!!!!!!!!!!!!!!!!!!!')

        #     print(pbbxs)
        #     dict_img['boxes']=torch.tensor([[10,10,20,20]],dtype=torch.float32).to(device='cuda')
        #     dict_img['labels']=torch.tensor([1],dtype=torch.int64)
        #     dict_img['scores']=torch.tensor([0]) #TODO: find a better way
        #     dict_img['masks']=torch.zeros((1,img.shape[-2],img.shape[-1]))
        #print(f'mask shape {instance_img_tensor.shape}' )
        #append
        if pbbxs.shape[0]==0:
            print("no predictions in this image")
            continue
        out_dict_list.append(dict_img)

    return out_dict_list





if __name__=="__main__":

    #test: IOUMatrix(pbbxs,gtbbxs,pclasses,gtclasses)
    #get data
    # mytransform=transforms.Compose([
    #     transforms.ToTensor(),
    #     #transforms.Resize((480,640)), #this cause mask and img to have different size
    #     transforms.ConvertImageDtype(torch.float),
    # ])

    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    # dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name='cat')
    # data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
    # for images,targets in data_loader:
    #     images = list(image for image in images) # each sample in the batch is no in a list 
    #     targets = [{k: v for k, v in t.items()} for t in targets] # each target in the batch is nowin a list   
    #     # print(targets) #list of ditionaries # each having boxes,labels,masks,image_id,iscrowd
    #     gtboxes=targets[0]["boxes"]
    #     gtclasses=targets[0]["labels"]
    #     print(gtboxes,gtclasses)
    #     print(IOUMatrix(torch.tensor([[10,10,50,50]]),gtboxes,gtclasses,gtclasses))
    #     print( torch.tensor([[10,10,50,50]]).shape, torch.tensor([10,10,50,50]).shape )
    #     break
    #apply it to real predictions #this requires new dataloader
    t=torch.rand(3,4)
    rows=[0,2]

    
