import torch.nn as nn
import torchinfo
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

#model for every class
class UnoModel(nn.Module):
    def __init__(self):
        super(UnoModel, self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()

    def forward(self,x):
        #maskHead
        out=self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out)
    
class UnoModelMaskRcnn(nn.Module): #will work for op2 and 3 (better less param)
    def __init__(self):
        super(UnoModelMaskRcnn, self).__init__()

        self.conv1=nn.Conv2d(in_channels=256,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()

    def forward(self,x):
        #maskHead
        out=self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out)
    
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device=device)

    
class UnoModelMaskRcnnSame(nn.Module): #will work for op2 and 3 (better less param)
    def __init__(self):
        super(UnoModelMaskRcnnSame, self).__init__()

        # self.model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device="cuda")
        # self.prediction_head=self.model.roi_heads.mask_predictor
        num_classes=2 #double check 2 or 1
        hidden_layer=256
        # #TODO: change this later
        # # print("change the pretrained from truie to false")
        # self.model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False).to(device="cuda")
        # #model2=torch.load("/home/jawad/codes/OneClassMethod/models/model_complete_cat_onlypredictionhead_softstatic_gt_41.pth")
        # for param in self.model1.parameters():
        #     param.requires_grad = False
        # #added this comment
        # # now get the number of input features for the mask classifier
        # in_features_mask = self.model1.roi_heads.mask_predictor.conv5_mask.in_channels
        # # and replace the mask predictor with a new one
        # self.model1.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        #                                                 hidden_layer,
        #                                         num_classes)
        # #TODO: build the model from scratch
        in_channels=256
        self.conv5_mask=nn.ConvTranspose2d(in_channels, hidden_layer, 2, 2, 0)
        self.relu=nn.ReLU(inplace=True) #make sure that relu is here
        self.mask_fcn_logits=nn.Conv2d(hidden_layer, num_classes, 1, 1, 0)
    def forward(self,x):
        #maskHead
        # out=self.model1.roi_heads.mask_predictor(x)
        out= self.mask_fcn_logits(self.relu(self.conv5_mask(x)))
        # out= self.relu(self.mask_fcn_logits(self.conv5_mask(x)))

        return out
class UnoModelMaskRcnnSameMmdet(nn.Module): #will work for op2 and 3 (better less param)
    def __init__(self):
        super(UnoModelMaskRcnnSameMmdet, self).__init__()

        # self.model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device="cuda")
        # self.prediction_head=self.model.roi_heads.mask_predictor
        num_classes=1 #double check 2 or 1
        hidden_layer=256
        # #TODO: change this later
        # # print("change the pretrained from truie to false")
        # self.model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False).to(device="cuda")
        # #model2=torch.load("/home/jawad/codes/OneClassMethod/models/model_complete_cat_onlypredictionhead_softstatic_gt_41.pth")
        # for param in self.model1.parameters():
        #     param.requires_grad = False
        # #added this comment
        # # now get the number of input features for the mask classifier
        # in_features_mask = self.model1.roi_heads.mask_predictor.conv5_mask.in_channels
        # # and replace the mask predictor with a new one
        # self.model1.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        #                                                 hidden_layer,
        #                                         num_classes)
        # #TODO: build the model from scratch
        in_channels=256
        self.conv5_mask=nn.ConvTranspose2d(in_channels, hidden_layer, 2, 2, 0)
        self.relu=nn.ReLU(inplace=True) #make sure that relu is here
        self.mask_fcn_logits=nn.Conv2d(hidden_layer, num_classes, 1, 1, 0)
    def forward(self,x):
        #maskHead
        # out=self.model1.roi_heads.mask_predictor(x)
        out= self.mask_fcn_logits(self.relu(self.conv5_mask(x)))
        # out= self.relu(self.mask_fcn_logits(self.conv5_mask(x)))
        return out   

class UnoModelMaskRcnnSameMmdetDetection(nn.Module): #will work for op2 and 3 (better less param)
    def __init__(self):
        super(UnoModelMaskRcnnSameMmdetDetection, self).__init__()
        num_classes=1 #double check 2 or 1
        hidden_layer=256
        #TODO: change this later
        # print("change the pretrained from truie to false")
        self.model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False).to(device="cuda")
        #model2=torch.load("/home/jawad/codes/OneClassMethod/models/model_complete_cat_onlypredictionhead_softstatic_gt_41.pth")
        for param in self.model1.parameters():
            param.requires_grad = False
        #added this comment
        # now get the number of input features for the mask classifier
        in_features_mask = self.model1.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        self.model1.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                num_classes)

    def forward(self,x):
        #maskHead
        # out=self.model1.roi_heads.mask_predictor(x)
        out= self.mask_fcn_logits(self.relu(self.conv5_mask(x)))
        # out= self.relu(self.mask_fcn_logits(self.conv5_mask(x)))
        return out     


#class for the combined model 
class AllModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.UnoModel=UnoModel()
        self.UnoModelMaskRcnn=UnoModelMaskRcnn()

    def forward(self,classes,features):
        out=[]
        for indx,ClassNum in enumerate(classes):
            if ClassNum=="cat":
                print(features[indx,:,:,:].unsqueeze(0).shape)
                out.append(self.UnoModel(features[indx,:,:,:].unsqueeze(0)))

            elif ClassNum=="dog":
                out.append(self.UnoModelMaskRcnn(features[indx,:,:,:].unsqueeze(0)))

        return out

                    



if __name__=="__main__":

    allmodel=AllModel()
    print(allmodel)
    print(torchinfo.summary(allmodel))

    for ParamName,_ in allmodel.named_parameters():
        print(ParamName)

    #inference 
    classes=["cat","dog"]
    features=torch.rand(size=(2,1,20,20)) #batch,channels,l,w
    out=allmodel(classes,features)
    print(out[0].shape)
