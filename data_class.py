# from gluoncv import data, utils
import numpy as np
import math
import cv2 as cv
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  
# from gluoncv.utils import viz 
# from gluoncv.data.transforms import mask as tmask
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import re
from helper_functions import *
from pycocotools.coco import COCO 
import skimage.io as io




#This is the dataset that we are using
def num_images(class_name):
        
        annFile=f'/home/jawad/datasets/annotations/instances_train2017.json'
        coco_obj=COCO(annFile)
        class_name=class_name
        # display COCO categories and supercategories
        cats = coco_obj.loadCats(coco_obj.getCatIds()) 
        nms=[cat['name'] for cat in cats]
        
        # get all images containing given categories, select one at random
        catIds = coco_obj.getCatIds(catNms=[class_name]) # more names means intersection not at least one
        
        imgIds = coco_obj.getImgIds(catIds=catIds ) #loop these images\
        return len(imgIds)


class COCODatasetUNOApi(Dataset):
    def __init__(self,transforms,stage,class_name):
        self.stage=stage #instances_train2017
        self.annFile=f'/home/jawad/datasets/annotations/instances_{self.stage}2017.json'
        self.coco_obj=COCO(self.annFile)

        self.class_name=class_name
        # display COCO categories and supercategories
        self.cats = self.coco_obj.loadCats(self.coco_obj.getCatIds()) 
        self.nms=[cat['name'] for cat in self.cats] #just to display all cat
        # print(self.nms)
        # get all images containing given categories, select one at random
        self.catIds = self.coco_obj.getCatIds(catNms=[self.class_name]) 
        # print(f'self.catIds {self.catIds}')
        # self.imgIds = self.coco_obj.getImgIds(catIds=[17] ) 
        if self.class_name!="all":
            self.imgIds = self.coco_obj.getImgIds(catIds=self.catIds ) ## more ids means intersection not at least one
        else:
            self.imgIds = self.coco_obj.getImgIds() ## more ids means intersection not at least one


        #print(f'image ids : {self.imgIds}')
        self.transforms=transforms


    def __getitem__(self, index):
        #the image id is 289393 with index 16 122 188 248 256 339(critical) 390 440 495 541 548
        print(f'the image id is {self.imgIds[index]} with index {index}')
        img_info = self.coco_obj.loadImgs(self.imgIds[16])[0] #87#np.random.randint(0,len(imgIds))# chose one random image
        file_name=img_info['file_name']
        img=cv.imread(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')#.transpose(2,0,1)

        #print(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')
        #img=img.permute(1, 2, 0).astype(np.uint8)
        #if image gray
        if len(img.shape)==2:
            #change to rgb from gray
            # img=cv.cvtColor(img,cv.COLOR_GRAY2RGB)
            img=cv.cvtColor(img,cv.COLOR_GRAY2BGR)

        else:
            #from bgr to rgb
            # img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            print("change the color rgb") #TODO: pytorch rgb mmdet bgr
            pass




        annIds = self.coco_obj.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)#this choses the clases ann
        anns = self.coco_obj.loadAnns(annIds)
        masks=[]
        boxes=[]
        areas=[]
        iscrowds=[]
        category_ids=[]
        #the image id is 153217 with index 54
        if len(anns)==0: #in case empty reqpeat image that we are sure not empty
            #TODO: this implementation will give the wrong image with teh annotation taken
            img_info = self.coco_obj.loadImgs(153217)[0] #87#np.random.randint(0,len(imgIds))# chose one random image
            file_name=img_info['file_name']
            img=cv.imread(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')
            if self.class_name!="all":
                print('empty anns')
                annIds = self.coco_obj.getAnnIds(imgIds=153217, catIds=self.catIds, iscrowd=None)
                anns = self.coco_obj.loadAnns(annIds)
            else:
                print('empty anns')
                self.catIds = self.coco_obj.getCatIds(catNms=[]) 
                print(f'self.catIds {self.catIds}')
                annIds = self.coco_obj.getAnnIds(imgIds=153217, catIds=self.catIds, iscrowd=None)
                anns = self.coco_obj.loadAnns(annIds)
                # print(anns)
        
        image_id=anns[0]['image_id']

        for ann in anns: #each object in img
            #print(ann)
            #mask=ann["segmentation"] #TODO: should not be zero 
            #change from poly to mask
            polys=[]
            #print(ann['segmentation'])
            mask=self.coco_obj.annToMask(ann)
            #print(mask.shape)
            #print(type(mask))
            box=[ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]]
            #print(box[0],box[1])
            if box[0]>=box[2]: 
                print(box)
                box=[box[0],box[1] + 2,box[2],box[3]]
                print("there is a problem")
                print(box)
            if box[1]>=box[3]:
                print(box)
                box=[box[0],box[1] ,box[2],box[3]+2]
                print("there is a problem")
                print(box)

            masks.append(mask)
            boxes.append(box)
            areas.append(ann['area'])
            if self.class_name!="all":
                category_ids.append(1) #ann['category_id'] #changed this to 1 since binary case
            else:
                category_ids.append(ann['category_id'])

            iscrowds.append(ann['iscrowd'])

        #stack them in arrays
        masks=np.stack(masks)
        boxes=np.array(boxes)
        category_ids=np.array(category_ids)
        # print(f'category_ids {category_ids}')
        iscrowds=np.array(iscrowds)
        areas=np.array(areas)
        #fix format
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) #.squeeze(0)
        labels = torch.as_tensor(category_ids, dtype=torch.int64) #.squeeze(0)
        masks = torch.as_tensor(masks, dtype=torch.uint8) #.squeeze(0)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64) #.squeeze(0)
        image_id = torch.as_tensor([image_id])
        areas=torch.as_tensor(areas)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowds #TODO: check this
        # target["boxes"] = torch.tensor([])
        # target["labels"] = torch.tensor([])
        # target["masks"] = torch.tensor([])
        # target["image_id"] = image_id
        # target["area"] = torch.tensor([])
        # target["iscrowd"] = torch.tensor([]) #TODO: check thi
        #print(f'masks {masks.shape},boxes {boxes.shape} areas {areas.shape} category_ids {labels.shape} {iscrowds.shape}')
        #print(target)
        #print("move to second images")

        if self.transforms is not None:
            train_image=self.transforms(img) # why not change the mask as well check
            #print(f'train_image {train_image.shape}')


        return train_image,target
    
    def __len__(self):
        return len(self.imgIds)

    

class COCODatasetTwoUNOApi(Dataset):
        def __init__(self,transforms,stage,class_names):
            self.stage=stage #instances_train2017
            self.annFile=f'/home/jawad/datasets/annotations/instances_{self.stage}2017.json'
            self.coco_obj=COCO(self.annFile)
            self.class_names=class_names
            # display COCO categories and supercategories
            self.cats = self.coco_obj.loadCats(self.coco_obj.getCatIds()) 
            self.nms=[cat['name'] for cat in self.cats]
            #print(self.nms)
            # get all images containing given categories, select one at random
            self.catIds1 = self.coco_obj.getCatIds(catNms=[self.class_names[0]])
            self.catIds2 = self.coco_obj.getCatIds(catNms=[self.class_names[1]]) 
 
            
            
            self.imgIds1 = self.coco_obj.getImgIds(catIds=self.catIds1 ) ## more ids means intersection not at least one
            self.imgIds2 = self.coco_obj.getImgIds(catIds=self.catIds2 ) ## more ids means intersection not at least one
            self.imgIds=self.imgIds1 + self.imgIds2 
            


            #print(f'image ids : {self.imgIds}')
            self.transforms=transforms
        
        def catidmap(self,catid):
            if catid==self.catIds1:
                idUno=1
            else: 
                idUno=2
            return idUno


        def __getitem__(self, index):
            print(f'the image id is {self.imgIds[index]} with index {index}')

            img_info = self.coco_obj.loadImgs(self.imgIds[index])[0] #np.random.randint(0,len(imgIds))# chose one random image
            file_name=img_info['file_name']
            #img = io.imread(f'/home/jawad/datasets/{self.stage}2017_imgs/{file_name}')


            img=cv.imread(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')#.transpose(2,0,1)
            #print(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')
            #img=img.permute(1, 2, 0).astype(np.uint8)
            #if image gray
            if len(img.shape)==2:
                #change to rgb from gray
                img=cv.cvtColor(img,cv.COLOR_GRAY2RGB)
            else:
                #from bgr to rgb
                img=cv.cvtColor(img,cv.COLOR_BGR2RGB)



            annIds = self.coco_obj.getAnnIds(imgIds=img_info['id'], catIds=np.array([self.catIds1,self.catIds2]), iscrowd=None)
            anns = self.coco_obj.loadAnns(annIds)
            masks=[]
            boxes=[]
            areas=[]
            iscrowds=[]
            category_ids=[]


            image_id=anns[0]['image_id']

            for ann in anns: #each object in img
                #print(ann)
                #mask=ann["segmentation"] #TODO: should not be zero 
                #change from poly to mask
                polys=[]
                #print(ann['segmentation'])
                mask=self.coco_obj.annToMask(ann)
                #print(mask.shape)
                #print(type(mask))
                box=[ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]]
                #print(box[0],box[1])
                if box[0]>=box[2]: 
                    print(box)
                    box=[box[0],box[1] + 2,box[2],box[3]]
                    print("there is a problem")
                    print(box)
                if box[1]>=box[3]:
                    print(box)
                    box=[box[0],box[1] ,box[2],box[3]+2]
                    print("there is a problem")
                    print(box)

                masks.append(mask)
                boxes.append(box)
                areas.append(ann['area'])
                
                category_ids.append(self.catidmap(ann['category_id'])) #ann['category_id'] #changed this to 1 since binary case


                iscrowds.append(ann['iscrowd'])

            #stack them in arrays
            masks=np.stack(masks)
            boxes=np.array(boxes)
            category_ids=np.array(category_ids)
            iscrowds=np.array(iscrowds)
            areas=np.array(areas)
            #fix format
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32) #.squeeze(0)
            labels = torch.as_tensor(category_ids, dtype=torch.int64) #.squeeze(0)
            masks = torch.as_tensor(masks, dtype=torch.uint8) #.squeeze(0)
            iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64) #.squeeze(0)
            image_id = torch.as_tensor([image_id])
            areas=torch.as_tensor(areas)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = areas
            target["iscrowd"] = iscrowds #TODO: check this
            #print(f'masks {masks.shape},boxes {boxes.shape} areas {areas.shape} category_ids {labels.shape} {iscrowds.shape}')
            #print(target)
            #print("move to second images")

            if self.transforms is not None:
                train_image=self.transforms(img) # why not change the mask as well check
                #print(f'train_image {train_image.shape}')


            return train_image,target

        def __len__(self):
            return len(self.imgIds)
        
class COCODatasetUNOApiBeta(Dataset):
    def __init__(self,transforms,stage,class_name):
        self.stage=stage #instances_train2017
        self.annFile=f'/home/jawad/datasets/annotations/instances_{self.stage}2017.json'
        self.coco_obj=COCO(self.annFile)

        self.class_name=class_name
        # display COCO categories and supercategories
        self.cats = self.coco_obj.loadCats(self.coco_obj.getCatIds()) 
        self.nms=[cat['name'] for cat in self.cats] #just to display all cat
        # print(self.nms)
        # get all images containing given categories, select one at random
        self.catIds = self.coco_obj.getCatIds(catNms=[self.class_name]) 
        print(f'self.catIds {self.catIds}')

        # self.imgIds = self.coco_obj.getImgIds(catIds=[17])#[:1] ## more ids means intersection not at least one
        self.imgIds = self.coco_obj.getImgIds(catIds=self.catIds )

        #print(f'image ids : {self.imgIds}')
        self.transforms=transforms


    def __getitem__(self, index):
        print(f'the image id is {self.imgIds[index]} with index {index}')
        img_info = self.coco_obj.loadImgs(self.imgIds[index])[0] #87#np.random.randint(0,len(imgIds))# chose one random image
        file_name=img_info['file_name']
        #img = io.imread(f'/home/jawad/datasets/{self.stage}2017_imgs/{file_name}')


        img=cv.imread(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')#.transpose(2,0,1)
        #print(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')
        #img=img.permute(1, 2, 0).astype(np.uint8)
        #if image gray
        if len(img.shape)==2:
            #change to rgb from gray
            # img=cv.cvtColor(img,cv.COLOR_GRAY2RGB)
            img=cv.cvtColor(img,cv.COLOR_GRAY2BGR)

        else:
            #from bgr to rgb
            # img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            print("change the color rgb") #TODO: pytorch rgb mmdet bgr
            pass



        #we are chosing all ids to prevent empty ann
        print(f'catIds=self.catIds {self.catIds}')
        annIds = self.coco_obj.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco_obj.loadAnns(annIds)
        masks=[]
        boxes=[]
        areas=[]
        iscrowds=[]
        category_ids=[]

        if len(anns)==0: #in case empty reqpeat image that we are sure not empty
            print(f'anns is empty')
            img_info = self.coco_obj.loadImgs(153217)[0] #87#np.random.randint(0,len(imgIds))# chose one random image
            file_name=img_info['file_name']
            img=cv.imread(rf'/home/jawad/datasets/{self.stage}2017/{file_name}')
            print('empty anns')
            annIds = self.coco_obj.getAnnIds(imgIds=153217, catIds=self.catIds, iscrowd=None)
            anns = self.coco_obj.loadAnns(annIds)
            
        
        image_id=anns[0]['image_id']

        for ann in anns: #each object in img
            #print(ann)
            #mask=ann["segmentation"] #TODO: should not be zero 
            #change from poly to mask
            polys=[]
            #print(ann['segmentation'])
            mask=self.coco_obj.annToMask(ann)
            #print(mask.shape)
            #print(type(mask))
            box=[ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]]
            #print(box[0],box[1])
            if box[0]>=box[2]: 
                print(box)
                box=[box[0],box[1] + 2,box[2],box[3]]
                print("there is a problem")
                print(box)
            if box[1]>=box[3]:
                print(box)
                box=[box[0],box[1] ,box[2],box[3]+2]
                print("there is a problem")
                print(box)

            masks.append(mask)
            boxes.append(box)
            areas.append(ann['area'])

            iscrowds.append(ann['iscrowd'])

        #stack them in arrays
        masks=np.stack(masks)
        boxes=np.array(boxes)
        category_ids=np.array(category_ids)
        print(f'category_ids {category_ids}')
        iscrowds=np.array(iscrowds)
        areas=np.array(areas)
        #fix format
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) #.squeeze(0)
        labels = torch.as_tensor(category_ids, dtype=torch.int64) #.squeeze(0)
        masks = torch.as_tensor(masks, dtype=torch.uint8) #.squeeze(0)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64) #.squeeze(0)
        image_id = torch.as_tensor([image_id])
        areas=torch.as_tensor(areas)
        # print(f'labels {labels}')
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowds #TODO: check this
        # target["boxes"] = torch.tensor([])
        # target["labels"] = torch.tensor([])
        # target["masks"] = torch.tensor([])
        # target["image_id"] = torch.tensor([])
        # target["area"] = torch.tensor([])
        # target["iscrowd"] = torch.tensor([]) #TODO: check thi
        #print(f'masks {masks.shape},boxes {boxes.shape} areas {areas.shape} category_ids {labels.shape} {iscrowds.shape}')
        #print(target)
        #print("move to second images")

        if self.transforms is not None:
            train_image=self.transforms(img) # why not change the mask as well check
            #print(f'train_image {train_image.shape}')


        return train_image,target
    
    def __len__(self):
        return len(self.imgIds)

#check first the general case 
if __name__=="__main__":
   



    
    mytransform=transforms.Compose([
       
        transforms.ToTensor(),
        
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        #transforms.Resize((480,640)), #this cause mask and img to have different size
        transforms.ConvertImageDtype(torch.float),
    ])

    def collate_fn(batch):
        return tuple(zip(*batch))

    dataset=COCODatasetUNOApi(transforms=mytransform,stage='val',class_name="cat")
    # dataset=COCODatasetTwoUNOApi(transforms=mytransform,stage='train',class_names=['cat','dog'])
    data_loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
    #dataset=COCODatasetUNOApi(transforms=mytransform,stage='instances_train2017',class_name='person')
 
    print(f'length of dataset {len(dataset.imgIds)}')
  
    for images,targets in data_loader:
     
 
        images = list(image for image in images) # each sample in the batch is no in a list 
        targets = [{k: v for k, v in t.items()} for t in targets] # each target in the batch is nowin a list   
        # print(targets)
        #trans=GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
        #images,targets=trans(images,targets)
        #print("here")
        #print(images)
        # print(images[0].shape)
        # print(targets[0]['boxes'].shape)
        #print(targets[0]['boxes'])
        #print(targets[0]['labels'])
        # print(targets[0]['masks'].shape)
        #test
       #print("look hereeeeeeeee")
        
        #out=project_masks_on_boxes_binary(gt_masks=targets[0]['masks'],bbxs=targets[0]['boxes'])
        
        #print(f'the shape of teh out: {out[0].shape}')
    






    
