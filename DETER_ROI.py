import torch
import torchinfo
from PIL import Image
import torchvision.transforms as T
import requests
from torch import nn
import sys
sys.path.append(r'/home/jawad/codes/references/detection')
from torchvision.models.detection.transform import GeneralizedRCNNTransform,GeneralizedRCNNTransformMy

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()
#register hook
model.transformer.decoder.norm.register_forward_hook(get_activation('transformer.decoder.norm'))
model.transformer.encoder.layers[5].norm2.register_forward_hook(get_activation('transformer.encoder.layers[5].norm2'))

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)
print(model)
torchinfo.summary(model)


for name, param in model.named_parameters():
    print(name)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = transform(im).unsqueeze(0)
model(img)
print(activation['transformer.decoder.norm'].shape)

# trans=GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
# images,targets=trans([torch.rand(3,500,400)])
# print(img.shape)
# print("##################################################################################################################################3")
# print(model.backbone(images))