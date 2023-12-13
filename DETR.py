import torch
import torchinfo
from PIL import Image
import torchvision.transforms as T
import requests
from torch import nn



model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
print(model)
torchinfo.summary(model)
model.eval()
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)

# mean-std normalize the input image (batch-size: 1)
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)
# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
probas1 = outputs['pred_logits'].softmax(-1)
print(probas.shape,probas1.shape)
keep = probas.max(-1).values > 0.9
print(keep.shape)