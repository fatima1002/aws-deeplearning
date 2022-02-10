import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np
import torch
from six import BytesIO
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# reference
# https://samuelabiodun.medium.com/how-to-deploy-a-pytorch-model-on-sagemaker-aa9a38a277b6

def model_fn(model_dir):
    logger.info('Loading the model.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, 133))
    with open(os.path.join(model_dir, "dogmodel_profdebug.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        # checkpoint = torch.load(f , map_location =device)
        # model.load_state_dict(checkpoint)
    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(request_body, content_type):
    logger.info('Deserializing the input data.')
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    
    if content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request['url']
        logger.info(f'Image url: {url}')
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))


def predict_fn(input_data, model):
    testing_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

    input_object=testing_transform(input_data)
#   convert to 4 dim
    input_object = input_object.unsqueeze(dim=0)
    
    model.eval()
    with torch.no_grad():
        return model(input_object)

