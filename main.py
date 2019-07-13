#%%
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
#%%
img = Image.open("dataset/emma1.jpg")
img_cropped1 = mtcnn(img)
#%%
img = Image.open("dataset/face5.jpg")
img_cropped2 = mtcnn(img)
#%%
img_embedding1 = resnet(img_cropped1.unsqueeze(0).to(device)).cpu().detach().numpy()
img_embedding2 = resnet(img_cropped2.unsqueeze(0).to(device)).cpu().detach().numpy()

dist = np.linalg.norm(img_embedding1 - img_embedding2)