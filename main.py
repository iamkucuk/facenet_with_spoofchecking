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
from scipy.spatial.distance import cosine
#%%
device = torch.device("cpu")#'cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
#%%
img = Image.open("dataset/matthew2.jpg")
img_cropped1 = mtcnn(img)
plt.imshow(img_cropped1.permute((1, 2, 0)))
#%%
img = Image.open("dataset/matthew3.jpg")
img_cropped2 = mtcnn(img)
plt.imshow(img_cropped2.permute((1, 2, 0)))
#%%
img_embedding1 = resnet(img_cropped1.unsqueeze(0).to(device)).cpu().detach().numpy()
img_embedding2 = resnet(img_cropped2.unsqueeze(0).to(device)).cpu().detach().numpy()

dist = cosine(img_embedding1, img_embedding2)

#%%
dist

#%%
