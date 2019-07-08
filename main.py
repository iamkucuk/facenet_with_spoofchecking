#%%
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#%%
img = Image.open("dataset/face1.jpg")
img_cropped = mtcnn(img)
plt.imshow(np.transpose(img_cropped, (1, 2, 0)))
plt.show()
#%%
img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).cpu()
