# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from time import time
import sys, os
import glob
from tqdm.auto import tqdm

from facenet_pytorch.models.mtcnn import MTCNN, prewhiten
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1, get_torch_home
from facenet_pytorch.models.utils.detect_face import extract_face


#%%
def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


#%%
trans = transforms.Compose([
    transforms.Resize(512)
])

trans_cropped = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    prewhiten
])


#%%
dataset = datasets.ImageFolder('dataset/lfw', transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0])


#%%
mtcnn = MTCNN(device=torch.device('cpu'))


#%%
total_item = len(dataset)
names = []
aligned = []
for img, idx in tqdm(loader):
    name = dataset.idx_to_class[idx]
    # start = time()
    img_align = mtcnn(img)#, save_path = "data/aligned/{}/{}.png".format(name, str(idx)))
    # print('MTCNN time: {:6f} seconds'.format(time() - start))

    if img_align is not None:
        names.append(name)
        aligned.append(img_align)

aligned = torch.stack(aligned)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().cpu()

start = time()
embs = resnet(aligned)
print('\nResnet time: {:6f} seconds\n'.format(time() - start))

# # dists = [[(emb - e).norm().item() for e in embs] for emb in embs

# print('\nOutput:')
# print(pd.DataFrame(dists, columns=names, index=names))


