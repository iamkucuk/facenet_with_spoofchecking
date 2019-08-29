#%%
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.mtcnn import prewhiten
from facenet_pytorch.models.utils.detect_face import extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
#%%
df = pd.DataFrame(columns=["id", "embedding"])
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
total_item = len(dataset)
loader = DataLoader(dataset, collate_fn=lambda x: x[0])
#%%
mtcnn = MTCNN(device=device)
#%%
names = []
embeddings = []
for img, idx in tqdm(loader):
    name = dataset.idx_to_class[idx]
    img_align = mtcnn(img)
    img_embedding = resnet(img_align.unsqueeze(0).to(device)).cpu().detach().numpy()
    if img_align is not None:
        names.append(name)
        embeddings.append(img_embedding)
#%%
df = pd.DataFrame(columns=["name", "embeddings"])
df.name = names
df.embeddings = embeddings
df.to_csv("lfw_embeddings.csv", index=False)
#%%
df = pd.read_csv("lfw_embeddings.csv")
#%%
img = Image.open("dataset/matthew2.jpg")
img_cropped1 = mtcnn(img)
#%%
img = Image.open("dataset/matthew3.jpg")
img_cropped2 = mtcnn(img)
#%%
img_embedding1 = resnet(img_cropped1.unsqueeze(0).to(device)).cpu().detach().numpy()
img_embedding2 = resnet(img_cropped2.unsqueeze(0).to(device)).cpu().detach().numpy()

dist = cosine(img_embedding1, img_embedding2)