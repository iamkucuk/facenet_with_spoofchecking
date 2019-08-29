#%%
from facenet_pytorch import MTCNN, InceptionResnetV1, prewhiten
from facenet_pytorch.models.utils.detect_face import extract_face
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img
#%%
img = get_image("dataset/spoof_dataset/ClientRaw/0001/0001_00_00_01_21.jpg", trans)
img_align = mtcnn(img)
embedding = resnet(img_align.unsqueeze(0))
print(model.predict(embedding.detach().numpy()))
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
dataset = datasets.ImageFolder('dataset/spoof_dataset', transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}
total_item = len(dataset)
# train_dataset, test_dataset = random_split(dataset, [total_item * .8, total_item * .2])
# train_loader, test_loader = DataLoader(train_dataset, collate_fn=lambda x: x[0]), DataLoader(test_dataset, collate_fn=lambda x: x[0])
loader = DataLoader(dataset, collate_fn=lambda x: x[0])
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
#%%
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

#%%
embeddings = []
for elem in tqdm(aligned):
    embeddings.append(resnet(elem.unsqueeze(0).to(device)).cpu().detach().numpy())

#%%
df = pd.DataFrame(columns=["classes", "embeddings"])
df.classes = names
df.embeddings = embeddings
df["numeric_classes"] = 1
df.loc[df["classes"] == "ClientRaw", "numeric_classes"] = 0
df.classes = df.numeric_classes
df = df.drop("numeric_classes", axis=1)
#%%
df.to_csv("spoof_embeddings.csv", index=False)
#%%
df = pd.read_csv("spoof_embeddings.csv")
#%%
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, Booster
from sklearn.metrics import accuracy_score, recall_score, f1_score
#%%
y = df.classes.values
X = list(df.apply(lambda x: x.embeddings.squeeze(), axis=1).values)
# X = df["embeddings"].as_matrix()
#%%
from sklearn.utils.validation import check_X_y
X, y = check_X_y(X,y,
                 accept_sparse=True, # allow all types of sparse
                 force_all_finite=False, # allow nan and inf because
                 multi_output=False)

#%%
# splitter = StratifiedShuffleSplit(n_splits=1, train_size=.8, test_size=.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#%%
model = XGBClassifier()
model.fit(X_train, y_train)

#%%
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#%%
import joblib
joblib.dump(model, "spoof_xgb.bin")
model.save_model("spoofing_xgb.model")

#%%
from xgboost import Booster
model = Booster()
model.load_model("spoofing_xgb.model")
#%%
import joblib
model = joblib.load("spoof_xgb.bin")