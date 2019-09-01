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
from scipy.spatial.distance import cosine, euclidean
import joblib
import cv2
import torch
import imp
import numpy as np
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#%%
mtcnn = MTCNN(device=device)
#%%
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
#%%
#Adding my embeddings to dict
img = Image.open("dataset/furkan.jpg")
img_cropped = mtcnn(img)
img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).cpu().detach().numpy()
saved_embeddings = {"FURKAN" : img_embedding}
#%%
threshold_cosine = .5
threshold_euc = .7
#%%
MainModel = imp.load_source('MainModel', "model/spoof_detection.py")
spoof_detector = torch.load("model/spoof_detection.pth").float().to(device)
spoof_detector.eval()

print("Loaded model from disk")

# Read the users data and create face encodings
#known_names, known_encods = get_users()

font = cv2.FONT_HERSHEY_DUPLEX

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
input_vid = []

flush_counter = 0

text = ""

while True:
    ret, frame = video_capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        pil_image = Image.fromarray(frame)
        img_cropped = mtcnn(pil_image)
    except:
        text = "ARMED"
        cv2.putText(frame, text, (100, 100), font, 1.0, (0, 0, 0), 1)
        cv2.imshow("Video", frame)
        flush_counter = flush_counter + 1
        if flush_counter >= 24:
            input_vid = []
            flush_counter = 0
        continue

    if img_cropped is None:
        text = "ARMED"
        cv2.putText(frame, text, (100, 100), font, 1.0, (0, 0, 0), 1)
        cv2.imshow("Video", frame)
        flush_counter = flush_counter + 1
        if flush_counter >= 24:
            input_vid = []
            flush_counter = 0
        continue

    flush_counter = 0

    liveimg = cv2.resize(frame, (100, 100))
    liveimg = cv2.cvtColor(liveimg, cv2.COLOR_RGB2GRAY)
    input_vid.append(torch.tensor(liveimg/255).float())
    flush_counter = 0

    if len(input_vid) >= 24:
        inp = torch.stack(input_vid[-24:])
        pred = spoof_detector(inp.unsqueeze(0).unsqueeze(0).to(device))
        input_vid = input_vid[-25:]
        if pred.data[0][0] > .9 :
            name = ""

            for saved_name in saved_embeddings:
                saved_embedding = saved_embeddings[saved_name]
                try:
                    current_embedding = resnet(img_cropped.unsqueeze(0).to(device)).cpu().detach().numpy()
                except:
                    continue
                dist_cosine = cosine(saved_embedding, current_embedding)
                dist_euclidian = euclidean(saved_embedding, current_embedding)
                if dist_cosine < threshold_cosine:
                    name = saved_name
                    break
                if dist_euclidian < threshold_euc:
                    name = saved_name
                    break

            if name != "":
                face_names.append(name)
                text = name
            else:
                text = "FACE UNIDENTIFIED"

        else:
            text = str(pred)

    else:
        text = "ARMED"

    cv2.putText(frame, text, (100, 100), font, 0.5, (0, 0, 0), 1)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#predict = ()

#%%
