#!/usr/bin/env python
# coding: utf-8

# # Author: Khursheed Akram

# # Import Libraries

# In[1]:


import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.patches as patches
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
import random
import warnings
warnings.filterwarnings('ignore')


# # Load Pre-Trained Model

# In[2]:


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


# # Load Test Image

# In[3]:


img = Image.open("test.jpg")
img


# # Apply the Image Transformations

# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0)
img_tensor.shape


# # Perform Object Detection

# In[5]:


with torch.no_grad():
    predictions = model(img_tensor)
predictions


# # COCO Dataset Class Labels

# In[6]:


coco_labels = ['__background__','person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'streetsign', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'plate', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'hairbrush']


# # Plot the Results

# In[18]:


fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)
for k in range(len(predictions[0]['boxes'])):
    box = predictions[0]['boxes'][k].cpu().numpy()
    score = predictions[0]['scores'][k].cpu().numpy()
    label = predictions[0]['labels'][k].cpu().numpy()
    if label < len(coco_labels):
            label_name = coco_labels[label.item()]
    else:
        label_name = "Unknown"
    threshold = 0.7
    if score > threshold:  
        colors = []
        for rgb in range(len('boxes')):
            colors.append((random.random(), random.random(), random.random()))
        x_min, y_min, x_max, y_max = box
        rec = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2.4 , fill=False , edgecolor=colors[rgb])
        ax.add_patch(rec)
        plt.text(x_min, y_min, f'{label_name}: {score.item():.2f}', color='white', fontsize=12.4, bbox=dict(facecolor='darkred', alpha=0.9))
plt.axis('off')
plt.show()

