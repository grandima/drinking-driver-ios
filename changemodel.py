import os
import shutil
import time
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import PIL.Image as PILImage
from IPython.display import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torchvision import models,transforms,datasets

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load("/Users/grandima/Desktop/Local/Personal/Observantai/kaggle/working/model-driver", map_location=torch.device('cpu')))
model.eval()
model.cpu()
dummy_input = torch.randn(1, 3, 400, 400)

# Define input / output names
input_names = ["my_input"]
output_names = ["my_output"]

# Convert the PyTorch model to ONNX
torch.onnx.export(model,
                  dummy_input,
                  "my_network.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)
                  
import coremltools as ct
model = ct.converters.onnx.convert(model='/Users/grandima/Desktop/Local/Personal/Observantai/my_network.onnx')
# Save the CoreML model

model.save('my_network.mlmodel')


path_test = "/Users/grandima/Desktop/Local/Personal/Observantai/kaggle/input/state-farm-distracted-driver-detection/imgs"
transform = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor()])
data = datasets.ImageFolder(root = path_test, transform = transform)

# torch.save(model.state_dict(), 'model_scripted.pt')
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)
# model.load_state_dict(torch.load("/Users/grandima/Desktop/Local/Personal/Observantai/model_scripted.pt", map_location=torch.device('cpu')))
# model.eval()
# model.cpu()
# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save


# img,c = data[0]
# import coremltools as ct
# scale = 1/(0.226*255.0)
# bias = [-0.485/(0.229), -0.456/(0.224), -0.406/(0.225)]
# input = ct.ImageType(
#     shape=img.shape,
#     bias=bias
# )

# model2 = ct.convert('/Users/grandima/Desktop/Local/Personal/Observantai/model_scripted.pt',  inputs=[input])
# mode2.save("model-driver.mlmodel")