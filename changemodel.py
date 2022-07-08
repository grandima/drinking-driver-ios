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
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)
                  
import coremltools as ct
model = ct.converters.onnx.convert(model='/Users/grandima/Desktop/Local/Personal/Observantai/my_network.onnx')
# Save the CoreML model
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
model.save('my_network.mlmodel')