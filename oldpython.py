import os
import shutil
import time
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import PIL.Image
from IPython.display import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torchvision import models,transforms,datasets

import coremltools as ct

# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)
# model.load_state_dict(torch.load("./model-driver", map_location=torch.device('cpu')))
# model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
# model.eval()
# model.cpu()

# class_dict = {0 : "safe driving",
#               1 : "texting - right",
#               2 : "talking on the phone - right",
#               3 : "texting - left",
#               4 : "talking on the phone - left",
#               5 : "operating the radio",
#               6 : "drinking",
#               7 : "reaching behind",
#               8 : "hair and makeup",
#               9 : "talking to passenger"}
# class_list = [(k, v) for k, v in class_dict.items()]
# sorted(class_list, key=lambda x: x[0])
# class_list = [i[1] for i in class_list]

# preprocess = transforms.Compose([
    
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

# input_shape = (3, 400, 400)

# img = PIL.Image.open('./img_29.jpg').resize(input_shape[1:])

# input_tensor = preprocess(img)
# input_batch = input_tensor.unsqueeze(0)
# out = model(input_batch.cpu())
# # proba = nn.Softmax(dim=1)(out)
# proba = out
# proba = [round(float(elem),4) for elem in proba[0]]
# print(proba)

# example_input = torch.rand(1, *input_shape)
# traced_model = torch.jit.trace(model, example_input)

# classifier_config = ct.ClassifierConfig(class_list)
# cml_model = ct.convert(traced_model,
#                        inputs=[ct.ImageType(color_layout='RGB', scale=1/(0.5*255.0),
#                                             bias=[-1, -1, -1],
#                                             shape=example_input.shape)
#                                             ],
#                         classifier_config=classifier_config,
# )
# cml_model.save('./BreakfastFinder/OldModel.mlpackage')

model = ct.models.MLModel("./BreakfastFinder/OldModel.mlpackage") 
spec = ct.utils.load_spec("./BreakfastFinder/OldModel.mlpackage")
input = spec.description.input[0]
import coremltools.proto.FeatureTypes_pb2 as ft 
input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
input.type.imageType.height = 400
input.type.imageType.width = 400
ct.utils.save_spec(spec, "./BreakfastFinder/OldModel.mlpackage")



model = ct.models.MLModel("./BreakfastFinder/OldModel.mlpackage") 
input_shape = (3, 400, 400)
img = PIL.Image.open('./img_29.jpg').resize(input_shape[1:])
proba = model.predict({"input_1": img})['var_805']
proba = round(float(proba['drinking']),4)
print(proba)