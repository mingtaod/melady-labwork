import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from skimage.segmentation import quickshift
from explainer import Archipelago
from application_utils.image_utils import *
from application_utils.utils_torch import ModelWrapperTorch
from burn_resnet50 import FineTuneResNet

import warnings
warnings.filterwarnings("ignore")

# Get model with trained weights here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.__dict__["resnet50"](pretrained=True)
model = FineTuneResNet(base_model, 2)
model.load_state_dict(torch.load("D:\\torch-model-weights\\burnCNN_imagewise_weights\\burnCNN_fold4.pt"))
model_wrapper = ModelWrapperTorch(model, device)

# Get Example
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,

])

image_path_txt = "burn_level_images_dataset\\train.txt"
image_path = "burn_level_images_dataset\\Combined"

with open(image_path_txt) as input_file:
    lines = input_file.readlines()
    img_names = [os.path.normpath("%s\%s" % (image_path, line.strip('\n').split(' ')[0])) for line in lines]
    img_labels = [int(line.strip('\n').split(' ')[-1]) for line in lines]

for img_name in img_names[0:1]:
    image = Image.open(img_name)
    image_tensor = preprocess(image)
    image = (
        image_tensor.cpu().numpy().transpose(1, 2, 0) / image_tensor.abs().max().item()
    )
    print("input image")
    plt.imshow(image/2+0.5)
    plt.axis("off")
    plt.show()

    # Get classification
    predictions = model_wrapper(np.expand_dims(image, 0))
    class_idx = predictions[0].argsort()[::-1][0]
    print("classification:", class_idx)

    # Explain prediction
    baseline = np.zeros_like(image)
    segments = quickshift(image, kernel_size=3, max_dist=300, ratio=0.2)
    xf = ImageXformer(image, baseline, segments)
    apgo = Archipelago(model_wrapper, data_xformer=xf, output_indices=class_idx, batch_size=20)
    inter_effects, main_effects = apgo.explain(top_k=15, separate_effects=True)

    # Show explanation
    show_image_explanation(inter_effects.items(), image, segments, main_effects=main_effects.items())
