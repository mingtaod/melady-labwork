import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients
import tensorflow as tf
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt


# class FineTuneResNet(nn.Module):
#     def __init__(self, original_model, num_classes):
#         super(FineTuneResNet, self).__init__()
#         fc1 = nn.Linear(2048, 1000)
#         relu1 = nn.ReLU()
#         fc2 = nn.Linear(1000, 256)
#         relu2 = nn.ReLU()
#         fc3 = nn.Linear(256, num_classes)
#         self.features = nn.Sequential(*list(original_model.children())[:-1])
#         self.classifier = nn.Sequential(fc1, relu1, fc2, relu2, fc3)
#
#     def forward(self, x):
#         out = self.features(x)
#         # print('prev out.shape: ', out.shape)
#         out = out.view(out.size(0), -1)
#         # print('out.shape: ', out.shape)
#         out = self.classifier(out)
#         return out


class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=8):
        super(FineTuneResnet50, self).__init__()
        self.num_class = num_class
        resnet50_instance = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50_instance.children())[:-1])
        self.classifier = nn.Linear(2048, self.num_class)

    def forward(self, x):
        output = self.features(x)
        print('output.shape: ', output.shape)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output


if __name__ == '__main__':
    model = FineTuneResnet50()
    model.load_state_dict(torch.load("D:\\torch-model-weights\\resnet50_8class.pt"))
    model.float()
    model.eval()
    print("Check point! Load state dict successful...")

    ig = IntegratedGradients(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    img = Image.open('D:\\natural_images_dataset\\natural_images\\cat\\cat_0066.jpg')
    img2 = Image.open('D:\\natural_images_dataset\\natural_images\\car\\car_0125.jpg')
    input_img = def_transform(img)
    input_img2 = def_transform(img2)
    input_img = input_img.unsqueeze(0)
    input_img2 = input_img2.unsqueeze(0)
    input_img = torch.tensor(input_img, dtype=torch.float32)
    input_img2 = torch.tensor(input_img2, dtype=torch.float32)

    baseline = torch.zeros([3, 224, 224], dtype=torch.float32)
    baseline = baseline.unsqueeze(0)
    baseline2 = torch.zeros([3, 224, 224], dtype=torch.float32)
    baseline2 = baseline2.unsqueeze(0)

    unloader = transforms.ToPILImage()

    attributions, approximation_error = ig.attribute(input_img, baseline, target=2, return_convergence_delta=True)
    # aggregated_img = torch.mul(input_img, attributions)
    aggregated_img = attributions
    temp, _ = torch.max(aggregated_img, 0)
    max_val, _ = torch.max(temp, 0)
    temp, _ = torch.min(aggregated_img, 0)
    min_val, _ = torch.min(temp, 0)
    aggregated_img = (aggregated_img - min_val)/(max_val - min_val)
    out_image = aggregated_img.squeeze(0)
    out_image = unloader(out_image)
    out_image.save('1.jpg')

    attributions, approximation_error = ig.attribute(input_img2, baseline2, target=1, return_convergence_delta=True)
    aggregated_img = torch.mul(input_img2, attributions)
    out_image = aggregated_img.squeeze(0)
    out_image = unloader(out_image)
    out_image.save('2.jpg')

