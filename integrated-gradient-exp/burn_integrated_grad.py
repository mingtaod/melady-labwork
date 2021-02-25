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
import copy


class FineTuneResNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneResNet, self).__init__()
        fc1 = nn.Linear(2048, 1000)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(1000, 256)
        relu2 = nn.ReLU()
        fc3 = nn.Linear(256, num_classes)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(fc1, relu1, fc2, relu2, fc3)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':

    orig_model = models.__dict__["resnet50"](pretrained=True)
    model = FineTuneResNet(orig_model, 2)
    model.load_state_dict(torch.load("D:\\torch-model-weights\\burnCNN_patientwise_fold4.pt"))
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

    def_not_normalized_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    img = Image.open('D:\\burn_level_images_dataset\\Images 1\\41_Lhandpalmar_PBD4.JPG')
    input_img = def_transform(img)
    input_img = input_img.unsqueeze(0)
    input_img = torch.tensor(input_img, dtype=torch.float32)

    # Question: what should be the baseline in this case? I suppose it should be the images of people's body without burning damage
    baseline = torch.zeros([3, 224, 224], dtype=torch.float32)
    baseline = baseline.unsqueeze(0)
    unloader = transforms.ToPILImage()

    attributions, approximation_error = ig.attribute(input_img, baseline, target=1, return_convergence_delta=True)

    aggregated_img = torch.mul(input_img, attributions).squeeze()
    aggregated_img = np.array(unloader(aggregated_img))

    input_img = def_not_normalized_transform(img)
    input_img = np.array(unloader(input_img.squeeze()))

    threshold = 15

    for exp_num in range(16):
        input_img_copy = copy.deepcopy(input_img)
        for row in range(aggregated_img.shape[0]):
            for col in range(aggregated_img.shape[1]):
                max_channel_value = 0
                for channel in range(aggregated_img.shape[2]):
                    max_channel_value = max(max_channel_value, aggregated_img[row][col][channel])

                if max_channel_value > threshold:
                    for channel in range(aggregated_img.shape[2]):
                        input_img_copy[row][col][channel] = aggregated_img[row][col][channel]

        input_img_copy = Image.fromarray(input_img_copy, mode='RGB')
        input_img_copy.save('burn_test1_imposed_threshold_' + str(threshold) + '.jpg')
        threshold -= 1

    out_image = unloader(aggregated_img)
    out_image.save('burn_test1.jpg')

    input_img = Image.fromarray(input_img, mode='RGB')
    input_img.save('cropped_input_img.jpg')



    #
    # img = Image.open('D:\\burn_level_images_dataset\\Images 1\\17_Rchest_PBD1.JPG')
    # input_img = def_transform(img)
    # input_img = input_img.unsqueeze(0)
    # input_img = torch.tensor(input_img, dtype=torch.float32)
    # baseline = torch.zeros([3, 224, 224], dtype=torch.float32)
    # baseline = baseline.unsqueeze(0)
    # unloader = transforms.ToPILImage()
    # attributions, approximation_error = ig.attribute(input_img, baseline, target=1, return_convergence_delta=True)
    # aggregated_img = torch.mul(input_img, attributions)
    # out_image = aggregated_img.squeeze(0)
    # out_image = unloader(out_image)
    # out_image.save('burn_test2.jpg')
