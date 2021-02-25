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


def plot_img_IG(image,
                attr,
                cmap=None,
                overlay_alpha=0.4):

    # Here take the absolute value of the attributions and sum them along the color channel.
    # Only want to visualize the pixels that influence the decision making of the network (do not distinguish between negative influence and positive influence).
    # This is the reason why we want to accumulate the attributions along three channels into a single value.

    # Meaning of negative attribution: if a pixel has negative attribution score, this means that if we get rid of this single pixel, the probability that the network can classify this image correctly will increase;
    # Meaning of positive attributions: if removed, the probability that the network correctly classifying the image will decrease

    attribution_mask = torch.sum(attr.abs(), dim=0)
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(attribution_mask.float(), cmap=cmap)
    axs[0, 0].axis('off')
    axs[0, 1].set_title('Overlay IG on Input image ')
    axs[0, 1].imshow(attribution_mask.float(), cmap=cmap)
    axs[0, 1].imshow(image.permute(1, 2, 0).float(), alpha=overlay_alpha)
    axs[0, 1].axis('off')
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    # TODO:
    # 日期 2/25/2021
    # 下一步: We aggregate integrated gradients along the color channel and overlay them on
    # actual image in gray scale with positive attribtutions along the green channel and
    # negative attributions along the red channel.

    # 若attribution score是negative，则把红色channel保留，其他两个channel全都变成黑色0
    # 若attribution score是positive，则把绿色保留

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

    # img = Image.open('D:\\burn_level_images_dataset\\Combined\\1_RfootLfoot_PBD1.JPG')
    img = Image.open('D:\\burn_level_images_dataset\\Combined\\41_Lhandpalmar_PBD4.JPG')
    input_img = def_transform(img)
    input_img = input_img.unsqueeze(0)
    input_img = torch.tensor(input_img, dtype=torch.float32)
    classification = model(input_img)
    out_prob = torch.nn.functional.softmax(classification, dim=1).tolist()
    print('Classification result is: ', out_prob)

    # Question: what should be the baseline in this case? I suppose it should be the images of people's body without burning damage
    baseline = torch.zeros([3, 224, 224], dtype=torch.float32)
    baseline = baseline.unsqueeze(0)
    unloader = transforms.ToPILImage()

    attributions, approximation_error = ig.attribute(input_img, baseline, target=1, return_convergence_delta=True)

    print('Attribution shape: ', attributions.shape)

    _ = plot_img_IG(def_not_normalized_transform(img),
                attributions.squeeze(0),
                cmap=None,
                overlay_alpha=0.3)

    np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
    print('Attributions: ', np.array(attributions.squeeze(0)))

    print("max: ", attributions.squeeze(0).max())

    for i in range(attributions.squeeze(0).shape[0]):
        for j in range(attributions.squeeze(0).shape[1]):
            for k in range(attributions.squeeze(0).shape[2]):
                if attributions.squeeze(0)[i][j][k] > 0.1:
                    print("YES: ", attributions.squeeze(0)[i][j][k])

    attributions_copy = copy.deepcopy(attributions.squeeze(0))
    attributions_copy = unloader(attributions_copy)
    attributions_copy.save('attributions_hand.jpg')

    aggregated_img = torch.mul(input_img, attributions).squeeze()
    aggregated_img = np.array(unloader(aggregated_img))

    input_img = def_not_normalized_transform(img)
    input_img = np.array(unloader(input_img.squeeze()))

    threshold = 15

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
    input_img_copy.save('burn_test3_imposed_threshold_' + str(threshold) + '.jpg')

    out_image = unloader(aggregated_img)
    out_image.save('burn_test3.jpg')

    input_img = Image.fromarray(input_img, mode='RGB')
    input_img.save('cropped_input_img3.jpg')
