import os

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Define the default data loader here
def default_loader(path):
    img_pil = Image.open(path).convert('RGB')
    return img_pil


class BurnDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.normpath("%s/%s" % (img_path, line.strip('\n').split(' ')[0])) for line in lines]
            self.img_label = [int(line.strip('\n').split(' ')[-1]) for line in lines]
        self.dataset = dataset
        self.data_transforms = data_transforms
        self.loader = loader
        # print('img[0] name: ', self.img_name[0], ' label[0]: ', self.img_label[0])
        # print('img[-1] name: ', self.img_name[-1], ' label[-1]: ', self.img_label[-1])

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        # print(img_name)
        label = self.img_label[item]
        img = self.loader(img_name)
        # loader equals to the default transformation set which normalizes and turns the img into a tensor
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except Exception as e:
                print('Cannot transform image: {}'.format(img_name))
                print(repr(e))
        # print(img)
        return img, label


class FineTuneResNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneResNet, self).__init__()
        fc1 = nn.Linear(2048, 1000)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(1000, 256)
        relu2 = nn.ReLU()
        # fc3 = nn.Linear(1000, num_classes)
        fc3 = nn.Linear(256, num_classes)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(fc1, relu1, fc2, relu2, fc3)
        # self.classifier = nn.Sequential(fc1, relu1, fc3)

    def forward(self, x):
        out = self.features(x)
        # print('prev out.shape: ', out.shape)
        out = out.view(out.size(0), -1)
        # print('out.shape: ', out.shape)
        out = self.classifier(out)
        return out


def predict_partition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weight = torch.load("/Users/anon/PycharmProjects/BurnCNN/burnCNN_fold4.pt", map_location=device)
    orig_model = models.__dict__["resnet50"](pretrained=True)
    model = FineTuneResNet(orig_model, 2)

    model.load_state_dict(torch.load("/Users/anon/PycharmProjects/BurnCNN/burnCNN_fold4.pt", map_location=device))
    model.eval()

    baseDir = "/Users/anon/PycharmProjects/BurnCNN/"
    dataset = "burn_level_images_dataset"
    dataset_dir = os.path.normpath("%s/%s/%s" % (baseDir, dataset, "Combined"))
    valid_txt = os.path.normpath("%s/%s/%s" % (baseDir, dataset, "valid.txt"))
    train_txt = os.path.normpath("%s/%s/%s" % (baseDir, dataset, "train.txt"))
    label_txt = os.path.normpath("%s/%s/%s" % (baseDir, dataset, "right_img_label.txt"))

    # Apply different transformations on training and validating data sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = BurnDataset(img_path=dataset_dir,
                                txt_path=train_txt,
                                dataset='train',
                                data_transforms=train_transform,
                                loader=default_loader)

    valid_dataset = BurnDataset(img_path=dataset_dir,
                                txt_path=valid_txt,
                                dataset='valid',
                                data_transforms=valid_transform,
                                loader=default_loader)
    import numpy as np

    print('train dataset')
    for i in range(train_dataset.__len__()):
        getitem__ = train_dataset.__getitem__(i)
        x ,y = getitem__[0] ,getitem__[1]
        predict = model(x.reshape(1, 3, 224, 224))

        argmax = np.argmax(predict.detach().numpy())

        if argmax == y:
            print(train_dataset.img_name[i].split('/')[7])

    print('valid dataset')
    for i in range(valid_dataset.__len__()):
        getitem__ = valid_dataset.__getitem__(i)
        x ,y = getitem__[0] ,getitem__[1]
        predict = model(x.reshape(1, 3, 224, 224))

        argmax = np.argmax(predict.detach().numpy())

        if argmax == y:
            print(valid_dataset.img_name[i].split('/')[7])

    print("done")


if __name__ == '__main__':
    predict_partition()
