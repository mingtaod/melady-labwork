import time
import torch
import torch.nn as nn
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchdata as td

#
# # 假设这是一个含有4个sample的batch运行出来的output，每一行都是没一个sample的output
# output = torch.tensor([[-5.4783, 0.2298, 0.1622, -9.2863, -0.8192, 4.1922, 7.2011],
#                            [-4.2573, -0.4794, 0.8293, -2.3922, 6.4868, 7.0002, 1.9827],
#                            [-0.1070, -5.1511, -5.4783, 9.1888, -10.9882, 3.2928, 0.1827],
#                            [-0.1785, -4.3339, -5.4783, 0.9987, -1.2822, 7.1727, 6.8881]])
#
# topk = (1, 5)
# maxk = max(topk)
# _, pred = output.topk(maxk, 1, True, True)
#
# print(_)
# print(pred)

data_dir = "D:/natural_images_dataset/natural_images"

# Wrap torchvision dataset with WrapDataset
model_dataset = td.datasets.WrapDataset(datasets.ImageFolder(data_dir))
dataset_size = len(model_dataset)
train_count = int(np.floor(0.8 * dataset_size))
data_left = dataset_size - train_count
test_count = int(np.floor(0.5 * data_left))
valid_count = data_left - test_count

# Split dataset
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset,
    (train_count, test_count, valid_count)
)

train_dataset = td.datasets.WrapDataset(train_dataset)

# Apply torchvision mappings ONLY to train dataset
train_dataset.map(
    td.maps.To(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    )
)
