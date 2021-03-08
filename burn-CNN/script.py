import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import random
import sklearn
from sklearn import metrics
from xlrd import xldate_as_tuple

from burn_resnet50 import FineTuneResNet, BurnDataset, default_loader
import pandas as pd
import xlrd
from xlrd import open_workbook

if __name__ == '__main__':
    # Get model with trained weights here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.__dict__["resnet50"](pretrained=True)
    model = FineTuneResNet(base_model, 2)
    model.load_state_dict(torch.load("D:\\torch-model-weights\\burnCNN_imagewise_weights\\burnCNN_fold4.pt"))

    # Create dataset for validation
    dataset_dir = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "Combined"))
    valid_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "valid.txt"))
    train_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "train.txt"))

    all_img_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "right_img_label.txt"))

    valid_right = os.path.normpath("%s\%s\%s\%s\%s" % ("C:", "Users", "dongm", "Desktop", "valid_right.txt"))
    train_right = os.path.normpath("%s\%s\%s\%s\%s" % ("C:", "Users", "dongm", "Desktop", "train_right.txt"))

    # Read excel
    excel_dir = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "ML Retrospective Data Collection Table_Dec15_2020.xlsx"))
    data = xlrd.open_workbook(excel_dir)
    table = data.sheet_by_name("Image Collection")
    nrows = table.nrows
    ncols = table.ncols

    # Read all train records
    records = [0, 0, 0, 0, 0]

    with open(train_txt) as input_file:
        lines = input_file.readlines()
        img_names = [line.strip('\n').split(' ')[0] for line in lines]
        img_labels = [int(line.strip('\n').split(' ')[-1]) for line in lines]

    with open_workbook(excel_dir) as workbook:
        for img_name in img_names:
            img_name = img_name.split('.')[0]
            # print(img_name)
            for i in range(nrows):
                for j in range(ncols):
                    curr_value = table.cell(i, j).value
                    # print(curr_value)
                    if img_name == curr_value:
                        category_col = j + 4
                        date_cell = table.cell(i, category_col).value
                        if table.cell_type(i, category_col) == 3:
                            date_cell = xldate_as_tuple(table.cell_value(i, category_col), workbook.datemode)
                            for item in date_cell:
                                if item > 0 and item < 5:
                                    records[item] += 1
                        elif table.cell_type(i, category_col) == 2:
                            records[int(date_cell)] += 1
                        else:
                            temp_list = date_cell.strip('\n').split(', ')
                            # print(img_name)
                            for item in temp_list:
                                if item == 'STSG':
                                    continue
                                elif item == '':
                                    continue
                                records[int(item)] += 1

    print(records)

    # Read all right records
    right_records = [0, 0, 0, 0, 0]

    with open(train_right) as input_file:
        lines = input_file.readlines()
        img_names = [line.strip('\n') for line in lines]

    print(img_names)

    with open_workbook(excel_dir) as workbook:
        for img_name in img_names:
            img_name = img_name.split('.')[0]
            # print(img_name)
            for i in range(nrows):
                for j in range(ncols):
                    curr_value = table.cell(i, j).value
                    # print(curr_value)
                    if img_name == curr_value:
                        category_col = j + 4
                        date_cell = table.cell(i, category_col).value
                        if table.cell_type(i, category_col) == 3:
                            date_cell = xldate_as_tuple(table.cell_value(i, category_col), workbook.datemode)
                            for item in date_cell:
                                if item > 0 and item < 5:
                                    right_records[item] += 1
                        elif table.cell_type(i, category_col) == 2:
                            right_records[int(date_cell)] += 1
                        else:
                            temp_list = date_cell.strip('\n').split(', ')
                            for item in temp_list:
                                if item == 'STSG':
                                    continue
                                elif item == '':
                                    continue
                                right_records[int(item)] += 1

    print(right_records)

    for i in range(1, 5):
        print("Train accuracy for burn depth category", i, "=", right_records[i]/records[i])
