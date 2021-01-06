import time
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
from sklearn.linear_model import LinearRegression


def train(model, train_loader, criterion, optimizer, epoch, lists, val_loader):
    losses = AverageMeter('Loss', ':.4e')

    total_record = 0
    correct_record = 0
    model.train()

    for i, (train_data, target) in enumerate(train_loader):
        train_data = train_data.to(device)
        target = target.to(device)

        output = model(train_data)
        loss = criterion(output, target)

        num_correct, num_total = count_correct_total(output, target)
        correct_record += num_correct
        total_record += num_total

        losses.update(loss.item(), train_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % printing_freq == 0:
            # progress.display(i)
        #     val_acc, _ = validate(model, val_loader, criterion, lists, True)
        #     max_val_acc = max(max_val_acc, val_acc)
        # else:
        #     val_acc, _ = validate(model, val_loader, criterion, lists, False)
        #     max_val_acc = max(max_val_acc, val_acc)

    accuracy_curr = correct_record / total_record
    lists['loss_train'].append(losses.sum)
    lists['acc_train'].append(accuracy_curr)
    print(epoch, '-th epoch     ', 'train loss sum: ', losses.sum, '   train acc avg: ', accuracy_curr)


def validate(model, val_loader, criterion, lists, whether_print):
    losses = AverageMeter('Loss', ':.4e')
    y_true = []
    y_score = []
    total_record = 0
    correct_record = 0

    model.eval()

    with torch.no_grad():
        for i, (val_data, target) in enumerate(val_loader):
            val_data = val_data.to(device)
            target = target.to(device)

            output = model(val_data)
            loss = criterion(output, target)
            # ********
            # 通过output shape来判断应该在哪个维度操作，如果shape的dim1数值是一行中元素的个数，那么dim=1就代表我们会在每行进行操作
            # print(output.shape)
            out_prob = torch.nn.functional.softmax(output, dim=1).tolist()
            positive_scores = [row[1] for row in out_prob]

            num_correct, num_total = count_correct_total(output, target)
            correct_record += num_correct
            total_record += num_total

            losses.update(loss.item(), val_data.size(0))

            y_true.extend(target.tolist())
            y_score.extend(positive_scores)

        if whether_print:
            fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, y_score, pos_label=1)
            plot_roc(fpr, tpr, thresholds_roc)
            precision, recall, thresholds_prc = metrics.precision_recall_curve(y_true, y_score, pos_label=1)
            plot_prc(precision, recall, thresholds_prc)

        auc_roc = metrics.roc_auc_score(y_true, y_score)
        auc_roc_float = round(auc_roc.item(), 3)
        auc_prc = metrics.average_precision_score(y_true, y_score)
        auc_prc_float = round(auc_prc.item(), 3)

        accuracy_curr = correct_record / total_record

        print('                ', 'valid loss sum: ', losses.sum, '   valid acc avg: ', accuracy_curr)
        print('                * AUC_ROC: ', auc_roc_float, '   AUC_PRC: ', auc_prc_float, '\n')
        lists['acc_val'].append(accuracy_curr)
        lists['loss_val'].append(losses.sum)
    return accuracy_curr, losses.sum, auc_roc_float, auc_prc_float


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_losses(lst_loss, title):
    plt.plot(lst_loss, '-r', label='loss')
    plt.xlabel('nth iteration')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()


def plot_acc(lst_acc_1, lst_acc_5, title):
    plt.plot(lst_acc_1, '-b', label='val accuracy@1')
    if lst_acc_5 is not None:
        plt.plot(lst_acc_5, '-r', label='val accuracy@5')
    plt.xlabel('nth iteration')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()


def plot_roc(fpr_input, tpr_input, threshold_input):
    plt.plot(fpr_input, tpr_input)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    index = 0
    for x, y in zip(fpr_input, tpr_input):
        plt.text(x, y+0.02, '%.2f' % threshold_input[index], ha='center', va='bottom', fontsize=8)
        index += 1
    plt.show()


# Todo: 修改这个函数，enable text
def plot_prc(precision, recall, threshold_input):
    # print(precision.shape, "---precision")
    # print(recall.shape, "---recall")
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    index = 0
    # for x, y in zip(recall, precision):
    #     plt.text(x, y+0.02, '%.2f' % threshold_input[index], ha='center', va='bottom', fontsize=8)
    #     index += 1
    plt.show()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_num = target.size(0)

        # Note: view(r,c)-changes a tensor into dimension (r,c)
        #       expand_as(t)-changes the subject into the dimension of parameter t's dimension (t is a tensor as well)
        #       topk(input, k, dim=None, largest=True, sorted=True, out=None)-return the k max values of input along the specified dimension
        #       t()-returns the transpose of the subject tensor

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_num))
        return res


def count_correct_total(output, target):
    with torch.no_grad():
        # print('output: ', output)
        # print('target: ', target)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print('correct: ', correct)
        correct_part_count = 0
        for bool_val in correct.tolist()[0]:
            if bool_val:
                correct_part_count += 1
        # print('correct_count: ', correct_part_count)
        # print('\n')
        return correct_part_count, len(target)


def main(epochs, model, train_loader, val_loader, criterion, optimizer):
    best_acc = 0.0
    best_auc_roc = 0.0
    best_auc_prc = 0.0
    lst_loss_train = []
    lst_acc_train = []
    lst_loss_val = []
    lst_acc_val = []
    lists = {'loss_train': lst_loss_train,
             'acc_train': lst_acc_train,
             'loss_val': lst_loss_val,
             'acc_val': lst_acc_val}
    for epoch in range(0, epochs):
        train(model, train_loader, criterion, optimizer, epoch, lists, val_loader)
        if epoch == epochs-1:
            curr_acc, curr_loss, auc_roc, auc_prc = validate(model, val_loader, criterion, lists, True)
        else:
            curr_acc, curr_loss, auc_roc, auc_prc = validate(model, val_loader, criterion, lists, False)
        best_acc = max(curr_acc, best_acc)
        best_auc_roc = max(auc_roc, best_auc_roc)
        best_auc_prc = max(auc_prc, best_auc_prc)

    plot_acc(lists['acc_train'], None, 'train_acc_plot')
    plot_losses(lists['loss_train'], 'train_loss_plot')
    plot_acc(lists['acc_val'], None, 'valid_acc_plot')
    plot_losses(lists['loss_val'], 'valid_loss_plot')

    print('Best validation accuracy = ', best_acc)
    print('Best auc_roc = ', best_auc_roc)
    print('Best auc_prc = ', best_auc_prc)
    print('\n')


def shuffle_label_file(file_name):
    lines = []
    with open(file_name, 'r') as infile:
        for line in infile:
            lines.append(line)
    random.shuffle(lines)
    return lines


def write_files(valid_file, train_file, labels, k, i):
    fold_size = len(labels) // k
    train_labels = []
    valid_labels = []
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        part = labels[idx]
        if j == i:
            valid_labels = part
        elif train_labels is []:
            train_labels = part
        else:
            train_labels.extend(part)
    print(train_labels)
    print(valid_labels)
    out_train = open(train_file, 'w')
    out_valid = open(valid_file, 'w')
    for train_label in train_labels:
        out_train.write(train_label)
    for valid_label in valid_labels:
        out_valid.write(valid_label)


# Define the default data loader here
def default_loader(path):
    img_pil = Image.open(path).convert('RGB')
    return img_pil


class BurnDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.normpath("%s\%s" % (img_path, line.strip('\n').split(' ')[0])) for line in lines]
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
        # num_out_features = original_model.fc.out_features
        # print('num_out_features', num_out_features)
        fc1 = nn.Linear(2048, 1000)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(1000, 256)
        relu2 = nn.ReLU()
        fc3 = nn.Linear(256, num_classes)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(fc1, relu1, fc2, relu2, fc3)

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':

    # Define hyperparameters here
    num_folds = 5
    batch_size = 2
    num_workers = 2
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 65
    printing_freq = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defining directories
    dataset_dir = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "Combined"))
    valid_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "valid.txt"))
    train_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "train.txt"))
    label_txt = os.path.normpath("%s\%s\%s" % ("D:", "burn_level_images_dataset", "right_img_label.txt"))

    # Writing into the txt files
    shuffled_labels = shuffle_label_file(label_txt)
    print(len(shuffled_labels))

    # Apply different transformations on training and validating data sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # K-folds main loop
    for i in range(num_folds):
        orig_model = models.__dict__["resnet50"](pretrained=True)
        model = FineTuneResNet(orig_model, 2)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)  # Consider only changing the weights on the last layer
        write_files(valid_txt, train_txt, shuffled_labels, num_folds, i)

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

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print('-------------------K-fold: ', i, '-th iteration starts below-------------------')
        main(epochs, model, train_loader, valid_loader, criterion, optimizer)





# 下一步：1.可以计算出train中每次iteration的id，然后存储到一个list里面，plot出来更完整
#        2. 可以尝试只改变model最后一层的parameters

# 目前问题：1. 路径问题：应该怎么表示绝对路径
#          2. AOC和PRC
#          3. validation的频率应该是啥

# TODO:
# 1. 把main.py给恢复到原状，跑一下main.py，检查一下output是不是也和这个文件一样，是大于1的
# 2. 把precision-recall curve部分的代码加进去
# 3. 检查是否模型真的学到了东西
# 4. Fine tune resnet50 *
# 5. 怎么把probability搞出来... 最坏的办法 考虑softmax


# Note:
# loss最好要除以instance的数量，如果在.5以上，则说明模型学到了东西
