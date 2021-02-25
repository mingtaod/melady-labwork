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
from sklearn import metrics


def train(model, train_loader, criterion, optimizer, epoch, lists, val_loader):
    # max_val_acc = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    model.train()

    for i, (train_data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_data = train_data.to(device)
        target = target.to(device)

        output = model(train_data)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), train_data.size(0))
        top1.update(acc1[0], train_data.size(0))
        lists['loss_train'].append(loss.item())
        lists['acc_train'].append(acc1[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % printing_freq == 0:
            progress.display(i)
        #     val_acc, _ = validate(model, val_loader, criterion, lists, True)
        #     max_val_acc = max(max_val_acc, val_acc)
        # else:
        #     val_acc, _ = validate(model, val_loader, criterion, lists, False)
        #     max_val_acc = max(max_val_acc, val_acc)

    # return max_val_acc


def validate(model, val_loader, criterion, lists, whether_print):
    batch_time = AverageMeter('Time', '6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    y_true = []
    y_score = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (val_data, target) in enumerate(val_loader):
            val_data = val_data.to(device)
            target = target.to(device)

            output = model(val_data)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            losses.update(loss.item(), val_data.size(0))
            top1.update(acc1[0], target.size(0))
            batch_time.update(time.time() - end, target.size(0))
            end = time.time()

            # print(type(target))
            y_true.extend(target.tolist())
            # print('target: ', target.tolist())
            scores, pred_labels = output.topk(1, 1, True, True)
            y_score.extend(scores.tolist())
            print('scores: ', scores.tolist())
            # print('pred_labels: ', pred_labels)

        if whether_print:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
            plot_roc(fpr, tpr, thresholds)

        auc_roc = metrics.roc_auc_score(y_true, y_score)
        auc_roc_float = round(auc_roc.item(), 3)

        print('top1.avg: ', top1.avg)
        print('losses.sum: ', losses.sum)
        # print(' * Validation Acc@1 {top1.avg:.3f} Validation Loss {losses.sum:.4e} '.format(top1=top1, losses=losses)) 咋整
        print(' * AUC_ROC ', auc_roc_float, '\n')

        lists['acc_val'].append(top1.avg)
        lists['loss_val'].append(losses.sum)

    return top1.avg, losses.sum, auc_roc_float


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


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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
        plt.text(x, y+0.02, '%.0f' % threshold_input[index], ha='center', va='bottom', fontsize=8)
        index += 1
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


def main(epochs, model, train_loader, val_loader, criterion, optimizer):
    best_acc = 0.0
    best_auc_roc = 0.0
    lst_loss_train = []
    lst_acc_train = []
    lst_loss_val = []
    lst_acc_val = []
    lists = {'loss_train': lst_loss_train,
             'acc_train': lst_acc_train,
             'loss_val': lst_loss_val,
             'acc_val': lst_acc_val}
    for epoch in range(0, epochs):
        # best_acc = max(train(model, train_loader, criterion, optimizer, epoch, lists, val_loader), best_acc)
        train(model, train_loader, criterion, optimizer, epoch, lists, val_loader)
        if epoch == epochs-1:
            curr_acc, curr_loss, auc_roc = validate(model, val_loader, criterion, lists, True)
        else:
            curr_acc, curr_loss, auc_roc = validate(model, val_loader, criterion, lists, False)
        best_acc = max(curr_acc, best_acc)
        best_auc_roc = max(auc_roc, best_auc_roc)

        # validate(...) 改成在这里进行验证，一个epoch验证一次，然后在最后一个epoch处进行验证，并且把PRC和AOC的面积求出来
        # 不需要几次迭代就验证一次，太浪费时间

    plot_acc(lists['acc_train'], None, 'train_acc_plot')
    plot_losses(lists['loss_train'], 'train_loss_plot')
    plot_acc(lists['acc_val'], None, 'valid_acc_plot')
    plot_losses(lists['loss_val'], 'valid_loss_plot')
    print('Best validation accuracy = ', best_acc)
    print('Best auc_roc = ', best_auc_roc)


def shuffle_label_file(file_name, shuffled_file_name=''):
    # out = open(shuffled_file_name, 'w')
    lines = []
    with open(file_name, 'r') as infile:
        for line in infile:
            lines.append(line)
    random.shuffle(lines)
    # for line in lines:
    #     out.write(line)
    return lines


def write_files(valid_file, train_file, labels, k, i):
    fold_size = len(labels) // k
    # if i == k-1:
    #     fold_size = len(labels) - (len(labels) // k) * i
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


if __name__ == '__main__':

    # Define hyperparameters here
    num_folds = 5
    batch_size = 50
    num_workers = 2
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 30
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
        transforms.RandomResizedCrop(224),
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

    # K-folds main loop
    for i in range(num_folds):
        model = models.__dict__["resnet50"](pretrained=True)
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
# 1. 把main.py给恢复到原状，跑一下main.py，检查一下output是不是也和这个文件一样，是大于一的
# 2. 把precision-recall curve部分的代码加进去
# 3. 检查是否模型真的学到了东西
