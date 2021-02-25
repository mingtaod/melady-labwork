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
import torchnet as tnt


def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
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
        top5.update(acc5[0], train_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % printing_freq == 0:
            progress.display(i)


def validate(model, val_loader, criterion):
    batch_time = AverageMeter('Time', '6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (val_data, target) in enumerate(val_loader):
            val_data = val_data.to(device)
            target = target.to(device)

            output = model(val_data)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), val_data.size(0))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            batch_time.update(time.time() - end, target.size(0))
            end = time.time()

            # if i % printing_freq == 0:
            #     progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

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
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(epochs, model, train_loader, val_loader, test_loader, criterion, optimizer):
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(0, epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        acc1 = validate(model, val_loader, criterion)
        if acc1 > best_acc:
            best_acc = acc1
            best_epoch = epoch+1

    print('Best validation is from epoch ', best_epoch, ', with accuracy of ', best_acc)
    final_acc = validate(model, test_loader, criterion)
    print('Final test set result: ', final_acc)


# class MapSubset(torch.utils.data.dataset.Subset):
#     """
#     Given a dataset, creates a dataset which applies a mapping function
#     to its items (lazily, only when an item is called).
#
#     Note that data is not cloned/copied from the initial dataset.
#     """
#
#     def __init__(self, subset, map_fn):
#         self.subset = subset
#         self.map = map_fn
#
#     def __getitem__(self, index):
#         return self.map(self.subset[index])
#
#     def __len__(self):
#         return len(self.subset)


if __name__ == '__main__':
    printing_freq = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "D:/natural_images_dataset/natural_images"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Transformations defined here
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

    # Some hyperparameters
    batch_size = 50
    num_workers = 2
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 128

    # Splitting dataset into three sets
    model_dataset = td.datasets.WrapDataset(datasets.ImageFolder(data_dir))

    dataset_size = len(model_dataset)
    train_count = int(np.floor(0.8 * dataset_size))
    data_left = dataset_size - train_count
    test_count = int(np.floor(0.5 * data_left))
    valid_count = data_left - test_count

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(model_dataset, (train_count, valid_count, test_count))

    # Wrapping the three Subset objects with the dataset wrapper in torchdata package
    train_dataset = td.datasets.WrapDataset(train_dataset)
    valid_dataset = td.datasets.WrapDataset(valid_dataset)
    test_dataset = td.datasets.WrapDataset(test_dataset)

    print(train_dataset[0][0])

    # index = 0
    # for img, id in train_dataset:
    #     train_dataset[index] = (train_transform(img), id)
    #     index += 1

    # train_dataset = train_transform(train_dataset)
    # print(train_dataset[0][0])

    # train_dataset.map(train_transform)
    # valid_dataset.map(valid_transform)
    # test_dataset.map(valid_transform)

    train_dataset = tnt.dataset.TransformDataset(train_dataset, train_transform)
    print(train_dataset.get(1))

    # Creating dataset loaders from the three subsets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Constructing the model & setting up
    model = models.__dict__["resnet50"](pretrained=True)
    print('Designated device: ', device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)

    main(epochs, model, train_loader, valid_loader, test_loader, criterion, optimizer)


