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
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from PIL import Image
import tensorflow as tf


def train(model, train_loader, criterion, optimizer, epoch, lists):
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

        # print('len(train_data): ', len(train_data))

        output = model(train_data)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(output)
        losses.update(loss.item(), train_data.size(0))
        top1.update(acc1[0], train_data.size(0))
        top5.update(acc5[0], train_data.size(0))
        lists[0].append(loss.item())
        # print('lst_loss: ', lists[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % printing_freq == 0:
            progress.display(i)


def validate(model, val_loader, criterion, lists):
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
        lists[1].append(top1.avg)
        lists[2].append(top5.avg)
        print('lst_acc1: ', lists[1])
        print('lst_acc5: ', lists[2])

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


class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=8):
        super(FineTuneResnet50, self).__init__()
        self.num_class = num_class
        resnet50_instance = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50_instance.children())[:-1])
        self.classifier = nn.Linear(2048, self.num_class)

    def forward(self, x):
        output = self.features(x)
        # print('output.shape: ', output.shape)
        # 下面这行是为了
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output


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
    plt.plot(lst_acc_5, '-r', label='val accuracy@5')
    plt.xlabel('nth iteration')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(title + '.png')
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
        # print('pred before transpose: ', pred)
        # print('target before transform: ', target)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print('pred after transpose: ', pred)
        # print('target: ', target.view(1, -1).expand_as(pred))
        # print('correct: ', correct)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_num))
        return res


def main(epochs, model, train_loader, val_loader, test_loader, criterion, optimizer):
    best_acc = 0.0
    best_epoch = 0
    lst_loss = []
    lst_acc_1 = []
    lst_acc_2 = []
    lists = (lst_loss, lst_acc_1, lst_acc_2)
    for epoch in range(0, epochs):
        train(model, train_loader, criterion, optimizer, epoch, lists)
        acc1 = validate(model, val_loader, criterion, lists)
        if acc1 > best_acc:
            best_acc = acc1
            best_epoch = epoch+1

    plot_acc(lst_acc_1, lst_acc_2, 'acc_plot')
    plot_losses(lst_loss, 'loss_plot')

    print('Best validation is from epoch ', best_epoch, ', with accuracy of ', best_acc.item())
    final_acc = validate(model, test_loader, criterion, lists)
    print('Final test set result: ', final_acc)


if __name__ == '__main__':
    data_dir = "D:/natural_images_dataset/natural_images"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    full_dataset = datasets.ImageFolder(data_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize
    ]))

    print('full_dataset.classes: ', full_dataset.classes)
    print(full_dataset.class_to_idx['motorbike'])

    # print(full_dataset[0][0])
    # print(type(full_dataset[0]))  # trainloader里面的每一个元素都是一个tuple，第一个element是一个tensor，第二个是label的编号，这个可能和main——alt中的bug有关系

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_split = int(np.floor(0.8 * dataset_size))
    train_indices, other_indices = indices[:train_split], indices[train_split:]

    other_indices_size = len(other_indices)
    test_split = int(np.floor(0.5 * other_indices_size))
    test_indices, val_indices = other_indices[:test_split], other_indices[test_split:]

    # Defining samplers for future loader use
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    print(torch.__version__)

    printing_freq = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Some hyperparameters
    batch_size = 50
    num_workers = 2
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 2

    # Need to apply data augmentation on train set split ......
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers)  # Augmentation
    val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)

    # Constructing the model & setting up
    # model = models.__dict__["resnet50"](pretrained=True)
    model = FineTuneResnet50(num_class=8)
    print('designated device: ', device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)

    main(epochs, model, train_loader, val_loader, test_loader, criterion, optimizer)

    model.eval()
    ig = IntegratedGradients(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # img = Image.open('D:\\natural_images_dataset\\natural_images\\cat\\cat_0066.jpg')
    # img = def_transform(img)
    # input_img = torch.tensor(img)
    # input_img.view(1, 3, 224, 224)
    # print(input_img)
    # print(input_img.shape)

    img = Image.open(tf.gfile.Open('D:\\natural_images_dataset\\natural_images\\cat\\cat_0066.jpg', 'rb')).convert(
                'RGB').resize((2, 256, 256), Image.BILINEAR)
    input_img = def_transform(img)

    print(input_img)
    print(input_img.shape)

    baseline = np.zeros((3, 224, 224))
    baseline = torch.tensor(baseline)

    attributions, approximation_error = ig.attribute(input_img, baseline, target=2, return_convergence_delta=True)
    print('attributions: ', attributions)
    print('approx error: ', approximation_error)

    # torch.save(model.state_dict(), "D:\\torch-model-weights\\resnet50_8class.pt")
