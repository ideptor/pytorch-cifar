'''Train Resnet18 with PyTorch.'''
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder

# import torchvision
import torchvision.transforms as transforms

import os
import argparse

from time import time

from models import *
from tqdm import tqdm
from utils import progress_bar



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # pbar = tqdm(enumerate(trainloader))

    tot_start = time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
    # for batch_idx, (inputs, targets) in pbar:
        step_start = time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # pbar.set_postfix(loss=(train_loss/(batch_idx+1)), acc=(100.*correct/total), hit=f"{correct}/{total}")
    end = time()

    log = (
        f"[{epoch+1}/{end_epoch}][TRAIN {batch_idx+1:02d}/65] "
        f"Step: {int(end-step_start):d}s{int(((end-step_start)-int(end-step_start))*1000):03d}ms | "
        f"Tot: {int((end-tot_start)/60):02d}m{int(end-tot_start)%60:02d}s | "
        f"Loss: {(train_loss/(batch_idx+1)):.6f} | Acc: {(100.*correct/total):.2f}% | Hit: {correct}/{total}"
    )
    # print(log)
    logfile_writer.write(log+"\n")
    logfile_writer.flush()
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # pbar = enumerate(testloader)
        tot_start = time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            step_start = time()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best: %.3f%% '
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc))
            # pbar.set_postfix(loss=(test_loss/(batch_idx+1)), acc=(100.*correct/total), hit=f"{correct}/{total}")
            # print(f"{batch_idx}/4, loss={(test_loss/(batch_idx+1)):.5f}, acc={(100.*correct/total)}({correct}/{total})")
        end = time()
        log = (
            f"[{epoch+1}/{end_epoch}][TEST {batch_idx+1:d}/4] "
            f"Step: {int(end-step_start):d}s{int(((end-step_start)-int(end-step_start))*1000):03d}ms | "
            f"Tot: {int((end-tot_start)/60):02d}m{int(end-tot_start)%60:02d}s | "
            f"Loss: {(test_loss/(batch_idx+1)):.6f} | Acc: {(100.*correct/total):.2f}% | Hit: {correct}/{total}"
        )
        # print(log)
        logfile_writer.write(log+"\n")
        logfile_writer.flush()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def get_args() -> dict:
    parser = argparse.ArgumentParser(description='PyTorch Resnet18 Training')
    parser.add_argument('--epoch', '-e', default=15, type=int, help='epochs')
    parser.add_argument('--size', '-s', default=128, type=int, help='image size')
    parser.add_argument('--batch-size', '-b', default=32, type=int, help='batch size')
    parser.add_argument('--dataset', '-d', type=str, default="dataset", help='dataset folder')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument("--gpu", "-g", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    os.makedirs("logs", exist_ok=True)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomRotation(90),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(
    #     root='./data', train=True, download=True, transform=transform_train)
    trainset = ImageFolder(f"{args.dataset}/train", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    testset = ImageFolder(f"{args.dataset}/val", transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #         'dog', 'frog', 'horse', 'ship', 'truck')
    classes = trainset.classes

    # Model
    print('==> Building model..')
    net = ResNet18(len(classes), args.size)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    logfile_writer = open(
        f"logs/{datetime.now().strftime('trainlog_%Y%m%d_%H%M%S')}{'_resume' if args.resume else ''}.txt", "w")
    # logfile_writer.write(
    #     json.dumps(
    #         dict(
    #             learning_rate = args.lr,
    #             resume=args.resume,
    #         )
    #     )+"\n"
    # )
    logfile_writer.write(
        json.dumps(vars(args))
    )
    end_epoch = start_epoch+args.epoch
    for epoch in range(start_epoch, end_epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

    logfile_writer.close()
