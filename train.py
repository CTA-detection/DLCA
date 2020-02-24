import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils.log_utils import *
import sys

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='ca detection')
parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input', default='', type=str, metavar='SAVE',
                    help='directory to save train images (default: none)')
parser.add_argument('--output', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--gpu', default='2, 3', type=str, metavar='N',
                    help='use gpu')


def main():
    global args
    args = parser.parse_args()
    start_epoch = args.start_epoch
    data_dir = args.input   
    save_dir = args.output
    
    train_name = []
    for name in os.listdir(data_dir):
        if name.endswith("nii.gz"):
            name = name.split(".")[-3]
            train_name.append(name)

    torch.manual_seed(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)

    dataset = data.TrainDetector(
        data_dir,
        train_name,
        config)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= 80:#args.epochs * 0.8:
            lr = args.lr
        elif epoch <= 120:#args.epochs * 0.9:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    loss_total_l,loss_class_l,loss_regress_l,tpr_l,tnr_l = [],[],[],[],[]

    for epoch in range(start_epoch, args.epochs + 1):
        print("epoch",epoch)
        loss_total,loss_class,loss_regress,tpr,tnr = train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)

        loss_total_l.append(loss_total)
        loss_class_l.append(loss_class)
        loss_regress_l.append(loss_regress)
        tpr_l.append(tpr)
        tnr_l.append(tnr)
        plot(save_dir + 'train_curves.png',loss_total_l,loss_class_l,loss_regress_l,tpr_l,tnr_l)
        np.savez(save_dir + 'train_curves.npz',
                 loss_total=np.array(loss_total_l),
                 loss_class=np.array(loss_class_l),
                 loss_regress=np.array(loss_regress_l),
                 tpr=np.array(tpr_l),
                 tnr=np.array(tnr_l))


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):

        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        coord = Variable(coord.cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        print("loss:\033[1;35m{}\033[0m, class:{}, reg:{},".format(loss_output[0],loss_output[1],loss_output[2]))
        metrics.append(loss_output)
       
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    tpr=100.0*np.sum(metrics[:,6])/np.sum(metrics[:,7])
    tnr=100.0*np.sum(metrics[:,8])/np.sum(metrics[:,9])
    loss_total=np.mean(metrics[:,0])
    loss_class=np.mean(metrics[:,1])
    loss_regress=[np.mean(metrics[:,2]),np.mean(metrics[:,3]),np.mean(metrics[:,4]),np.mean(metrics[:,5])]

    print("metrics",metrics[:, 6])
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    return loss_total,loss_class,loss_regress,tpr,tnr


if __name__ == '__main__':
    main()
