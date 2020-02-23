import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils.log_utils import *
import sys
from utils.inference_utils import SplitComb
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='ca detection')
parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                    help='model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input', default='', type=str, metavar='data',
                    help='directory to save images (default: none)')
parser.add_argument('--output', default='', type=str, metavar='SAVE',
                    help='directory to save prediction results(default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    test_name = (args.input).split("/")[-1]
    data_dir = (args.input).split("/")[-2]
    save_dir = (args.output).split("/")[-2]

    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    
    margin = config["margin"]
    sidelen = config["split_size"]

    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
    dataset = data.TestDetector(
        data_dir,
        test_name,
        config,
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        collate_fn = data.collate,
        pin_memory=False)

    test(test_loader, net, get_pbb, save_dir, config)
    return


def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        data = data[0][0]
        coord = coord[0][0]
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()

            output = net(input,inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)

        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))


if __name__ == '__main__':
    main()
