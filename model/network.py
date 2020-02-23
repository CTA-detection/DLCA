import torch
from torch import nn
from model.layers import *
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

config = {}
config['anchors'] = [4, 6, 10, 30, 60]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 0. 
config['sizelim2'] = 3.
config['sizelim3'] = 8.
config['aug_scale'] = False
config['r_rand_crop'] = 0.3
config['pad_value'] = 0
config["margin"] = 16
config["split_size"] = 144

config['augtype'] = {'flip':True,'swap':False,'scale':False,'rotate':False}

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=[3, 3, 3], stride=3)
        self.pool3 = nn.MaxPool3d(kernel_size=[5, 5, 5], stride=5)
        self.pool4 = nn.MaxPool3d(kernel_size=[6, 6, 6], stride=6)

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w, z = x.size(1), x.size(2), x.size(3), x.size(4)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w, z), mode='trilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w, z), mode='trilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w, z), mode='trilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w, z), mode='trilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
            
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))


        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(68, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))
        self.dblock = DACblock(64)
        self.spp = SPPblock(64)


    def forward(self, x, coord):
        
        out = self.preBlock(x)
        out_pool,indices0 = self.maxpool1(out)

        out1 = self.forw1(out_pool)
        out1_pool,indices1 = self.maxpool2(out1)

        out2 = self.forw2(out1_pool)
        out2_pool,indices2 = self.maxpool3(out2)

        out3 = self.forw3(out2_pool)
        out3_pool,indices3 = self.maxpool4(out3)

        out4 = self.forw4(out3_pool)

        center4 = self.dblock(out4)
        center4 = self.spp(center4)

        rev3 = self.path1(center4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2,coord), 1))

        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return out


def get_model():
    net = Net()
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
