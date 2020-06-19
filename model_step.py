import torch
import torch.nn.functional as F
from torch import nn
from module.convGRU import ConvGRU
from module.convLSTM import ConvLSTM
from matplotlib import pyplot as plt

from resnext.resnext101 import ResNeXt101


class R3Net(nn.Module):
    def __init__(self, motion='GRU'):
        super(R3Net, self).__init__()

        self.motion = motion

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )
        if self.motion == 'GRU':
            self.reduce_high_motion = ConvGRU(input_size=(119, 119), input_dim=256,
                                          hidden_dim=256,
                                          kernel_size=(3, 3),
                                          num_layers=1,
                                          batch_first=True,
                                          bias=True,
                                          return_all_layers=False)
            self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)
        elif self.motion == 'LSTM':
            self.reduce_high_motion = ConvLSTM(input_size=(119, 119), input_dim=256,
                                           hidden_dim=256,
                                           kernel_size=(3, 3),
                                           num_layers=1,
                                           padding=1,
                                           dilation=1,
                                           batch_first=True,
                                           bias=True,
                                           return_all_layers=False)
            self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.predict4 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.predict5 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # self.predict6 = nn.Sequential(
        #     nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]

        reduce_low = self.reduce_low(torch.cat((
            layer0,
            F.upsample(layer1, size=l0_size, mode='bilinear', align_corners=True),
            F.upsample(layer2, size=l0_size, mode='bilinear', align_corners=True)), 1))

        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.upsample(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))
        reduce_high = F.upsample(reduce_high, size=l0_size, mode='bilinear', align_corners=True)

        if len(self.motion) > 0:
            # low_side, low_state = self.reduce_low_GRU(reduce_low.unsqueeze(0))
            # reduce_low = low_side[0].squeeze(0)
            high_side, high_state = self.reduce_high_motion(reduce_high.unsqueeze(0))
            high_motion = high_side[0].squeeze(0)
            motion_predict = self.motion_predict(high_motion)
            # predict0 = self.predict0(reduce_high) + motion_predict

        predict0 = self.predict0(reduce_high)
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) + predict0
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) + predict1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) + predict2
        predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) + predict3
        predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) + predict4

        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5 = F.upsample(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)

        # motion_predict = F.upsample(motion_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict0, predict1, predict2, predict3, predict4, predict5
        return F.sigmoid(predict5)


class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = int(in_dim / 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
