import torch
import torch.nn.functional as F
from torch import nn
from module.convGRU import ConvGRU
from module.convLSTM import ConvLSTM
from matplotlib import pyplot as plt
from module.se_layer import SELayer
from resnext.resnext101 import ResNeXt101
from resnext.resnext50 import ResNeXt50
from resnext.resnet50 import ResNet50
from resnext.resnet101 import ResNet101
from functools import partial


class R3Net(nn.Module):
    def __init__(self, motion='GRU', se_layer=False, dilation=True, basic_model='resnext50'):
        super(R3Net, self).__init__()

        self.motion = motion
        self.se_layer = se_layer
        self.dilation = dilation
        if basic_model == 'resnext50':
            resnext = ResNeXt50()
        elif basic_model == 'resnext101':
            resnext = ResNeXt101()
        elif basic_model == 'resnet50':
            resnext = ResNet50()
        else:
            resnext = ResNet101()
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
            self.reduce_low_GRU = ConvGRU(input_size=(119, 119), input_dim=256,
                                     hidden_dim=256,
                                     kernel_size=(3, 3),
                                     num_layers=1,
                                     batch_first=True,
                                     bias=True,
                                     return_all_layers=False)

            self.reduce_high_GRU = ConvGRU(input_size=(119, 119), input_dim=256,
                                          hidden_dim=256,
                                          kernel_size=(3, 3),
                                          num_layers=1,
                                          batch_first=True,
                                          bias=True,
                                          return_all_layers=False)
            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        elif self.motion == 'LSTM':
            # self.reduce_low_GRU = ConvLSTM(input_size=(119, 119), input_dim=256,
            #                               hidden_dim=256,
            #                               kernel_size=(3, 3),
            #                               num_layers=1,
            #                               padding=1,
            #                               dilation=1,
            #                               batch_first=True,
            #                               bias=True,
            #                               return_all_layers=False)

            self.reduce_high_GRU = ConvLSTM(input_size=(119, 119), input_dim=256,
                                           hidden_dim=256,
                                           kernel_size=(3, 3),
                                           num_layers=1,
                                           padding=1,
                                           dilation=1,
                                           batch_first=True,
                                           bias=True,
                                           return_all_layers=False)
            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        if self.se_layer:
            self.reduce_high_se = SELayer(256)
            self.reduce_low_se = SELayer(256)
            # self.motion_se = SELayer(32)

        if dilation:
            resnext.layer3.apply(partial(self._nostride_dilate, dilate=2))
            resnext.layer4.apply(partial(self._nostride_dilate, dilate=4))

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
        self.predict6 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

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

        if self.se_layer:
            reduce_low = self.reduce_low_se(reduce_low)
            reduce_high = self.reduce_high_se(reduce_high)

        if len(self.motion) > 0:
            low_side, low_state = self.reduce_low_GRU(reduce_low.unsqueeze(0))
            reduce_low = low_side[0].squeeze(0)
            high_side, high_state = self.reduce_high_GRU(reduce_high.unsqueeze(0))
            high_motion = high_side[0].squeeze(0)
            reduce_high = high_side[0].squeeze(0)
            motion_predict = self.motion_predict(high_motion)
            predict0 = self.predict0(reduce_high) + motion_predict
        else:
            predict0 = self.predict0(reduce_high)

        # predict0 = self.predict0(reduce_high)
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) + predict0
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) + predict1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) + predict2
        predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) + predict3
        predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) + predict4
        predict6 = self.predict6(torch.cat((predict5, reduce_high), 1)) + predict5

        # pre = predict6.data.cpu().numpy()
        # plt.rcParams.update({'font.size': 22})
        #
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # im = axes.flat[0].imshow(pre[-2, 0, :, :], vmin=-20, vmax=5, cmap='jet')
        # im = axes.flat[1].imshow(pre[-1, 0, :, :], vmin=-20, vmax=5, cmap='jet')

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes.flat[1])
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cbar = fig.colorbar(im, ax=axes.ravel().tolist())

        # plt.imshow(pre[-2, 0, :, :], vmin=-20, vmax=5, cmap='jet')
        # plt.subplot(1, 2, 2)
        # plt.imshow(pre[-1, 0, :, :], vmin=-20, vmax=5, cmap='jet')
        # # plt.grid(True)
        # plt.colorbar()
        # plt.show()

        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5 = F.upsample(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict6 = F.upsample(predict6, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict0, predict1, predict2, reduce_high, reduce_low
        return F.sigmoid(predict6)


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
