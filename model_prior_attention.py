import torch
import torch.nn.functional as F
from torch import nn
from module.convGRU import ConvGRU
from module.convLSTM import ConvLSTM
from module.spa_module import PAM_Module, STA_Module
from matplotlib import pyplot as plt
from module.se_layer import SELayer
from module.attention import BaseOC_Context_Module
from resnext.resnext50 import ResNeXt50
from resnext.resnext101 import ResNeXt101
from resnext.resnet101 import ResNet101
from resnext.resnet50 import ResNet50

import numpy as np

class R3Net_prior(nn.Module):
    def __init__(self, motion='GRU', se_layer=False, attention=False, pre_attention=False, isTriplet=False,
                 basic_model='resnext50', sta=False, naive_fuse=False):
        super(R3Net_prior, self).__init__()

        self.motion = motion
        self.se_layer = se_layer
        self.attention = attention
        self.pre_attention = pre_attention
        self.isTriplet = isTriplet
        self.sta = sta
        self.naive_fuse = naive_fuse

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
            self.reduce_high_motion = ConvGRU(input_size=(119, 119), input_dim=256,
                                          hidden_dim=64,
                                          kernel_size=(3, 3),
                                          num_layers=1,
                                          batch_first=True,
                                          bias=True,
                                          return_all_layers=False)
            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        elif self.motion == 'LSTM':
            self.reduce_high_motion = ConvLSTM(input_size=(119, 119), input_dim=256,
                                           hidden_dim=64,
                                           kernel_size=(3, 3),
                                           num_layers=1,
                                           padding=1,
                                           dilation=1,
                                           batch_first=True,
                                           bias=True,
                                           return_all_layers=False)
        elif self.motion == 'no':
            self.reduce_high_motion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, kernel_size=1)
        )
            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        if self.se_layer:
            self.reduce_high_se = SELayer(256)
            # self.reduce_low_se = SELayer(256)
            self.motion_se = SELayer(32)

        if self.attention:
            self.reduce_atte = BaseOC_Context_Module(256, 256, 128, 128, 0.05, sizes=([2]))

        if self.pre_attention:
            self.pre_sals_attention2 = SELayer(2, 1)
            self.pre_sals_attention3 = SELayer(3, 1)
            self.pre_sals_attention4 = SELayer(4, 1)
            self.pre_sals_attention5 = SELayer(5, 1)

        if self.sta:
            self.sta_module = STA_Module(64)
            self.sp_down = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1), nn.PReLU()
            )

        if self.naive_fuse:
            self.sp_down = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1), nn.PReLU()
            )
            # self.sp_down2 = nn.Sequential(
            #     nn.Conv2d(128, 64, kernel_size=1), nn.PReLU()
            # )


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
        # self.predict4 = nn.Sequential(
        #     nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )
        # self.predict5 = nn.Sequential(
        #     nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )
        # self.predict6 = nn.Sequential(
        #     nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )

        self.predict1_motion = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.predict2_motion = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.predict3_motion = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.predict4_motion = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.predict5_motion = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

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

        if self.se_layer:
            # reduce_low = self.reduce_low_se(reduce_low)
            reduce_high = self.reduce_high_se(reduce_high)

        if self.attention:
            reduce_high = self.reduce_atte(reduce_high)

        if len(self.motion) > 0:
            # low_side, low_state = self.reduce_low_GRU(reduce_low.unsqueeze(0))
            # reduce_low = low_side[0].squeeze(0)
            if self.motion == 'no':
                high_motion = self.reduce_high_motion(reduce_high)
            else:
                high_side, high_state = self.reduce_high_motion(reduce_high.unsqueeze(0))
                high_motion = high_side[0].squeeze(0)
            if self.se_layer:
                high_motion = self.motion_se(high_motion)
            if self.sta:
                high_motion = F.upsample(high_motion, size=(70, 70), mode='bilinear', align_corners=True)
                reduce_high_down = F.upsample(reduce_high, size=(70, 70), mode='bilinear', align_corners=True)
                reduce_high_down = self.sp_down(reduce_high_down)
                # high_motion = self.sta_module(high_motion,
                #                               reduce_high_down.normal_(mean=float(high_motion.mean().data.cpu().numpy()),
                #                                                        std=float(high_motion.std().data.cpu().numpy())))
                high_motion = self.sta_module(high_motion, reduce_high_down)
                high_motion = F.upsample(high_motion, size=(119, 119), mode='bilinear', align_corners=True)

            if self.naive_fuse:
                high_motion = F.upsample(high_motion, size=(70, 70), mode='bilinear', align_corners=True)
                reduce_high_down = F.upsample(reduce_high, size=(70, 70), mode='bilinear', align_corners=True)
                reduce_high_down = self.sp_down(reduce_high_down)
                # high_motion = self.sp_down2(torch.cat([high_motion, reduce_high_down], dim=1))
                high_motion = high_motion * reduce_high_down
                high_motion = F.upsample(high_motion, size=(119, 119), mode='bilinear', align_corners=True)

        predict0 = self.predict0(reduce_high)
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) + predict0
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) + predict1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) + predict2
        # predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) + predict3
        # predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) + predict4
        # predict6 = self.predict6(torch.cat((predict5, reduce_high), 1)) + predict5

        first_sal = predict3.narrow(0, 0, 1)
        first_reduce_high = high_motion.narrow(0, 1, 5)
        first_motion = high_motion.split(1, dim=0)
        first_sal_guide = torch.cat([first_sal, first_sal, first_sal, first_sal, first_sal], dim=0)
        first_motion = torch.cat(first_motion[:-1], dim=0)
        predict1_motion = self.predict1_motion(torch.cat([first_sal_guide, first_reduce_high + first_motion], 1)) + predict3.narrow(0, 1, 5)

        second_sal = predict1_motion.narrow(0, 0, 1)
        second_reduce_high = high_motion.narrow(0, 2, 4)
        second_motion = high_motion.split(1, dim=0)
        second_sal_guide = self.attention_pre_sal([first_sal, second_sal])
        second_motion = torch.cat(second_motion[1:-1], dim=0)
        predict2_motion = self.predict2_motion(torch.cat([second_sal_guide, second_reduce_high + second_motion], 1)) + predict1_motion.narrow(0, 1, 4)

        third_sal = predict2_motion.narrow(0, 0, 1)
        third_reduce_high = high_motion.narrow(0, 3, 3)
        third_motion = high_motion.split(1, dim=0)
        third_sal_guide = self.attention_pre_sal([first_sal, second_sal, third_sal])
        third_motion = torch.cat(third_motion[2:-1], dim=0)
        predict3_motion = self.predict3_motion(torch.cat([third_sal_guide, third_reduce_high + third_motion], 1)) + predict2_motion.narrow(0, 1, 3)

        fourth_sal = predict3_motion.narrow(0, 0, 1)
        fourth_reduce_high = high_motion.narrow(0, 4, 2)
        fourth_sal_guide = self.attention_pre_sal([first_sal, second_sal, third_sal, fourth_sal])
        fourth_motion = high_motion.narrow(0, 3, 2)
        predict4_motion = self.predict4_motion(torch.cat([fourth_sal_guide, fourth_reduce_high + fourth_motion], 1)) + predict3_motion.narrow(0, 1, 2)

        fifth_sal = predict4_motion.narrow(0, 0, 1)
        fifth_reduce_high = high_motion.narrow(0, 5, 1)
        fifth_sal_guide = self.attention_pre_sal([first_sal, second_sal, third_sal, fourth_sal, fifth_sal])
        fifth_motion = high_motion.narrow(0, 4, 1)
        predict5_motion = self.predict5_motion(torch.cat([fifth_sal_guide, fifth_reduce_high + fifth_motion], 1)) + predict4_motion.narrow(0, 1, 1)

        if self.isTriplet:
            triplet = self.extract_region(torch.cat([predict2, predict3], dim=1),
                                      predict4_motion)

        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1_motion = F.upsample(predict1_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2_motion = F.upsample(predict2_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3_motion = F.upsample(predict3_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4_motion = F.upsample(predict4_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5_motion = F.upsample(predict5_motion, size=x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            if self.isTriplet:
                return predict3, predict1_motion, predict2_motion, predict3_motion, predict4_motion, predict5_motion, triplet
            else:
                return predict3, predict1_motion, predict2_motion, predict3_motion, predict4_motion, predict5_motion
        return F.sigmoid(predict5_motion)

    def attention_pre_sal(self, pre_sals):
        expand_num = len(pre_sals)
        pre_sals = torch.cat(pre_sals, dim=1)
        if expand_num == 2:
            pre_sals = self.pre_sals_attention2(pre_sals)
        elif expand_num == 3:
            pre_sals = self.pre_sals_attention3(pre_sals)
        elif expand_num == 4:
            pre_sals = self.pre_sals_attention4(pre_sals)
        elif expand_num == 5:
            pre_sals = self.pre_sals_attention5(pre_sals)
        reduce_pre_sals = torch.mean(pre_sals, dim=1, keepdim=True)
        reduce_sals = []
        for i in range(0, 6 - expand_num):
            reduce_sals.append(reduce_pre_sals)
        reduce_sals = torch.cat(reduce_sals, dim=0)
        return reduce_sals

    def extract_region(self, feats, final_map):
        batch_size = feats.size(0)
        final_map = torch.sigmoid(final_map)
        tmp_zeros = torch.zeros_like(final_map)
        tmp_ones = torch.ones_like(final_map)
        neg = torch.where(final_map < 0.3, tmp_ones, tmp_zeros)
        anchor = torch.where(final_map > 0.9, tmp_ones, tmp_zeros)
        pos = torch.where(final_map > 0.6, final_map, tmp_zeros)
        pos = torch.where(pos < 0.9, pos, tmp_zeros)
        pos = torch.where(pos > 0, tmp_ones, tmp_zeros)

        feats_anchor = F.adaptive_avg_pool2d(feats * anchor, 8)
        feats_pos = F.adaptive_avg_pool2d(feats * pos, 8)
        feats_neg = F.adaptive_avg_pool2d(feats * neg, 8)

        return [F.normalize(feats_anchor.view(batch_size, -1)), F.normalize(feats_pos.view(batch_size, -1)),
                F.normalize(feats_neg.view(batch_size, -1))]


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

if __name__ == '__main__':
    net = R3Net_prior(motion='GRU').cuda()
    input = torch.zeros([5, 3, 200, 200]).cuda()
    output = net(input)
    # VideoSaliency_2019-05-01 23:29:39 and VideoSaliency_2019-04-20 23:11:17/30000.pth
