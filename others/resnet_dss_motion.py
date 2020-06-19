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
from module.attention import BaseOC_Context_Module
from model import _ASPP

class DSS(nn.Module):
    def __init__(self, motion='GRU', se_layer=False, attention=False, pre_attention=True):
        super(DSS, self).__init__()

        self.motion = motion
        self.se_layer = se_layer
        self.attention = attention
        self.pre_attention = pre_attention

        resnext = ResNet101()

        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )

        if self.motion == 'GRU':
            self.reduce_high_motion = ConvGRU(input_size=(119, 119), input_dim=256,
                                              hidden_dim=128,
                                              kernel_size=(3, 3),
                                              num_layers=1,
                                              batch_first=True,
                                              bias=True,
                                              return_all_layers=False)
            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        elif self.motion == 'LSTM':
            self.reduce_high_motion = ConvLSTM(input_size=(119, 119), input_dim=256,
                                               hidden_dim=32,
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
                nn.Conv2d(128, 32, kernel_size=1)
            )

        self.predict1_motion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict2_motion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict3_motion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict4_motion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        if self.pre_attention:
            self.pre_sals_attention2 = SELayer(2, 1)
            self.pre_sals_attention3 = SELayer(3, 1)
            self.pre_sals_attention4 = SELayer(4, 1)

        self.dsn6 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        self.dsn5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        self.dsn4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        self.dsn4_fuse = nn.Conv2d(3, 1, kernel_size=1)

        self.dsn3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        self.dsn3_fuse = nn.Conv2d(3, 1, kernel_size=1)

        self.dsn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.dsn2_fuse = nn.Conv2d(5, 1, kernel_size=1)

        self.dsn_all_fuse = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x):
        x_size = x.size()[2:]
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]
        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.upsample(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))
        reduce_high = F.upsample(reduce_high, size=l0_size, mode='bilinear', align_corners=True)

        if len(self.motion) > 0:
            if self.motion == 'no':
                high_motion = self.reduce_high_motion(reduce_high)
            else:
                high_side, high_state = self.reduce_high_motion(reduce_high.unsqueeze(0))
                high_motion = high_side[0].squeeze(0)
            if self.se_layer:
                high_motion = self.motion_se(high_motion)

        dsn6_out = self.dsn6(layer4)
        dsn6_out_up = F.upsample(dsn6_out, size=x_size, mode='bilinear', align_corners=True)

        dsn5_out = self.dsn5(layer3)
        dsn5_out_up = F.upsample(dsn5_out, size=x_size, mode='bilinear', align_corners=True)

        dsn4_out = self.dsn4(layer2)
        dsn6_out_up_4 = F.upsample(dsn6_out, size=dsn4_out.size()[2:], mode='bilinear', align_corners=True)
        dsn5_out_up_4 = F.upsample(dsn5_out, size=dsn4_out.size()[2:], mode='bilinear', align_corners=True)
        dsn4_fuse = self.dsn4_fuse(torch.cat([dsn4_out, dsn6_out_up_4, dsn5_out_up_4], dim=1))
        dsn4_out_up = F.upsample(dsn4_fuse, size=x_size, mode='bilinear', align_corners=True)

        dsn3_out = self.dsn3(layer1)
        dsn6_out_up_3 = F.upsample(dsn6_out, size=dsn3_out.size()[2:], mode='bilinear', align_corners=True)
        dsn5_out_up_3 = F.upsample(dsn5_out, size=dsn3_out.size()[2:], mode='bilinear', align_corners=True)
        dsn3_fuse = self.dsn3_fuse(torch.cat([dsn3_out, dsn6_out_up_3, dsn5_out_up_3], dim=1))
        dsn3_out_up = F.upsample(dsn3_fuse, size=x_size, mode='bilinear', align_corners=True)

        dsn2_out = self.dsn2(layer0)
        dsn6_out_up_2 = F.upsample(dsn6_out, size=dsn2_out.size()[2:], mode='bilinear', align_corners=True)
        dsn5_out_up_2 = F.upsample(dsn5_out, size=dsn2_out.size()[2:], mode='bilinear', align_corners=True)
        dsn4_out_up_2 = F.upsample(dsn4_out, size=dsn2_out.size()[2:], mode='bilinear', align_corners=True)
        dsn3_out_up_2 = F.upsample(dsn3_out, size=dsn2_out.size()[2:], mode='bilinear', align_corners=True)
        dsn2_fuse = self.dsn2_fuse(torch.cat([dsn2_out, dsn5_out_up_2, dsn4_out_up_2, dsn6_out_up_2, dsn3_out_up_2], dim=1))
        dsn2_out_up = F.upsample(dsn2_fuse, size=x_size, mode='bilinear', align_corners=True)

        final_dsn = self.dsn_all_fuse(torch.cat([dsn6_out_up, dsn5_out_up, dsn4_out_up, dsn3_out_up, dsn2_out_up], dim=1))

        final_dsn_down = F.upsample(final_dsn, size=high_motion.size()[2:], mode='bilinear', align_corners=True)
        #########  MEN module  #########
        first_sal = final_dsn_down.narrow(0, 0, 1)
        first_reduce_high = high_motion.narrow(0, 1, 4)
        first_motion = high_motion.split(1, dim=0)
        first_sal_guide = torch.cat([first_sal, first_sal, first_sal, first_sal], dim=0)
        first_motion = torch.cat(first_motion[:-1], dim=0)
        predict1_motion = self.predict1_motion(
            torch.cat([first_sal_guide, first_reduce_high + first_motion], 1)) + final_dsn_down.narrow(0, 1, 4)

        second_sal = predict1_motion.narrow(0, 0, 1)
        second_reduce_high = high_motion.narrow(0, 2, 3)
        second_motion = high_motion.split(1, dim=0)
        second_sal_guide = self.attention_pre_sal([first_sal, second_sal])
        second_motion = torch.cat(second_motion[1:-1], dim=0)
        predict2_motion = self.predict2_motion(
            torch.cat([second_sal_guide, second_reduce_high + second_motion], 1)) + predict1_motion.narrow(0, 1, 3)

        third_sal = predict2_motion.narrow(0, 0, 1)
        third_reduce_high = high_motion.narrow(0, 3, 2)
        third_motion = high_motion.split(1, dim=0)
        third_sal_guide = self.attention_pre_sal([first_sal, second_sal, third_sal])
        third_motion = torch.cat(third_motion[2:-1], dim=0)
        predict3_motion = self.predict3_motion(
            torch.cat([third_sal_guide, third_reduce_high + third_motion], 1)) + predict2_motion.narrow(0, 1, 2)

        fourth_sal = predict3_motion.narrow(0, 0, 1)
        fourth_reduce_high = high_motion.narrow(0, 4, 1)
        fourth_sal_guide = self.attention_pre_sal([first_sal, second_sal, third_sal, fourth_sal])
        fourth_motion = high_motion.narrow(0, 3, 1)
        predict4_motion = self.predict4_motion(
            torch.cat([fourth_sal_guide, fourth_reduce_high + fourth_motion], 1)) + predict3_motion.narrow(0, 1, 1)

        predict1_motion = F.upsample(predict1_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2_motion = F.upsample(predict2_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3_motion = F.upsample(predict3_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4_motion = F.upsample(predict4_motion, size=x.size()[2:], mode='bilinear', align_corners=True)
        if self.training:
            return final_dsn, predict1_motion, predict2_motion, predict3_motion, predict4_motion
        return F.sigmoid(predict4_motion)

    def attention_pre_sal(self, pre_sals):
        expand_num = len(pre_sals)
        pre_sals = torch.cat(pre_sals, dim=1)
        if expand_num == 2:
            pre_sals = self.pre_sals_attention2(pre_sals)
        elif expand_num == 3:
            pre_sals = self.pre_sals_attention3(pre_sals)
        elif expand_num == 4:
            pre_sals = self.pre_sals_attention4(pre_sals)
        reduce_pre_sals = torch.mean(pre_sals, dim=1, keepdim=True)
        reduce_sals = []
        for i in range(0, 5 - expand_num):
            reduce_sals.append(reduce_pre_sals)
        reduce_sals = torch.cat(reduce_sals, dim=0)
        return reduce_sals

if __name__ == '__main__':
    model = DSS(motion='GRU').cuda()
    x = torch.zeros(5, 3, 473, 473).cuda()
    out = model(x)
    print(out[-1].size())