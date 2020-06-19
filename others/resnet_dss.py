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

class DSS(nn.Module):
    def __init__(self, motion='GRU', se_layer=False, attention=False):
        super(DSS, self).__init__()

        self.motion = motion
        self.se_layer = se_layer
        self.attention = attention

        resnext = ResNet101()

        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

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
        if self.training:
            return dsn6_out_up, dsn5_out_up, dsn4_out_up, dsn3_out_up, dsn2_out_up, final_dsn
        return F.sigmoid(final_dsn)

if __name__ == '__main__':
    model = DSS()
    x = torch.zeros(5, 3, 473, 473)
    out = model(x)
    print(out[-1].size())