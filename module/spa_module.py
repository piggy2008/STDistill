import torch.nn as nn
import torch
import torch.nn.functional as F

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = F.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class STA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(STA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv_spatial = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_temporal = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.out_fuse = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x_spatial, x_temporal):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x_spatial.size()
        proj_query = self.query_conv(x_spatial).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_temporal).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        energy = ((self.chanel_in // 8) ** -.5) * energy
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv_spatial(x_spatial).view(m_batchsize, -1, width*height)
        proj_value2 = self.value_conv_temporal(x_temporal).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out2 = torch.bmm(proj_value2, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out2 = out2.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x_spatial
        # out2 = self.gamma2*out2 + x_temporal
        # out = self.gamma*out + x_spatial + self.gamma2*out2 + x_temporal
        out = out + x_spatial + out2 + x_temporal
        # out = out + out2
        # out = self.gamma * out + self.gamma2 * out2
        # out = self.out_fuse(torch.cat([out, out2], dim=1))
        return out

class STA2_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, out_dim):
        super(STA2_Module, self).__init__()
        self.value_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0), nn.BatchNorm2d(256), nn.PReLU(),
            )


    def forward(self, pre, cur):
        b, c, h, w = cur.size()
        pre_a = pre.view(b, c, h * w).permute(0, 2, 1)
        cur_a = cur.view(b, c, h * w)

        feat = torch.matmul(pre_a, cur_a)
        feat = F.softmax((c ** -.5) * feat, dim=-1)
        feat = torch.matmul(feat, cur_a.permute(0, 2, 1)).permute(0, 2, 1)
        feat_mutual = torch.cat([feat, cur_a], dim=1).view(b, 2 * c, h, w)

        feat_mutual = self.value_conv(feat_mutual)
        return feat_mutual

if __name__ == '__main__':
    x = torch.rand([5, 64, 70, 70])
    y = torch.rand([5, 64, 70, 70])
    model = STA_Module(in_dim=64)
    out = model(x, y)
    print(out.size())