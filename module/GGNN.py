import torch.nn as nn
from torch.autograd import Variable
import torch


class GGNN(nn.Module):
    def __init__(self, time_step, input_dim, hidden_dim, kernel_size, padding, bias=True):
        super(GGNN, self).__init__()

        self.time_step = time_step

        self.padding = padding
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.update_gate_w = nn.Conv2d(in_channels=input_dim,
                                    out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.update_gate_u = nn.Conv2d(in_channels=input_dim,
                                       out_channels=self.hidden_dim,
                                       kernel_size=kernel_size,
                                       padding=self.padding,
                                       bias=self.bias)

        self.reset_gate_w = nn.Conv2d(in_channels=input_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)

        self.reset_gate_u = nn.Conv2d(in_channels=input_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.cur_memory_w = nn.Conv2d(in_channels=input_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

        self.cur_memory_u = nn.Conv2d(in_channels=input_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

    def forward(self, input):
        # frame_feats = torch.split(input, 1)
        b, c, w, h = input.size()
        # for i, feat in enumerate(frame_feats):
        feats = input

        for j in range(self.time_step):

            frame_feats = torch.sum(feats, 0, keepdim=True)
            message = []
            for i in range(b):
                message.append(frame_feats)
            message = torch.cat(message, dim=0)

            z = torch.sigmoid(self.update_gate_w(message) + self.update_gate_u(feats))
            r = torch.sigmoid(self.reset_gate_w(message) + self.reset_gate_u(feats))

            h = torch.tanh(self.cur_memory_w(message) + self.cur_memory_u(r * feats))
            feats = (1 - z) * feats + z * h

        return feats

if __name__ == '__main__':
    a = torch.ones([5, 1, 10, 10])
    model = GGNN(5, 1, 1, 3, 1)
    b = model(a)
    print(b.size())





