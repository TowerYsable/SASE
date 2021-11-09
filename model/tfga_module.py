# reference:https://github.com/luuuyi/CBAM.PyTorch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# Spatial Attention
class ChannelPool(nn.Module):
    def forward(self, x):
        # x [batch, chaanel, freq, time]
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        # x [batch, chaanel, freq, time]
        x_compress = self.compress(x) # [batch, max+mean=2, freq, time]
        x_out = self.spatial(x_compress) # [batch, 1, freq, time]]
        scale = F.sigmoid(x_out) # [batch, 1, freq, time]
        return x * scale


# Channel Attention
class Flatten(nn.Module):
    def forward(self, x):
        # x [batch, channel, 1, 1]
        return x.view(x.size(0), -1) # [batch, channel]


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        # x [batch, channel, freq, time]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type =='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # [batch, channel, 1, 1]
                channel_att_raw = self.mlp( avg_pool )
                # [batch, channel]
                # print(channel_att_raw.size())
            elif pool_type =='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # [batch, channel, 1, 1]
                channel_att_raw = self.mlp( max_pool )
                # [batch, channel]
                # print(channel_att_raw.size())
            elif pool_type =='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type =='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # [batch, channel, freq, time]

        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

###############################
# Freq, Temporal Attention
###############################

class FreqGate(nn.Module):

    def __init__(self):
        super(FreqGate, self).__init__()

        self.conv = BasicConv(2, 1, kernel_size=(7, 1), stride=1, padding=(3, 0), relu=False)
        self.compress = ChannelPool()

    def forward(self, x):
        # x [Batch, C, Freq, Time]
        x_avg = F.avg_pool2d(x, (1, x.size(3)), stride=(1, x.size(3))) # [batch, C, Freq, 1]
        compress = self.compress(x_avg) # [batch, 2, Freq, 1]
        conv = self.conv(compress) # [batch, 1, Freq, 1]
        # _,_,f_vim,_ = conv.size()
        scale = F.sigmoid(conv)

        return scale


class TemporalGate(nn.Module):

    def __init__(self):
        super(TemporalGate, self).__init__()

        self.conv = BasicConv(2, 1, kernel_size=(1, 7), stride=1, padding=(0, 3), relu=False)
        self.compress = ChannelPool()

    def forward(self, x):
        # x [Batch, C, Freq, Time]
        x_avg = F.avg_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1)) # [batch, C, 1, Time]
        compress = self.compress(x_avg) # [batch, 2, 1, Time]
        conv = self.conv(compress) # [batch, 1, 1, TIme]
        scale = F.sigmoid(conv)

        return scale