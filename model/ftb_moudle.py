import torch
import torch.nn as nn


class GLayerNorm2d(nn.Module):

    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones([1, in_channel, 1, 1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel, 1, 1]))

    def forward(self, inputs):
        mean = torch.mean(inputs, [1, 2, 3], keepdim=True)
        var = torch.var(inputs, [1, 2, 3], keepdim=True)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps) * self.beta + self.gamma
        return outputs

class TSB(nn.Module):

    def __init__(self,channel_phase=2):
        super(TSB, self).__init__()
        self.phase_conv1 = nn.Sequential(
            nn.Conv2d(channel_phase, channel_phase, kernel_size=(5, 5), padding=(2, 2)),
            GLayerNorm2d(channel_phase),
        )

    def forward(self, inputs):
        phase_out1 = self.phase_conv1(inputs)
        return phase_out1

class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=9, r_channel=5):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU()
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention
        conv1_out = self.conv1(inputs)
        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs