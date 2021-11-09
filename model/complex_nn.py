"""
official code
https://github.com/huyanxin/DeepComplexCRN/blob/bc6fd38b0af9e8feb716c81ff8fbacd7f71ad82f/complexnn.py#L79
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.tfga_module import FreqGate,TemporalGate

def get_casual_padding1d():
    pass


def get_casual_padding2d():
    pass


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


class cPReLU(nn.Module):

    def __init__(self, complex_axis=1):
        super(cPReLU, self).__init__()
        self.r_prelu = nn.PReLU()
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real, imag], self.complex_axis)


class cLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(cLSTM, self).__init__()
        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)

        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs

        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)

        r2r_out = self.real_lstm(real)[0] # real + real lstm
        r2i_out = self.imag_lstm(real)[0] # real + imag lstm
        i2r_out = self.real_lstm(imag)[0] # imag + real lstm
        i2i_out = self.imag_lstm(imag)[0] # imag + imag lstm

        # Complex multiplication
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out

        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)

        output = [real_out, imag_out]

        return output

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


class cConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            groups=1,
            # causal=False,
            causal=True,
            complex_axis=1):
        super(cConv2d, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis

        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)
        self.FreqGate_real = FreqGate()
        self.TemporalGate_real = TemporalGate()
        self.FreqGate_imag = FreqGate()
        self.TemporalGate_imag = TemporalGate()

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[0], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)

            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)
            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out_1 = torch.cat([real, imag], self.complex_axis)
        batch_size, channels, dims, lengths = out_1.size()
        self.dispaly_feature_color(torch.reshape(out_1, [batch_size, lengths, dims * channels]), "complex_gate_attention_1")

        real_freq_out = self.FreqGate_real(real)
        real_time_out = self.TemporalGate_real(real)
        scale_real = (real_freq_out * real_time_out)
        real = real * scale_real

        imag_freq_out = self.FreqGate_imag(imag)
        imag_time_out = self.TemporalGate_imag(imag)
        scale_imag = (imag_freq_out * imag_time_out)
        imag = real * scale_imag

        out = torch.cat([real, imag], self.complex_axis)

        self.dispaly_feature_color(torch.reshape(out, [batch_size, lengths, dims * channels]), "complex_gate_attention_2")

        return out

    def example_plot(self,ax, input,title, fontsize=12, hide_labels=False):
        pc = ax.pcolormesh(input, vmin=-2.5, vmax=2.5)

        if not hide_labels:
            ax.set_xlabel('frequency', fontsize=fontsize)
            ax.set_ylabel('time', fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
        return pc

    def dispaly_feature_color(self,input, title="none"):
        fig = plt.figure(constrained_layout=True, figsize=(10, 5))

        axsLeft = fig.subplots(1, 2, sharey=True)
        fig.set_facecolor('0.75')
        fig.set_facecolor('w')

        input = input[0]
        input = input.cpu().detach().numpy()

        for ax in axsLeft:
            pc = self.example_plot(ax, input,title)

        fig.suptitle('Figure suptitle', fontsize='xx-large')
        plt.savefig(title + "_img.png")
        plt.show()


class cConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1,1),
            stride=(1,1),
            padding=(0,0),
            output_padding=(0,0),
            causal=False,
            complex_axis=1,
            groups=1):
        super(cConvTranspose2d, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.complex_axis=complex_axis

        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)
        self.FreqGate_real = FreqGate()
        self.TemporalGate_real = TemporalGate()
        self.FreqGate_imag = FreqGate()
        self.TemporalGate_imag = TemporalGate()

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)

        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)

            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        real_freq_out = self.FreqGate_real(real)
        real_time_out = self.TemporalGate_real(real)
        scale_real = (real_freq_out * real_time_out)
        real = real * scale_real

        imag_freq_out = self.FreqGate_imag(imag)
        imag_time_out = self.TemporalGate_imag(imag)
        scale_imag = (imag_freq_out * imag_time_out)
        imag = real * scale_imag

        out = torch.cat([real, imag], self.complex_axis)

        return out


class cBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=False, complex_axis=1):
        super(cBatchNorm2d, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) 
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):

        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None: 
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else: 
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            # TODO：detele
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
