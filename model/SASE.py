# references:https://github.com/maggie0830/DCCRN
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import flwr as fl
from collections import OrderedDict


from model.conv_stft import ConvSTFT, ConviSTFT
from model.complex_nn import *
from model.sa_goea_module import AMB
from model.ftb_moudle import TSB
import matplotlib.pyplot as plt

class EncoderBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cConv = cConv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                             stride=self.stride, padding=self.padding).cuda(self.args.gpu)


        self.cBN = cBatchNorm2d(self.out_channels).cuda(self.args.gpu)
        self.prelu = nn.PReLU().cuda(self.args.gpu)


    def forward(self, x):
        cConv = self.cConv(x) # cConv.shape torch.Size([4, 256, 4, 1653])

        cBN = self.cBN(cConv)
        output = self.prelu(cBN) # Real PReLU

        return output


class DecoderBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, kernel_size, stride, padding, output_padding, last_decoder=False):
        super(DecoderBlock, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.last_decoder = last_decoder

        self.trans_cConv = cConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                                            stride=self.stride, padding=self.padding, output_padding=self.output_padding).cuda(self.args.gpu)
        self.cBN = cBatchNorm2d(self.out_channels).cuda(self.args.gpu)
        self.prelu = nn.PReLU().cuda(self.args.gpu)

    def forward(self, x):
        trans_cConv = self.trans_cConv(x)
        if not self.last_decoder:
            cBN = self.cBN(trans_cConv)
            output = self.prelu(cBN)
        else:
            output = trans_cConv

        return output

class SASE(nn.Module):

    def __init__(
            self,
            args,
            rnn_layers=2,
            rnn_dim=128,
            win_len=400,
            win_inc=100,
            fft_len=512,
            win_type='hanning',
            masking_mode='E',
            use_clstm=False,
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256]):
        super(SASE, self).__init__()

        self.args = args
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_dim = rnn_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size

        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix

        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix).cuda(self.args.gpu)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix).cuda(self.args.gpu)

        self.encoder = nn.ModuleList().cuda(self.args.gpu)
        self.decoder = nn.ModuleList().cuda(self.args.gpu)

        self.ftb_pre = TSB(channel_phase = 2)


        # Encoder Part
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(EncoderBlock(
                args=self.args,
                in_channels=self.kernel_num[idx],
                out_channels=self.kernel_num[idx + 1],
                kernel_size=(self.kernel_size, 2),
                stride=(2, 1),
                padding=(2, 1)
            ))

        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                # todo print all
                rnns.append(
                    cLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_dim,
                        hidden_size=rnn_dim,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None
                    ).cuda(self.args.gpu)
                )
                self.enhance = nn.Sequential(*rnns).cuda(self.args.gpu)

        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_dim,
                num_layers=rnn_layers,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            ).cuda(self.args.gpu)
            self.projection = nn.Linear(self.rnn_dim * fac, hidden_dim * self.kernel_num[-1]).cuda(self.args.gpu)

        # Decoder Part
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1: 
                self.decoder.append(DecoderBlock(
                    args=self.args,
                    in_channels=self.kernel_num[idx]*2,
                    out_channels=self.kernel_num[idx -1],
                    kernel_size=(self.kernel_size, 2),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(1, 0)
                ))

            else: # last decoder
                self.decoder.append(DecoderBlock(
                    args=self.args,
                    in_channels=self.kernel_num[idx]*2,
                    out_channels=self.kernel_num[idx - 1],
                    kernel_size=(self.kernel_size, 2),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(1, 0)
                ))


    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

 
    def forward(self, inputs,lens=None):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len//2+1]
        imag = specs[:, self.fft_len//2+1:]
        spec_mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        spec_phase = torch.atan2(imag, real)

        complexSpec = torch.stack([real, imag], 1)
        complexSpec = complexSpec[:, :, 1:] #

        out = complexSpec
        out = self.ftb_pre(out)

        encoder_out = []


        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)


        batch_size, channels, dims, lengths = out.size()

        out = out.permute(3, 0, 1, 2)


        if self.use_clstm:

            real_rnn_input = out[:, :, :channels//2]
            imag_rnn_input = out[:, :, channels//2:]

            real_rnn_input = torch.reshape(real_rnn_input, [lengths, batch_size, channels//2*dims])
            imag_rnn_input = torch.reshape(imag_rnn_input, [lengths, batch_size, channels//2*dims])

            real_rnn_output, imag_rnn_output = self.enhance([real_rnn_input, imag_rnn_input])

            real_rnn_output = torch.reshape(real_rnn_output, [lengths, batch_size, channels//2, dims])
            imag_rnn_output = torch.reshape(imag_rnn_output, [lengths, batch_size, channels//2, dims])

            out = torch.cat([real_rnn_output, imag_rnn_output], 2)
       
        else: # complex lstm
            rnn_input = torch.reshape(out, [lengths, batch_size, channels*dims])
            rnn_output, _ = self.enhance(rnn_input)
            projection = self.projection(rnn_output)
            mask_out = torch.reshape(projection, [lengths, batch_size, channels, dims])
          
            attention_mlp_block = AMB(batch_size=batch_size, channels=channels, dims=dims).cuda(self.args.gpu)
            attention_input = torch.reshape(out, [batch_size, lengths, channels * dims])
           
            attention_mlp_block_output = attention_mlp_block(attention_input)
            attention_mask = torch.reshape(attention_mlp_block_output,
                                           [lengths, batch_size, channels, dims])  # torch.Size([1653, 10, 256, 4])
            out = attention_mask * mask_out
            

        out = out.permute(1, 2, 3, 0) 

        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1) 
            out = self.decoder[idx](out)
            out = out[..., 1:]

        out = self.ftb_pre(out)

        mask_real = out[:, 0]
        mask_imag = out[:, 1]

        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])


        if self.masking_mode == 'E':
            mask_mag = (mask_real ** 2 + mask_imag ** 2)**0.5
            real_phase = mask_real / (mask_mag + 1e-8)
            imag_phase = mask_imag / (mask_mag + 1e-8)

            mask_phase = torch.atan2(imag_phase, real_phase)
            mask_mag = torch.tanh(mask_mag)

            est_mag = mask_mag * spec_mag
            est_phase = spec_phase + mask_phase

            real = est_mag * torch.cos(est_phase)
            imag = est_mag * torch.sin(est_phase)

        elif self.masking_mode == 'C': # complex
            real, imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real

        elif self.masking_mode == 'R': # Real
            real, imag = real * mask_real, imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        
        out_wav = self.istft(out_spec)


        out_wav = torch.squeeze(out_wav, 1)

        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_spec, out_wav


def set_model(args, mode='CL'):
    if mode == 'E':
        model = SASE(args=args, rnn_dim=256, masking_mode='E')
    elif mode == 'R':
        model = SASE(args=args, rnn_dim=256, masking_mode='R')
    elif mode == 'C':
        model = SASE(args=args, rnn_dim=256, masking_mode='C')
    elif mode == 'CL':
        model = SASE(args=args, rnn_dim=256, masking_mode='E',
                      use_clstm=False, kernel_num=[32, 64, 128, 256, 256, 256])
    elif mode == 'attention':
        model = SASE(args=args, rnn_dim=256, masking_mode='E')
    else:
        raise Exception('non-supported mode!')
    return model








