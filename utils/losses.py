'''
Author: your name
Date: 2021-09-15 10:00:36
LastEditTime: 2021-10-22 17:32:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Speech_Enhancement-DCCRN/utils/losses.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .per_loss import perceptual_distance
from .pmsqe_loss import PMSQE
# from asteroid.filterbanks import transforms
# from asteroid.losses import SingleSrcPMSQE, PITLossWrapper
# from asteroid.filterbanks import STFTFB, Encoder
from .STOI_oss import STOILoss
from .contrastive_loss import contrastive_disentanglement_loss
# 使用不用的loss
"""
basis loss:
    1. MSE：Mean squared Error
    2. SI-SNR:The Scale-Invariant Source-to-Noise
perceptual loss:
    3. PMSQE:Perceptual Metric for Speech Quality Evaluation()
    4. LMS:Log Mel Spectra
"""

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target

    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

# 2 SDR
class SDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SDRLoss, self).__init__()
        self.eps = eps

    def forward(self, s1, s2):
        sn = l2_norm(s1, s1)
        sn_m_shn = l2_norm(s1 - s2, s1 - s2)
        sdr_loss = 10 * torch.log10(sn ** 2 / (sn_m_shn ** 2 + self.eps))
        return torch.mean(sdr_loss)

# 4 SISDRLoss
class SISDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SISDRLoss, self).__init__()
        self.eps = eps

    def forward(self, reference, estimation):
        reference_energy = torch.sum(reference ** 2, axis=-1, keepdims=True)
        # This is $\alpha$ after Equation (3) in [1].
        optimal_scaling = torch.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy + self.eps
        # This is $e_{\text{target}}$ in Equation (4) in [1].
        projection = optimal_scaling * reference
        # This is $e_{\text{res}}$ in Equation (4) in [1].
        noise = estimation - projection
        ratio = torch.sum(projection ** 2, axis=-1) / torch.sum(noise ** 2, axis=-1) + self.eps

        ratio = torch.mean(ratio)
        return 10 * torch.log10(ratio + self.eps)

# 3 SI-SNR
class SISNRLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, s1, s2):
        s2 = torch.squeeze(s2, 1)
        return -(si_snr(s1, s2, eps=self.eps))

# 1 MSE
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, x, y):
        return F.mse_loss(x, y, reduction='mean')

class SISNRMSELoss(nn.Module):
    def __init__(self,eps=1e-8):
        super(SISNRMSELoss, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = eps


    def forward(self, x, y):
        mseloss = F.mse_loss(x, y, reduction='mean')
        s2 = torch.squeeze(y, 1)
        sisnrloss = -(si_snr(x, s2, eps=self.eps))
        loss = (sisnrloss+self.gamma*mseloss)
        return loss


def get_array_mel_loss(clean_array, est_array):
    array_mel_loss = 0
    get_mel_loss = perceptual_distance()
    for i in range(len(clean_array)):
        mel_loss = get_mel_loss(clean_array[i], est_array[i])
        array_mel_loss += mel_loss
    avg_mel_loss = array_mel_loss / len(clean_array)
    return avg_mel_loss

class LMSLoss(nn.Module):
    def __init__(self,stft):
        super(LMSLoss, self).__init__()
        self.stft = stft

    def forward(self, x, y):
        clean_mags, clean_phase = self.stft(y)

        est_mags,est_phase = self.stft(x)
        mel_loss = get_array_mel_loss(clean_mags,est_mags)
        return mel_loss

class STOILoss2(nn.Module):
    def __init__(self):
        super(STOILoss2, self).__init__()

    def forward(self, x, y):
        print(x.shape)
        print("yyy",y.shape)
        stoi_func = STOILoss(sample_rate=48000)
        stoi_loss = stoi_func(x,y)
        return stoi_loss

def getlossmode(loss_mode="SI-SNR",stft=None,batch_size=None):
    if loss_mode == 'MSE':
        return MSELoss()
    elif loss_mode == 'SDR':
        return SDRLoss()
    elif loss_mode == 'SI-SNR':
        return SISNRLoss()
    elif loss_mode == 'SI-SDR':
        return SISDRLoss()
    elif loss_mode == 'SISNRMSELoss':
        return SISNRMSELoss()
    elif loss_mode == 'LMS':
        return LMSLoss(stft)
    elif loss_mode == 'STOI':
        return STOILoss2()
    elif loss_mode == 'CON':
        return contrastive_disentanglement_loss(batch_size=batch_size,stft=stft)

