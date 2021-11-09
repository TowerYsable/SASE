'''
Author: your name
Date: 2021-09-15 10:00:36
LastEditTime: 2021-09-15 11:17:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Speech_Enhancement-DCCRN/utils/metrics.py
'''
from tqdm import tqdm
import numpy as np

import torch
import torchaudio
from pypesq import pesq

from .matlab_eval import eval_composite

def pesq_score(model, dataloader, criterion, args):
    model.eval()
    test_pesq = 0.
    total_loss = 0
    with torch.no_grad():
        total_nan = 0
        for i, (mixed, target) in tqdm(enumerate(dataloader)):
            mixed = mixed.cuda(args.gpu)
            target = target.cuda(args.gpu)

            spec, wav = model(mixed) # time domain
            # print(pred_x)
            loss = criterion(wav, target)
            total_loss += loss.item()

            target = torch.squeeze(target, 1)

            psq = 0.
            nan = 0

            for idx in range(len(target)):
                clean_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(target[idx, :].view(1, -1))
                pred_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(wav[idx, :].view(1, -1))

                clean_x_16 = clean_x_16.cpu().numpy()
                pred_x_16 = pred_x_16.detach().cpu().numpy()

                score = pesq(clean_x_16.flatten(), pred_x_16.flatten(), 16000)

                if np.isnan(score):
                    nan += 1
                    total_nan += 1
                else:
                    psq += score

            psq /= (len(target) - nan)
            test_pesq += psq


        test_pesq /= (len(dataloader) - total_nan)
        loss_avg = total_loss / len(dataloader)

    return test_pesq, loss_avg
