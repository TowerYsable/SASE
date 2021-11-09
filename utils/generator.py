import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import torch
import torchaudio
from model.conv_stft import *


def generate_wav(model, max_len, args):
    model.eval()
    file_list = os.listdir(args.denoising_file)

    for idx in range(len(file_list)):
        name = file_list[idx]
        mixed = os.path.join(args.denoising_file, name)
        waveform, _ = torchaudio.load(mixed)


        waveform = waveform.numpy()

        current_len = waveform.shape[1]
        pad = np.zeros((1, max_len), dtype='float32')
        pad[0, -current_len:] = waveform[0, :max_len]

        input = torch.from_numpy(pad).cuda(args.gpu)
        with torch.no_grad():
            spec, wav = model(input)

        name = "predict_" + name
        output = os.path.join("output_only_one", name)

        wav_cpu = wav.cpu()
        sf.write(output, wav.squeeze(0)[-current_len:].cpu().numpy(), samplerate=48000, format='WAV', subtype='PCM_16')
