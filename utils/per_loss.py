import math
import numpy as np
import torch

fs = 48000
win_len = 400
DEVICE = torch.device("cuda")

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

# generate Mel filter bank
def melFilterBank(numCoeffs, fftSize=None):
    minHz = 0
    maxHz = fs / 2  # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = win_len
    else:
        numFFTBins = int(fftSize / 2) + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.array(range(numCoeffs + 2))
    melRange = melRange.astype(np.float32)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = math.floor(numFFTBins * melCenterFilters[i] / maxHz)

    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, numFFTBins))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)

        startRange = int(melCenterFilters[i - 1])
        midRange = int(melCenterFilters[i])
        endRange = int(melCenterFilters[i + 1])

        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))

        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat

############################################################################
#      Finally: a perceptual loss function (based on Mel scale)            #
############################################################################

FFT_SIZE = 512

# multi-scale MFCC distance
MEL_SCALES = [16, 32, 64]  # for LMS
# PAM : MEL_SCALES = [32, 64]

# given a (symbolic Theano) array of size M x WINDOW_SIZE
#     this returns an array M x N where each window has been replaced
#     by some perceptual transform (in this case, MFCC coeffs)
def perceptual_transform(x):
    # precompute Mel filterbank: [FFT_SIZE x NUM_MFCC_COEFFS]
    MEL_FILTERBANKS = []
    for scale in MEL_SCALES:
        filterbank_npy = melFilterBank(scale, FFT_SIZE).transpose()
        torch_filterbank_npy = torch.from_numpy(filterbank_npy).type(torch.FloatTensor)
        MEL_FILTERBANKS.append(torch_filterbank_npy.to(DEVICE))

    transforms = []
    # powerSpectrum = torch_dft_mag(x, DFT_REAL, DFT_IMAG)**2

    powerSpectrum = x.view(-1, FFT_SIZE // 2 + 1)
    powerSpectrum = 1.0 / FFT_SIZE * powerSpectrum

    for filterbank in MEL_FILTERBANKS:
        filteredSpectrum = torch.mm(powerSpectrum, filterbank)
        filteredSpectrum = torch.log(filteredSpectrum + 1e-7)
        transforms.append(filteredSpectrum)

    return transforms

class rmse(torch.nn.Module):
    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = torch.sqrt(mse + 1e-7)

        return torch.mean(rmse)

# perceptual loss function
class perceptual_distance(torch.nn.Module):

    def __init__(self):
        super(perceptual_distance, self).__init__()

    def forward(self,y_pred, y_true):
        rmse_loss = rmse()
        # y_true = torch.reshape(y_true, (-1, WINDOW_SIZE))
        # y_pred = torch.reshape(y_pred, (-1, WINDOW_SIZE))

        pvec_true = perceptual_transform(y_true)
        pvec_pred = perceptual_transform(y_pred)

        distances = []
        for i in range(0, len(pvec_true)):
            error = rmse_loss(pvec_pred[i], pvec_true[i])
            error = error.unsqueeze(dim=-1)
            distances.append(error)
        distances = torch.cat(distances, axis=-1)

        loss = torch.mean(distances, axis=-1)
        return torch.mean(loss)