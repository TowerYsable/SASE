import torch
import torch.nn as nn
import numpy as np


class contrastive_disentanglement_loss(nn.Module):

    def __init__(self, dev=torch.device('cuda'), batch_size=None,eps=1e-8,stft=None):
        super(contrastive_disentanglement_loss, self).__init__()
        self.dev = dev
        self.nt_xent_criterion = NTXentLoss(device=self.dev, batch_size=batch_size, use_cosine_similarity=1,temperature=0.2)
        self.eps = eps
        self.alpha = 0.5
        self.stft = stft
        self.batch_size = batch_size
        self.softmax = nn.LogSigmoid()

    def forward(self, x1, x2): #, normalise=1 x1:est x2:truth ,相位尽可能相似
        clean_mags, clean_phase = self.stft(x2)
        est_mags, est_phase = self.stft(x1)

        # print("self.est_mags", est_mags.shape)
        # print("self.est_mags", est_phase.shape)
        # print("self.est_mags", x1.shape)
        est_phase = torch.reshape(est_phase,[self.batch_size,-1])
        clean_phase = torch.reshape(clean_phase,[self.batch_size,-1])
        est_mags = torch.reshape(est_mags, [self.batch_size, -1])
        clean_mags = torch.reshape(clean_mags, [self.batch_size, -1])

        phase_loss = self.nt_xent_criterion.forward(clean_phase, est_phase)
        sisnr_loss = -(self.si_snr(x1, torch.squeeze(x2, 1), eps=self.eps))


        weight_loss = AutomaticWeightedLoss(2)
        loss_sum = (1-self.alpha)*sisnr_loss + self.alpha*phase_loss

        return loss_sum

    def si_snr(self,s1, s2, eps=1e-8):
        s1_s2_norm = self.l2_norm(s1, s2)
        s2_s2_norm = self.l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
        e_noise = s1 - s_target

        target_norm = self.l2_norm(s_target, s_target)
        noise_norm = self.l2_norm(e_noise, e_noise)

        snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
        return torch.mean(snr)

    def l2_norm(self,s1, s2):
        norm = torch.sum(s1 * s2, -1, keepdim=True)
        return norm


#################
# http://github.com/Mikoto10032/AutomaticWeightedLoss
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, use_cosine_similarity, temperature=0.2):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _PearsonCorrelation(self,tensor_1, tensor_2):
        x = tensor_1
        y = tensor_2
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    # https://github.com/Sara-Ahmed/SiT/blob/main/losses.py
    def forward(self, zis, zjs): #zis:est zjs:truth #相位尽可能相似.   幅度谱尽可能不相似?
        representations = torch.cat([zjs.squeeze(1), zis.squeeze(1)], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        # print("self.batch_size",self.batch_size)
        # print("self.similarity_matrix", similarity_matrix.shape)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        # logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)