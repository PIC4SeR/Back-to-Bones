import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix


class MixStyle2(nn.Module):
    """MixStyle (w/ domain prior).
    The input should contain two equal-sized mini-batches from two distinct domains.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2], keepdim=True)
        var = x.var(dim=[2], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.arange(B - 1, -1, -1) # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2)]
        perm_a = perm_a[torch.randperm(B // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix
    
    
class Intra_ADR(nn.Module):
    def __init__(self, inp, outp, Norm=None, group=1, stride=1, **kwargs):
        super(Intra_ADR, self).__init__()
        self.E_space = nn.Sequential(
            nn.ConvTranspose1d(inp, outp, kernel_size=2, stride=stride, padding=0, output_padding=0, groups=1,
                            bias=True, dilation=1, padding_mode='zeros'),
            nn.InstanceNorm1d(outp),
            nn.ReLU(inplace=True),
            )
        self.mixstyle = MixStyle(p=.5, alpha=.3)
        
    def cc_kth_p(self, input, kth=0):
        kth = 10
        input = torch.topk(input, kth, dim=1)[0]  # n,k,h,w

        input = input.mean(1, keepdim=True)
        return input

    def forward(self, x):
        branch = self.E_space(x)
        branch2 = branch
        x_adr = branch
        branch_ = branch #.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch_, 2)
        branch_out = self.cc_kth_p(branch)
        return branch_out, branch2, x_adr
    
    
def Inter_ADR(t_cls_pred, cls_pred, t_fms, fms, label, device):
    t_mask = [(t_cls_pred[i] == label.data) * 1. for i in range(len(t_cls_pred))]
    t_mask = [t_mask[i].view(1, -1).permute(1, 0) for i in range(len(t_cls_pred))]
    mask = (cls_pred == label.data) * 1.
    mask = mask.view(1, -1).permute(1, 0)
    t_ats = [at(t_fms[i]) for i in range(len(t_cls_pred))]
    ats = at(fms)
    l2_dirs, l2_dvrs = 0, 0
    
    for res_i in range(len(ats)):
        t_mask_dir = [t_mask[i].repeat(1, ats[res_i].size()[1]).to(device) for i in range(len(t_cls_pred))]
        mask_dir = mask.repeat(1, ats[res_i].size()[1]).to(device)

        t_mask_dvr = [1. - t_mask_dir[i] for i in range(len(t_cls_pred))]
        mask_dvr = 1. - mask_dir

        u_plus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dir[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_plus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dir.unsqueeze(2).contiguous())]
        u_plus_temp = torch.cat(u_plus_temp, dim=2)
        u_plus = u_plus_temp.max(2)[0]

        u_minus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dvr[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_minus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dvr.unsqueeze(2).contiguous())]
        u_minus_temp = torch.cat(u_minus_temp, dim=2)
        u_minus = u_minus_temp.max(2)[0]
        
        mask_plus_0 = torch.gt(u_plus, torch.zeros_like(u_plus)).to(device)
        mask_minus_0 = torch.gt(u_minus, torch.zeros_like(u_minus)).to(device)

        l2_dir = ((ats[res_i] * mask_plus_0 - u_plus.detach())**2).mean() * fms[res_i].shape[1]
        l2_dvr = ((ats[res_i] * mask_minus_0 - 0 *u_minus.detach())**2).mean() * fms[res_i].shape[1]
        l2_dirs += l2_dir
        l2_dvrs += l2_dvr
        
    return l2_dirs, l2_dvrs

def at(fms):
    ats = []
    for fm in fms:
        (N, C, HW) = fm.shape
        ats.append(F.softmax(fm.reshape(N, C, -1), -1).mean(1))
    return ats