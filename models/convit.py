""" ConViT Model

@article{d2021convit,
  title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  author={d'Ascoli, St{\'e}phane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  journal={arXiv preprint arXiv:2103.10697},
  year={2021}
}

Paper link: https://arxiv.org/abs/2103.10697
Original code: https://github.com/facebookresearch/convit, original copyright below
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, PatchEmbed, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer_hybrid import HybridEmbed

import torch.autograd as autograd

import torch
import torch.nn as nn
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # ConViT
    'convit_base': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_base.pth")
}


class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 locality_strength=1.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.locality_strength = locality_strength

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.rel_indices: torch.Tensor = torch.zeros(1, 1, 1, 3)  # silly torchscript hack, won't work with None

    def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= self.locality_strength

    def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / N
        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., representation_size=3072, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=3, locality_strength=1., use_pos_embed=True, drop_b=2.0/3.0*100, drop_f=2.0/3.0*100, 
                 meth=None, gaussian=False, is_cond=False):
        
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = self.criterion.to(self.device)
        
        self.drop_b = drop_b
        self.drop_f = drop_f
        self.meth = meth
        self.gaussian = gaussian
        
        embed_dim *= num_heads
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i < local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        for n, m in self.named_modules():
            if hasattr(m, 'local_init'):
                m.local_init()

        if self.meth == 'Transfer':
            self.adv_classifier = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.adv_classifier.load_state_dict(self.head.state_dict())
            
            self.register_buffer('update_count', torch.tensor([0]))
            self.d_steps_per_g = 10
            
            self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=1e-3)

        elif self.meth == 'CAD':
            self.is_conditional = is_cond  # whether to use bottleneck conditioned on the label
            self.base_temperature = 0.07
            self.temperature = 0.1
            self.is_project = False  # whether apply projection head
            self.is_normalized = False # whether apply normalization to representation when computing loss

            # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
            # the two versions have the same optima, but we find the latter is more stable
            self.is_flipped = True

            if self.is_project:
                self.project = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim, 128),
                )
                params += list(self.project.parameters())
            
        elif self.meth == 'MMD':
            self.gaussian = True
            self.meth = 'CORAL'
            
        elif self.meth == 'CausIRL_MMD':
            self.gaussian = True
            self.meth = 'CausIRL_CORAL'
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u, blk in enumerate(self.blocks):
            if u == self.local_up_to_layer:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.gaussian:
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def loss_gap(self, minibatches, device='cuda'):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.forward_features(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss    
    
    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)
    
    def forward(self, x, y=None): 
                    
        if not isinstance(x, list): # Standard input
            x_all = x.to(self.device)
            
            all_f = self.forward_features(x_all)
            all_p = self.head(all_f)
            
            if y is None:
                return all_p
            
            y = y.to(self.device)
        
        if self.meth == 'Mixup': # Alternative input
            mixup_alpha = 0.2
            objective = 0

            #print(x[1][0].shape)
            
            for (xi, yi), (xj, yj) in random_pairs_of_minibatches(x):
                lam = np.random.beta(mixup_alpha, mixup_alpha)

                x_m = lam * xi + (1 - lam) * xj

                predictions = self.head(self.forward_features(x_m))

                objective += lam * self.criterion(predictions, yi)
                objective += (1 - lam) * self.criterion(predictions, yj)
            
            objective /= len(x)
            return objective
        
        
       
        elif self.meth == 'RSC': # Standard input
            all_o = torch.nn.functional.one_hot(y, self.num_classes)
            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

            percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
            percentiles = torch.Tensor(percentiles)
            percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
            mask_f = all_g.lt(percentiles.to(self.device)).float()

            all_f_muted = all_f * mask_f
            all_p_muted = self.head(all_f_muted)

            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), self.drop_b)
            mask_b = changes.lt(percentile).float().view(-1, 1)
            mask = torch.logical_or(mask_f, mask_b).float()

            all_p_muted_again = self.head(all_f * mask)
            
            loss = self.criterion(all_p_muted_again, y)
            
            return loss
        
        elif self.meth == 'CORAL_old':
            objective = 0
            penalty = 0
            mmd_lambda = 1
            nmb = len(x)
            for i in range(nmb):
                objective += self.criterion(all_p[None,i], y[None,i])
                for j in range(i + 1, nmb):
                    penalty += self.mmd(all_f[i], all_f[j])
                    
            objective /= nmb
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)
                
            loss = objective + (mmd_lambda*penalty)

            return loss

    
        elif self.meth == 'CORAL': # Alternative input
            objective = 0
            penalty = 0
            mmd_lambda = 1
            nmb = 3
                        
            features = [self.forward_features(xi) for xi, _ in x]
            classifs = [self.head(fi) for fi in features]
            targets = [yi for _, yi in x]
                        
            for i in range(nmb):
                objective += self.criterion(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    penalty += self.mmd(features[i], features[j])
                    
            objective /= nmb
            penalty /= (nmb * (nmb - 1) / 2)
                
            loss = objective + (mmd_lambda*penalty)

            return loss   

        elif self.meth == 'CausIRL_CORAL': # gaussian=True variant ######################################################
            objective = 0
            penalty = 0
            mmd_gamma = 1
            nmb = len(x)

            first = None
            second = None

            features = [self.forward_features(xi) for xi, _ in x]
            classifs = [self.head(fi) for fi in features]
            targets = [yi for _, yi in x]
            
            for i in range(nmb):
                objective += self.criterion(classifs[i] + 1e-16, targets[i])
                slice = np.random.randint(0, len(features[i]))
                if first is None:
                    first = features[i][:slice]
                    second = features[i][slice:]
                else:
                    first = torch.cat((first, features[i][:slice]), 0)
                    second = torch.cat((second, features[i][slice:]), 0)
            if len(first) > 1 and len(second) > 1:
                penalty = torch.nan_to_num(self.mmd(first, second))
            else:
                penalty = torch.tensor(0)
            objective /= nmb


            if torch.is_tensor(penalty):
                penalty = penalty.item()
                
            #print(objective, penalty)
            return objective + mmd_gamma*penalty

        
        elif self.meth == 'CAD': # is_conditional=True variant ######################################################
            all_x = torch.cat([x for x, y in x]).cuda()
            all_y = torch.cat([y for x, y in x]).cuda()
            all_z = self.forward_features(all_x)
            all_d = torch.cat([
                torch.full((x.shape[0],), i, dtype=torch.int64, device=self.device)
                for i, (x, y) in enumerate(x)
            ])

            bn_loss = self.bn_loss(all_z, all_y, all_d)
            clf_out = self.head(all_z)
            clf_loss = self.criterion(clf_out, all_y) # F.cross_entropy(clf_out, all_y) #############################
            total_loss = clf_loss + 0.001 * bn_loss
            return total_loss
        
        else: # Baseline
            all_x = torch.cat([x for x, y in x]).cuda()
            all_y = torch.cat([y for x, y in x]).cuda()
            
            all_p = self.forward_features(all_x)
            loss = self.criterion(self.head(all_p), all_y)
            
            return loss

def _create_convit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    return build_model_with_cfg(
        ConViT, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)


@register_model
def convit_base(pretrained=True, **kwargs):
    model_args = dict(
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_base', pretrained=pretrained, **model_args)
    return model

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
    return pairs
