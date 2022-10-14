from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math

from einops.layers.torch import Rearrange
from torch import nn, einsum
from einops import rearrange, repeat

class ResNet(nn.Module):
    def __init__(self, block, layers, drop_perc=0.5, jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        #--------ATTENTION
        self.pecent_attention_batch = 1 # batch
        self.drop_mask = drop_perc
        
        if layers[3] == 2:
            self.feature_backbone_dim = 512
            self.dim_attention = 128
            self.heads = 4
        else:
            self.feature_backbone_dim = 2048
            self.dim_attention = 256
            self.heads = 4
        
        self.inner_dim = self.dim_attention *  self.heads
        
        self.path_reduce = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1),
            nn.Linear(self.feature_backbone_dim, self.inner_dim * 3, bias = False)
        )

        
        self.soft = nn.Softmax(dim = -1)

        
        self.to_mask = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim_attention),
            Rearrange('b (h w) c -> b c h w', h = 7, w = 7),
            nn.Conv2d(self.dim_attention, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        #--------ATTENTION
       
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)
        
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if flag:

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.class_classifier(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            
            # ---------------------------- spatial -----------------------           
            spatial_drop_num = math.ceil(HW * self.drop_mask)
            th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
            mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                        torch.ones(spatial_mean.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)


            # ----------------------------------- batch ----------------------------------------
#             cls_prob_before = F.softmax(output, dim=1)
#             x_new_view_after = x_new * mask_all
#             x_new_view_after = self.avgpool(x_new_view_after)
#             x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
#             x_new_view_after = self.class_classifier(x_new_view_after)
#             cls_prob_after = F.softmax(x_new_view_after, dim=1)

#             sp_i = torch.ones([2, num_rois]).long()
#             sp_i[0, :] = torch.arange(num_rois)
#             sp_i[1, :] = index
#             sp_v = torch.ones([num_rois])
#             one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
#             before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
#             after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
#             change_vector = before_vector - after_vector - 0.0001
#             change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
#             th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent_attention_batch))]
#             drop_index_fg = change_vector.gt(th_fg_value).long()
#             ignore_index_fg = 1 - drop_index_fg
#             not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
#             print(not_01_ignore_index_fg)

#             mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()

            mask_all = Variable(mask_all, requires_grad=True)
            
            self.mask_all = mask_all
            
        
        qkv = self.path_reduce(x).chunk(3, dim = -1) # b num_path c
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = self.soft(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
       
        self.attention = self.to_mask(out)
        
        x = x * self.attention
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return self.class_classifier(x)


def resnet18(pretrained=True, drop_perc=0.333, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], drop_perc, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, drop_perc=0.333, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], drop_perc, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
