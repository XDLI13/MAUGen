import torch
from torch import nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import numpy as np
import cv2
from torch import nn
import argparse
import torchvision
import face_alignment
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from .mod import *

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor, mean, std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return tensor * std + mean


class IMG2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.i2t_first = IMG2TEXT_MLP(embed_dim, middle_dim, output_dim, n_layer, dropout)
        self.i2t_last = IMG2TEXT_MLP(embed_dim, middle_dim, output_dim, n_layer, dropout)

    def forward(self, x: torch.Tensor):
        first = self.i2t_first(x)
        last = self.i2t_last(x)

        return first, last


class IMG2TEXT_MLP(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.LeakyReLU(0.2))
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class IMG2TEXTwithEXP(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1, exp_dim=64):
        super().__init__()
        self.i2t_first = IMG2TEXT_MLP(embed_dim + exp_dim, middle_dim, output_dim, n_layer, dropout)
        self.i2t_last = IMG2TEXT_MLP(embed_dim + exp_dim, middle_dim, output_dim, n_layer, dropout)
        self.null_exp = torch.nn.Parameter(torch.zeros([exp_dim]))

    def forward(self, x, exp=None, mask=None):
        if exp is None:
            exp = self.null_exp.unsqueeze(0)
        if mask is None:
            mask = torch.ones((len(x),), device=x.device)
        mask = mask.reshape((-1, 1))
        exp = exp * mask + self.null_exp.unsqueeze(0) * (1 - mask)
        x = torch.cat([x, exp], -1)
        first = self.i2t_first(x)
        last = self.i2t_last(x)

        return first, last

    def register_to_config(self, **kwargs):
        pass

class MSIDEncoder(VisionTransformer):
    def __init__(self, ext_depthes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.head = None

        trunc_normal_(self.pos_embed, std=.02)

        self.ext_depthes = ext_depthes
        self.norms = nn.ModuleDict(
             {f'layer_{i}': nn.Sequential(nn.BatchNorm1d(self.embed_dim), nn.LayerNorm(self.embed_dim)) for i in
              self.ext_depthes}
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.ext_depthes:
                feats.append(self.norms[f'layer_{i}'](x[:, 0]))
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        feats = torch.cat(feats, 1)  # [feat_1,feat_2,...] -> (B,D*T)
        return feats

    def extract_mlfeat(self, x, ext_depthes):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in ext_depthes:
                if f'layer_{i}' in self.norms:
                    # print(x[:, 0].shape)
                    y = self.norms[f'layer_{i}'](x[:, 0])
                else:
                    y = x[:, 0]
                feats.append(y)
        feats = torch.cat(feats, 1)
        return feats


@register_model
def msid_base_patch8_112(pretrained=False, ext_depthes=[11], **kwargs):
    model = MSIDEncoder(ext_depthes=ext_depthes,
                        img_size=112, patch_size=8, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

