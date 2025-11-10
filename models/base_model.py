#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import sys
import math
import os
import json
from dataclasses import dataclass
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextTransformer
import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import deprecate, BaseOutput, USE_PEFT_BACKEND, unscale_lora_layers
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn
from utils import *

class FixedSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, num_patches, embed_dim)

    def forward(self, x):
        return x + self.pe


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.position_encoding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.position_encoding

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        pe = torch.zeros(channels, height, width)
        for h in range(height):
            for w in range(width):
                pos = h * width + w
                for i in range(0, channels, 2):

                    scale_factor = torch.tensor(10000 ** (i / channels), dtype=torch.float32)
                    pe[i, h, w] = torch.sin(torch.tensor(pos, dtype=torch.float32) / scale_factor)
                    if i + 1 < channels:
                        pe[i + 1, h, w] = torch.cos(torch.tensor(pos, dtype=torch.float32) / scale_factor)

        # register as untrained buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class IdentityDecouplingModule(nn.Module):
    def __init__(self, d_identity, num_identity_features=2,layer_idx=0,hidden_dim=128):
        super(IdentityDecouplingModule, self).__init__()
        dim_config = {
            0: 1280,
            1: 640,
            2: 320
        }
        self.hidden_dim = hidden_dim
        assert layer_idx in dim_config, f"Unsupported idx: {layer_idx}"

        out_dim = dim_config[layer_idx]
        self.mlp = nn.Linear(d_identity * num_identity_features, self.hidden_dim)
        self.linear_projection = nn.Linear(d_identity * num_identity_features, out_dim)
        self.W_Q = nn.Linear(out_dim, self.hidden_dim)
        self.W_K = nn.Linear(out_dim, self.hidden_dim)
        self.W_V = nn.Linear(out_dim, self.hidden_dim)
        self.W_O = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, z, C_id):
        """
        :param z: Feature representation, with shape (batch_size, 1280, 32, 32)
        :param C_id: Identity feature, with shape (batch_size, 2, 768)
        :return: Identity-decoupled feature representation

        """
        # 1. flatten z (batch_size, 1280, 32, 32) as (batch_size, 1280, 32 * 32)
        check_nan(z, 'input z')
        batch_size, channels, height, width = z.shape
        z_flattened = z.view(batch_size, channels, height * width).permute(0, 2, 1)
        check_nan(z_flattened, 'z_flattened')
        z_flattened = F.layer_norm(z_flattened, (z_flattened.size(-1),))

        Q = self.W_Q(z_flattened)  # (batch_size, 1280, d_feature)
        K = self.W_K(z_flattened)  # (batch_size, 1280, d_feature)
        V = self.W_V(z_flattened)  # (batch_size, 1280, d_feature)


        if isinstance(C_id, list):
            C_id = torch.cat(C_id, dim=1)  # (batch_size, 2, 768)


        # 3.  C_id = (batch_size, d_identity * num_identity_features)
        C_id_flattened = C_id.view(C_id.size(0), -1)  # (batch_size, 2 * 768)
        # 4. M_ID

        identity_vector = self.mlp(C_id_flattened)  # [B, d]
        identity_vector = identity_vector.unsqueeze(1).repeat(batch_size, 1, 1)  # [B, 1, d] # (batch_size, d_feature)
        mask_logits = torch.bmm(identity_vector, K.transpose(1, 2))  # [B, 1, L]
        mask = torch.sigmoid(mask_logits)

        # 5. attention
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(1))
        attn_logits = attn_logits * mask
        A = torch.softmax(attn_logits, dim=-1)

        # 6. residual calculation
        C_id_projected = self.linear_projection(C_id_flattened)
        check_nan(C_id_projected, 'C_id_projected')

        # 7. identity decoupled feature
        attended = torch.matmul(A, V)  #(batch_size, 1280, 32 * 32)
        attended = self.W_O(attended)
        sim = torch.sum(attended * C_id_projected, dim=-1, keepdim=True)  # [B, L, 1]
        weight = torch.sigmoid(sim)  # [B, L, 1]
        attended_filtered = attended - weight * C_id_projected  # [B, L, d_v]
        # 8. layer norm
        output = F.layer_norm(attended_filtered, attended_filtered.size()[1:])

        return output.view(batch_size, channels, height, width)  # (batch_size, 1280, 32, 32)

class AuxiliaryModel(nn.Module):
    def __init__(self, input_dim, temb_dim, temb_out_channels, patch_size=4, token_dim=768,scale=32,idx=0,use_idm=False):
        super(AuxiliaryModel, self).__init__()
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.use_idm = use_idm
        if temb_dim is not None:
            self.time_emb_proj = nn.Linear(temb_dim, temb_out_channels)
            self.nonlinearity = nn.SiLU()

        self.depthwise_conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.pointwise_conv1 = nn.Conv2d(input_dim, 512, kernel_size=1)  # dot conv
        self.conv_in1 = nn.InstanceNorm2d(512)
        self.gelu1 = nn.GELU()

        self.depthwise_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=256)
        self.pointwise_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_in2 = nn.InstanceNorm2d(256)
        self.gelu2 = nn.GELU()
        self.patch_embedding = nn.Linear(256 * self.patch_size * self.patch_size, self.token_dim)
        if self.use_idm:
            self.idm = IdentityDecouplingModule(d_identity=self.token_dim,layer_idx=idx)
    def forward(self, x, temb,id_emb):
        if temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            x = x + temb
        if self.use_idm:
            x = self.idm(x,id_emb)

        # First depthwise separable convolution layer
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.conv_in1(x)
        x = self.gelu1(x)
        # Second depthwise separable convolution layer
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.conv_in2(x)
        x = self.gelu2(x)
        batch_size, channels, height, width = x.shape
        # Patchify x

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4,
                      5).contiguous()  # (batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
        x = x.view(batch_size, -1,
                   channels * self.patch_size * self.patch_size)  # (batch_size, num_patches, patch_dim)
        x = self.patch_embedding(x)
        return x

class DualPredictionHead(nn.Module):
    def __init__(self, token_dim, num_layers, num_labels):
        super(DualPredictionHead, self).__init__()
        self.token_dim = token_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.token_fc_layers_det = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.token_dim * self.num_layers, self.token_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.token_dim // 2, 1)
            ) for _ in range(self.num_labels)
        ])
        self.token_fc_layers_reg = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.token_dim * self.num_layers, self.token_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.token_dim // 2, 1)
            ) for _ in range(self.num_labels)
        ])

    def forward(self, x):
        logits_list_det = []
        for i, fc in enumerate(self.token_fc_layers_det):
            token_logits = fc(x[:, i, :])  # (batch_size, token_dim)
            logits_list_det.append(token_logits)

        # (batch_size, num_labels)
        logits_det = torch.stack(logits_list_det, dim=1).squeeze(2)
        logits_list_reg = []
        for i, fc in enumerate(self.token_fc_layers_reg):
            token_logits = fc(x[:, i, :])  # (batch_size, token_dim)
            logits_list_reg.append(token_logits)

        # (batch_size, num_labels)
        logits_reg = torch.stack(logits_list_reg, dim=1).squeeze(2)
        logits_reg = torch.sigmoid(logits_reg)

        return logits_det, logits_reg

class AUTransformer(nn.Module):
    def __init__(self, num_labels, token_dim, use_LGFE):
        super(AUTransformer, self).__init__()
        self.use_LGFE = use_LGFE
        self.num_labels = num_labels
        self.num_learnable_tokens = num_labels
        self.token_dim = token_dim
        self.num_layers = 1
        self.hidden_size = 512
        self.norm_layers_dec_1 = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        self.norm_layers_dec_2 = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        if self.use_LGFE:
            self.norm_layers_dec_3 = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        self.norm_layers_enc_1 = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        self.norm_layers_enc_2 = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        self.feedforward_layers_dec = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.token_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.token_dim)
            ) for _ in range(self.num_layers)
        ])
        self.feedforward_layers_enc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.token_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.token_dim)
            ) for _ in range(self.num_layers)
        ])
        self.attention_layers_enc = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=8, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.attention_layers_1 = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=8, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.attention_layers_2 = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=8, batch_first=True)
            for _ in range(self.num_layers)
        ])
        if self.use_LGFE:
            self.attention_layers_text = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ])

    def forward(self, x, x_pos, learnable_tokens, encoder_hidden_states):
        for i in range(self.num_layers):
            residual_input = x
            x, _ = self.attention_layers_enc[i](x, x_pos, x_pos)
            x = self.norm_layers_enc_1[i](x + residual_input)
            residual_input = x
            x = self.feedforward_layers_enc[i](x)
            x = self.norm_layers_enc_2[i](x + residual_input)

            residual_tokens = learnable_tokens
            learnable_tokens, _ = self.attention_layers_1[i](learnable_tokens, learnable_tokens,
                                                             learnable_tokens)  # self attention
            learnable_tokens = self.norm_layers_dec_1[i](learnable_tokens + residual_tokens)
            residual_tokens = learnable_tokens
            learnable_tokens, _ = self.attention_layers_2[i](learnable_tokens, x, x)  # Cross attention
            learnable_tokens = self.norm_layers_dec_2[i](learnable_tokens + residual_tokens)  # Add & Norm
            residual_tokens = learnable_tokens
            learnable_tokens = self.feedforward_layers_dec[i](learnable_tokens)  # Add & Norm with Feedforward
            if self.use_LGFE:
                learnable_tokens = self.norm_layers_dec_3[i](learnable_tokens + residual_tokens)
                learnable_tokens, _ = self.attention_layers_text[i](learnable_tokens, encoder_hidden_states,
                                                                    encoder_hidden_states)

        return learnable_tokens

class CustomUNet(nn.Module):

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def __init__(self,
                 unet_base,
                 num_labels=12,
                 dtype=torch.float16,
                 id_head=False,
                 lang_init=False,
                 au_cond=True,
                 use_LGFE=True,
                 aux_loss=False,
                 pe_learnable=False,
                 token_dim=768,
                 scale_list = [32, 64, 128, 256],
                 lang_init_model="ruwnayml/stable-diffusion-inpainting",
                 use_idm=False,
                 train=False,
                 ):
        super(CustomUNet, self).__init__()
        self.unet = unet_base
        self.id_head = id_head
        self.num_labels = num_labels
        self.lang_init = lang_init
        self.au_cond = au_cond
        self.use_LGFE = use_LGFE
        self.aux_loss = aux_loss
        self.pe_learnable = pe_learnable
        self.token_dim = token_dim
        self.scale_list = scale_list
        self.config = self.unet.config
        self.use_idm = use_idm
        self.train = train
        if self.train:
            if self.lang_init:
                self.learnable_tokens = language_init(lang_init_model)
            else:
                self.learnable_tokens = nn.Parameter(torch.randn(self.num_labels, self.token_dim))
        else:
            self.learnable_tokens = None
        self.total_input_dim, self.up_dim_list = self._calculate_total_input_dim(self.unet.config)
        self.num_layers = len(self.up_dim_list)


        if self.pe_learnable:
            self.position_embedding = nn.ModuleList(
                [LearnablePositionalEncoding(num_patches=64 if i == 0 else 256, embed_dim=self.token_dim) for i in
                 range(self.num_layers)])
        else:
            self.position_embedding = nn.ModuleList(
                [FixedSinusoidalPositionalEncoding(num_patches=64 if i == 0 else 256, embed_dim=self.token_dim) for
                 i in
                 range(self.num_layers)])
        self.auxiliary_head = nn.ModuleList(
            [AuxiliaryModel(input_dim=self.up_dim_list[i], temb_dim=1280, temb_out_channels=self.up_dim_list[i],
                            token_dim=self.token_dim,scale=self.scale_list[i],idx=i, use_idm=self.use_idm) for i in range(self.num_layers)])
        self.au_transformer = nn.ModuleList(
            [AUTransformer(num_labels, self.token_dim, self.use_LGFE) for _ in range(self.num_layers)])

        self.dual_prediction_head = DualPredictionHead(self.token_dim, 1, self.num_labels)
        self.image_branch_last_layer = None
        self.label_branch_last_layer = None

    def _calculate_total_input_dim(self, config):
        """
        Calculate the sum of the output dimensions of all Cross-Attention layers.
        The output dimensions of each layer need to be obtained from `config` and accumulated.
        """
        cross_attention_layer_types = ["CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"]
        total_output_dim = 0
        total_output_dim_list = []
        #  `down_block_types`  `up_block_types`:compute output dimensions
        for idx, block_type in enumerate(config.down_block_types):
            if block_type in cross_attention_layer_types:
                total_output_dim += config.block_out_channels[idx]

        for idx, block_type in enumerate(config.up_block_types):
            if block_type in cross_attention_layer_types:
                total_output_dim += config.block_out_channels[-(idx + 1)]
                total_output_dim_list.append(config.block_out_channels[-(idx + 1)])
        # check Cross-Attention layer
        if "UNetMidBlock2DCrossAttn" in cross_attention_layer_types and config.mid_block_type == "UNetMidBlock2DCrossAttn":
            total_output_dim += config.block_out_channels[-1]

        return total_output_dim, total_output_dim_list

    def get_unet(self):
        return self.unet

    def get_cls_head(self):
        return self.classification_head

    def get_projection(self):
        return self.projection_layer

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # save model
        torch.save(self.state_dict(), os.path.join(save_directory, "diffusion_pytorch_model.bin"))
        # save config
        config_path = os.path.join(save_directory, "config.json")
        if hasattr(self, "config") and self.config is not None:
            config_dict = dict(self.config)

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=4)

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            id_emb: Optional[list] = None,
            AU_cue: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        """
        Simplified `forward` method of the UNet2DConditionModel.

        Args:
            sample (`torch.Tensor`):
                Input noise tensor with shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`):
                The timestep in the denoising process.
            encoder_hidden_states (`torch.Tensor`):
                Encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return the output as a dictionary.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, returns a `UNet2DConditionOutput` dictionary; otherwise, returns a tuple.
        """

        default_overall_up_factor = 2 ** self.unet.num_upsamplers
        forward_upsample_size = False

        cross_attention_features = []
        logits_det_tmp = []
        logits_reg_tmp = []

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break


        if self.unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time emb
        t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        emb = self.unet.time_embedding(t_emb)

        encoder_hidden_states = self.unet.process_encoder_hidden_states(encoder_hidden_states=encoder_hidden_states,
                                                                        added_cond_kwargs=added_cond_kwargs)
        # 2. input preprocess
        sample = self.unet.conv_in(sample)
        # 3. down_blocks
        down_block_res_samples = (sample,)
        if self.au_cond:
            encoder_hidden_states = torch.cat([encoder_hidden_states, AU_cue], dim=1)
        for block_idx, downsample_block in enumerate(self.unet.down_blocks):
            if type(downsample_block).__name__ == "CrossAttnDownBlock2D":

                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid_block
        if self.unet.mid_block is not None:
            if type(self.unet.mid_block).__name__ == "UNetMidBlock2DCrossAttn":
                sample = self.unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
                cross_attention_features.append(sample)  # keep mid block features
            else:
                sample = self.unet.mid_block(sample, emb)
        batch_size, channel, height, width = sample.shape
        if self.learnable_tokens is None:
            learnable_tokens = AU_cue
            # learnable_tokens = learnable_tokens.repeat(batch_size, 1, 1)
        else:
            learnable_tokens = self.learnable_tokens
            learnable_tokens = learnable_tokens.unsqueeze(0).repeat(batch_size, 1, 1)

        encoder_hidden_states_residual = encoder_hidden_states
        # 5. up_blocks
        for i, upsample_block in enumerate(self.unet.up_blocks):
            is_final_block = i == len(self.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]


            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            else:
                upsample_size = None

            if type(upsample_block).__name__ == "CrossAttnUpBlock2D":

                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
                sample_cls = self.auxiliary_head[i - 1](sample, emb, id_emb)
                sample_pos = self.position_embedding[i - 1](sample_cls)

                learnable_tokens = self.au_transformer[i - 1](sample_cls, sample_pos, learnable_tokens,
                                                              encoder_hidden_states_residual)
                residual = learnable_tokens
                if self.aux_loss:
                    det_tmp, reg_tmp = self.dual_prediction_head(learnable_tokens)
                    logits_det_tmp.append(det_tmp)
                    logits_reg_tmp.append(reg_tmp)
                cross_attention_features.append(sample)

            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        self.image_branch_last_layer = sample
        # 6. post process
        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        # DPH
        self.label_branch_last_layer = sample
        logits_det, logits_reg = self.dual_prediction_head(learnable_tokens)
        learnable_tokens_cur = learnable_tokens.detach()
        if not return_dict:
            model_output = (sample, logits_det, logits_reg, learnable_tokens_cur)
        else:
            model_output = UNet2DConditionOutput(sample=sample)
            model_output['logits_det'] = logits_det
            model_output['logits_reg'] = logits_reg
            model_output['au_tokens'] = learnable_tokens
            if self.aux_loss:
                model_output['logits_det_tmp'] = logits_det_tmp
                model_output['logits_reg_tmp'] = logits_reg_tmp
        return model_output
