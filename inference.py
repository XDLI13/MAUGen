import sys
import argparse
import time
import logging
import math
import os
import random
import json
from pathlib import Path
import cv2
import PIL
from PIL import Image
import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextTransformer

import diffusers
from diffusers import UNet2DConditionModel

from torchvision.transforms import ToTensor
from utils import *
from models.pipeline import *
from models.msid import IMG2TEXTwithEXP, MSIDEncoder, msid_base_patch8_112
from models.mod import *


def main(args):
    tqdm_bar = tqdm(total=args.num_validation_images)
    device = args.device
    generator = None if args.seed is None else torch.Generator(device=args.device).manual_seed(args.seed)

    negative_prompt = (
        '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, '
        'anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, '
        'jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, '
        'poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, '
        'bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, '
        'missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, '
        'uneven skin texture, rough skin, noisy skin, patchy cheeks, unnatural lighting, blotchy color, red spots, '
        'discoloration, uneven tone, overexposed skin, deformed face, asymmetric cheeks, distorted features, '
        'unnatural proportions, sunken cheeks, plastic-like skin, overly smooth skin, waxy texture, overly edited '
        'face, harsh shadows, patchy texture, artifact, unrealistic lighting, unnatural cheek blush, uncanny valley,'
        'excessive frown lines, overly deep forehead wrinkles,'
        'unnaturally harsh forehead creases, excessively visible facial stress lines, overly defined horizontal lines '
        'on the forehead, over-exaggerated facial expressions, unnatural deep wrinkles, overly harsh shadows on skin, '
        'unnatural texture exaggeration, asymmetrical wrinkle distribution, overly strained facial features, '
        'exaggerated skin folds)'
    )




    if args.ID_head:
        img2text = IMG2TEXTwithEXP(384 * 4, 384 * 4, 768)
        img2text.load_state_dict(torch.load(args.w_map, map_location='cpu'))
        img2text = img2text.to(device)
        img2text.eval()

        msid = msid_base_patch8_112(ext_depthes=[2, 5, 8, 11])
        msid.load_state_dict(torch.load(args.w_msid))
        msid = msid.to(device)
        msid.eval()
        id_emb = extract_id_embeddings(args.id_image_path, args.resolution, msid, img2text, device)
    else:
        id_emb = None


    mask, masked_image = mask_preprocess(args.mask_path, args.masked_image_path)
    # load AU cue
    AU_cue = torch.load(args.AU_cue_load_path)
    # load unet
    unet_base = UNet2DConditionModel.from_pretrained(
        args.aux_model_dir, subfolder="unet", revision=None,
        variant=None
    )
    unet = CustomUNet(unet_base=unet_base,
                      num_labels=args.num_labels,
                      lang_init=args.lang_init,
                      au_cond=args.au_cond,
                      use_LGFE=args.use_LGFE,
                      lang_init_model=args.aux_model_dir,
                      use_idm=args.use_idm,
                      ).to(device)

    state_dict = torch.load(args.model_path)
    # filtered_state_dict = {k: v for k, v in state_dict.items() if k in unet.state_dict()}

    unet.load_state_dict(state_dict, strict=False)



    # define pipeline
    tokenizer = CLIPTokenizer.from_pretrained(
        args.aux_model_dir, subfolder="tokenizer", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.aux_model_dir, subfolder="text_encoder", revision=None
    )
    special_tokens = {"additional_special_tokens": ["<ID>", "<ID2>"]}
    tokenizer.add_special_tokens(special_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.text_model.forward = forward_texttransformer.__get__(text_encoder.text_model,
                                                                      CLIPTextTransformer)
    text_encoder.forward = forward_textmodel.__get__(text_encoder, CLIPTextModel)
    weight_dtype = torch.float16
    text_encoder.to(device, dtype=weight_dtype)
    pipeline = CustomStableDiffusionPipeline.from_pretrained(
        args.aux_model_dir,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        revision=None,
    )
    pipeline.safety_checker = None
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    images = []
    for idx in range(args.num_validation_images):
        with torch.autocast("cuda"):
            with torch.no_grad():
                start_time = time.time()
                output_image, _ , output_logits_reg,_,_ = pipeline(prompt=args.prompt,
                                                                                  negative_prompt=negative_prompt,
                                                                                  num_inference_steps=args.num_inference_steps,
                                                                                  generator=generator,
                                                                                  return_dict=False,
                                                                                  id_emb=id_emb,
                                                                                  mask=mask,
                                                                                  masked_image=masked_image,
                                                                                  AU_cue=AU_cue,)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Inference time: {elapsed_time:.4f} seconds")
                # _, cond_output_det = torch.chunk(output_logits_det, 2, dim=0)
                _, cond_output_reg = torch.chunk(output_logits_reg, 2, dim=0)
                # outputs = torch.sigmoid(cond_output_det).cpu().detach().numpy().round()
                images.append(output_image)
                pred_intensity = cond_output_reg * 5.0
                pred_intensity = pred_intensity.cpu().detach().numpy().round()

        image = images[-1]


        image.save(args.save_path)
        print(f"prompt: {args.prompt} \n")
        print(f"label intensity is {pred_intensity.squeeze()}\n")

        tqdm_bar.update(1)
        tqdm_bar.set_description(
            f"idx = {idx:02}, prompt = {args.prompt[:10]} ")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Diffusion-based AU-conditioned face generation")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_labels", type=int, default=12, help="Number of AU labels")
    parser.add_argument("--ID_head", type=bool, default=True, help="Enable ID examplar")
    parser.add_argument("--lang_init", type=bool, default=True, help="Enable language-based initialization")
    parser.add_argument("--au_cond", type=bool, default=True, help="Enable AU conditioning")
    parser.add_argument("--use_LGFE", type=bool, default=True, help="Enable LGFE module")
    parser.add_argument("--use_idm", type=bool, default=True, help="Enable IDM module")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of validation images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps for generation")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained UNet checkpoint")
    parser.add_argument("--AU_cue_load_path", type=str, required=True, help="Path to AU learnable token checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing facial expression")
    parser.add_argument("--save_path", type=str, default="output_img.png", help="Path to save output image")
    parser.add_argument("--id_image_path", type=str, required=True, help="Path to reference ID image")
    parser.add_argument("--aux_model_dir", type=str, required=True, help="Directory to auxiliary SD model")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to binary mask image")
    parser.add_argument("--masked_image_path", type=str, required=True, help="Path to input masked image")
    parser.add_argument("--w_map", type=str, required=True, help="Path to w latent mapping file")
    parser.add_argument("--w_msid", type=str, required=True, help="Path to multi-scale identity file")

    args = parser.parse_args()
    main(args)


