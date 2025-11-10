import sys
import time
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import CLIPTextModel, CLIPTokenizer

def language_init(pretrained_model_name_or_path):
    au_list = ["Inner Brow Raiser", "Outer Brow Raiser", "Brow Lowerer", "Upper Lid Raiser", "Cheek Raiser",
               "Nose Wrinkler",
               "Lip Corner Puller", "Lip Corner Depressor", "Chin Raiser", "Lip Stretcher", "Lips Part", "Jaw Drop"]

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer",
                                                       revision=None)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                          revision=None)
    initial_embs = []
    for i in range(len(au_list)):
        print(f"{au_list[i]} emb")
        encoded_inputs = tokenizer(au_list[i], return_tensors='pt',
                                        max_length=tokenizer.model_max_length, truncation=True,
                                        padding='max_length')
        text_emb = text_encoder(input_ids=encoded_inputs['input_ids'],
                                  attention_mask=encoded_inputs['attention_mask'],
                                 return_dict=False)[1]
        initial_embs.append(text_emb.clone().detach())
        stacked_embs = torch.stack(initial_embs, dim=1).squeeze()

        learnable_tokens = nn.Parameter(stacked_embs).to('cuda')
    return learnable_tokens

def check_nan(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"❗ NaN or Inf in {name} | shape: {x.shape}")
        return True
    return False

def register_attention_hook(unet, container):
    def hook_fn(module, input, output):
        container.append(output.detach().cpu())

    for name, module in unet.named_modules():
        if name.endswith("up_blocks.3.attentions.2.transformer_blocks.0.attn2"):
            module.register_forward_hook(hook_fn)
            print(f"Hook registered to: {name}")


def eval_metrics_tensor(y_true, y_pred):
    y_true = torch.clamp(y_true, max=1)
    # TP, FP, FN
    TP = torch.sum((y_pred == 1) & (y_true == 1), dim=1).float()
    FP = torch.sum((y_pred == 1) & (y_true == 0), dim=1).float()
    FN = torch.sum((y_pred == 0) & (y_true == 1), dim=1).float()
    TN = torch.sum((y_pred == 0) & (y_true == 0), dim=1).float()
    # Precision, Recall
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    return accuracy, precision, recall, f1


def calculate_clip_score(image_path, text, processor, model, device):
    """
    calculate CLIP Score。

    Params:
    - image_path (str):
    - text (str):

    Return:
    - clip_score (float): CLIP Score
    """
    image = Image.open(image_path)
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # normalize
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    #（CLIP Score）
    clip_score = (image_embeds * text_embeds).sum(dim=-1).item()
    return clip_score

def read_lines_as_tensors(file_path):
    tensors = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().strip('[]')
            labels = list(map(int, line.split(', ')))
            tensors.append(torch.tensor(labels))
    return torch.stack(tensors)


def save_tensor_as_png(tensor, output_dir):
    batch_size = tensor.size(0)

    tensor = tensor.squeeze(1)  #  -> [batch, 256, 256]
    tensor = tensor * 255.0  #
    tensor = tensor.byte()  #  uint8

    for i in range(batch_size):
        img = tensor[i].cpu().numpy()  # NumPy
        img_pil = Image.fromarray(img, mode='L')  # gray image
        img_pil.save(output_dir)

def load_image_paths(txt_file, prefix=None):
    if prefix is None:
        with open(txt_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        with open(txt_file, 'r') as f:
            return [prefix + '/' + line.strip() for line in f if line.strip()]

def save_metrics_to_txt(epoch, file_path, accuracy, precision, recall, f1, true_label, gen_label, label_intensity,
                        prompt=None, prompt_idx=None):

    current_time = datetime.datetime.now()

    with open(file_path, 'a') as f:
        f.write(f'current_time {current_time}\n')
        f.write(f'Epoch {epoch}\n')
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"True Label: {true_label}\n")
        f.write(f"Gen. Label: {gen_label}\n")
        f.write(f"Label intensity: {label_intensity}\n")
        if prompt_idx is not None:
            f.write(f"Prompt idx: {prompt_idx}\n")
        if prompt is not None:
            f.write(f"prompt: {prompt} \n")

        f.write("-" * 30 + "\n")

def calculate_mean(lst):
    return sum(lst) / len(lst) if len(lst) > 0 else 0

def eval_metrics(y_true, y_pred):
    if isinstance(y_true, str):
        y_true = eval(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.nan_to_num(y_pred, nan=0)
    y_true[y_true > 1] = 1
    accuracy_tmp = accuracy_score(y_true, y_pred)
    precision_tmp = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_tmp = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_tmp = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy_tmp, precision_tmp, recall_tmp, f1_tmp

def resize_and_concat_features(features, method="interpolate"):
    """
    Adjust multiple feature maps to the same size and concatenate them.

    Args:
        features (list of torch.Tensor): List of original feature maps.
        method (str): The adjustment method, which can be "interpolate" (upsampling/downsampling), "pooling" (pooling), or "padding" (padding).

    Returns:
        torch.Tensor: The concatenated feature map.
    """
    max_height = max(feat.shape[2] for feat in features)
    max_width = max(feat.shape[3] for feat in features)

    if method == "interpolate":
        resized_features = [F.interpolate(feat, size=(max_height, max_width), mode="bilinear", align_corners=False) for
                            feat in features]
    elif method == "pooling":
        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        resized_features = [global_avg_pool(feat) for feat in features]
    elif method == "padding":
        resized_features = [pad_to_shape(feat, (max_height, max_width)) for feat in features]
    else:
        raise ValueError(f"Unsupported method: {method}")

    fused_feature = torch.cat(resized_features, dim=1)
    return fused_feature

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

# pretrained InceptionV3, `pool3` output
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self,local_model_path=None):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = models.inception_v3(pretrained=False, transform_input=False)
        if local_model_path:
            state_dict = torch.load(local_model_path)
            inception.load_state_dict(state_dict)
        # `pool3`
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c
        )
    def forward(self, x):
        # extract `pool3` output
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.squeeze(-1).squeeze(-1)



# for InceptionV3 input
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # [batch_size, channels, height, width]


def extract_features(image_paths, model, device):
    model.eval()
    features = []

    with torch.no_grad():
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image = preprocess_image(image).to(device)  #  GPU/CPU
                # dimension alignment
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                feature = model(image)
                features.append(feature.cpu().numpy())

            except UnidentifiedImageError:
                print(f"Skipping unidentifiable image: {image_path}")
                continue
    return np.concatenate(features, axis=0)


# mean and cov
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
        Calculate the Fréchet Distance between two distributions.

        Args:
            mu1 (numpy.ndarray): Mean vector of the first distribution.
            sigma1 (numpy.ndarray): Covariance matrix of the first distribution.
            mu2 (numpy.ndarray): Mean vector of the second distribution.
            sigma2 (numpy.ndarray): Covariance matrix of the second distribution.

        Returns:
            float: The Fréchet Distance between the two distributions.
        """
    diff = mu1 - mu2

    # Compute the square root of the covariance matrices' product
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)

# FID
def calculate_fid(real_image_paths, generated_image_paths, device='cuda'):

    model = InceptionV3FeatureExtractor(local_model_path='/inception_v3.pth').to(device)

    # real
    real_features = extract_features(real_image_paths, model, device)
    mu_real, sigma_real = calculate_statistics(real_features)

    # generated
    generated_features = extract_features(generated_image_paths, model, device)
    mu_generated, sigma_generated = calculate_statistics(generated_features)

    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

    del model
    torch.cuda.empty_cache()

    return fid_value

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_id_embeddings(id_image_path,resolution,msid,img2text,device):
    id_image = cv2.imread(id_image_path)
    id_image = Image.fromarray(cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB))
    ID_transforms = transforms.Compose(
        [
            transforms.Resize(resolution,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    id_image = ID_transforms(id_image)
    id_image = id_image.to(memory_format=torch.contiguous_format).float()
    id_image = id_image.unsqueeze(0)
    id_image = torch.nn.functional.interpolate(id_image,
                                               size=(112, 112),
                                               mode='bilinear',
                                               align_corners=False).to(device)
    idvec = msid.extract_mlfeat(id_image.to(device).float(), [2, 5, 8, 11])
    tokenized_identity_first, tokenized_identity_last = img2text(idvec, exp=None)
    return [tokenized_identity_first, tokenized_identity_last]

def mask_preprocess(mask_path,masked_image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask = cv2.imread(mask_path)
    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    mask = transform(mask)
    mask = (mask + 1) / 2

    masked_image = cv2.imread(masked_image_path)
    masked_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    masked_image = transform(masked_image)

    return mask, masked_image

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):

        mse_loss = (prediction - target) ** 2

        focal_factor = (1 - torch.exp(-mse_loss)) ** self.gamma

        focal_mse_loss = focal_factor * mse_loss

        if self.reduction == 'mean':
            return focal_mse_loss.mean()
        elif self.reduction == 'sum':
            return focal_mse_loss.sum()
        else:
            return focal_mse_loss

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Binary cross entropy with logits combines the sigmoid activation and BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # p_t, i.e., the probability of predicting the correct label for each class
        pt = torch.exp(-bce_loss)  # p_t = sigmoid(logits) for binary classification

        #Focal Loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss