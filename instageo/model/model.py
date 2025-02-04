# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Model Module."""

import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore
from absl import logging

from instageo.model.Prithvi import ViTEncoder


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int):
        """Initializes the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# NEW: 轻量级学生模型
class TinyViT(nn.Module):
    """Lightweight ViT for knowledge distillation"""
    def __init__(self, 
                 embed_dim: int = 256, 
                 depth: int = 6,
                 num_heads: int = 8,
                 image_size: int = 224,
                 patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size//patch_size)**2 + 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # [B, C, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, L, C]
        x = x + self.pos_embed[:, 1:, :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class PrithviSeg(nn.Module):
    """Prithvi Segmentation Model with Knowledge Distillation Support"""
    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        # NEW: 蒸馏相关参数
        use_distill: bool = False,
        student_config: Optional[dict] = None,
    ):
        super().__init__()
        self.use_distill = use_distill
        self.image_size = image_size
        
        # 初始化教师模型
        self.teacher = self._init_teacher(
            temporal_step, image_size, freeze_backbone
        )
        
        # NEW: 初始化学生模型
        self.student = None
        if use_distill:
            student_args = student_config or {
                'embed_dim': 256,
                'depth': 6,
                'num_heads': 8,
                'image_size': image_size,
                'patch_size': 16
            }
            self.student = TinyViT(**student_args)
            self._init_distill_head(num_classes)
            
    def _init_teacher(self, temporal_step, image_size, freeze_backbone):
        # [原有教师模型初始化代码，保持参数加载逻辑不变]
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
        cfg_path = weights_dir / "config.yaml"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]
        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        self.model_args = model_args
        
        model = ViTEncoder(**model_args)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        filtered_checkpoint_state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder.")
        }
        filtered_checkpoint_state_dict["pos_embed"] = torch.zeros(
            1, (temporal_step * (image_size // 16) ** 2 + 1), 768
        )
        model.load_state_dict(filtered_checkpoint_state_dict)
        return model

    # NEW: 初始化蒸馏分割头
    def _init_distill_head(self, num_classes):
        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        embed_dim = self.student.embed_dim if hasattr(self.student, 'embed_dim') else 256
        self.distill_head = nn.Sequential(
            *[upscaling_block(embed_dim//(2**i), embed_dim//(2**(i+1))) for i in range(4)],
            nn.Conv2d(embed_dim//16, num_classes, kernel_size=1)
        )

    # NEW: 蒸馏损失函数
    def distill_loss(self, student_out, teacher_out, labels, temp=3.0, alpha=0.7):
        teacher_probs = F.softmax(teacher_out / temp, dim=1)
        student_log_probs = F.log_softmax(student_out / temp, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp**2)
        ce_loss = F.cross_entropy(student_out, labels)
        return alpha * kl_loss + (1 - alpha) * ce_loss

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.use_distill and self.student is not None:
            # 学生模式
            features = self.student(img)
            features = features.permute(0, 2, 1).reshape(
                img.size(0), -1, 
                self.image_size//self.student.patch_size,
                self.image_size//self.student.patch_size
            )
            return self.distill_head(features)
        else:
            # 教师模式
            features = self.teacher(img)
            reshaped_features = features[:, 1:, :]
            feature_img_side_length = int(
                np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
            )
            reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
                features.shape[0], -1, feature_img_side_length, feature_img_side_length
            )
            return self.segmentation_head(reshaped_features)

    # NEW: 模型压缩方法
    def quantize_model(self, dtype=torch.qint8):
        return torch.quantization.quantize_dynamic(
            self.student,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )

    # NEW: 参数冻结控制
    def set_requires_grad(self, model_part: str, requires_grad: bool):
        """控制不同部分的梯度计算"""
        if model_part == 'teacher':
            for param in self.teacher.parameters():
                param.requires_grad = requires_grad
        elif model_part == 'student':
            for param in self.student.parameters():
                param.requires_grad = requires_grad
        elif model_part == 'distill_head':
            for param in self.distill_head.parameters():
                param.requires_grad = requires_grad

    # NEW: 导出ONNX
    def export_onnx(self, student_model=True, output_path="model.onnx"):
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
        if student_model:
            torch.onnx.export(
                self.student,
                dummy_input,
                output_path,
                opset_version=13,
                input_names=['input'],
                output_names=['output']
            )
        else:
            torch.onnx.export(
                self.teacher,
                dummy_input,
                output_path,
                opset_version=13,
                input_names=['input'],
                output_names=['output']
            )