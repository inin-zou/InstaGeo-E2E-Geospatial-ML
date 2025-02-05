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
from typing import Optional, Tuple, Dict, List
import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Block(nn.Module):
    """
    一个简单的 Transformer 块，用于 TinyViT 模型。
    包含：LayerNorm -> MultiheadAttention -> 残差连接 -> LayerNorm -> MLP -> 残差连接
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4, qkv_bias: bool = True):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 假设输入 x 的形状为 [B, L, C]，这里需要将其转置为 [L, B, C] 以符合 MultiheadAttention 的要求
        residual = x
        x = self.norm1(x)
        x = x.transpose(0, 1)  # [L, B, C]
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.transpose(0, 1)  # 转回 [B, L, C]
        x = x + residual  # 残差连接

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual  # 残差连接
        return x


# NEW: 轻量级学生模型
class TinyViT(nn.Module):
    """Lightweight ViT for knowledge distillation"""
    def __init__(self, 
                 embed_dim: int = 256, 
                 depth: int = 6,
                 num_heads: int = 8,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 21):  # 确保最终输出通道数等于类别数
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = nn.Conv2d(6 * 3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 3 * (image_size // patch_size) ** 2, embed_dim)
        )

        
        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True)  # 确保 Block 正确实现
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # 轻量级上采样模块（用于生成语义分割 mask）
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)  # 最终输出通道数等于类别数
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass for TinyViT student model."""
        batch_size = img.shape[0]

        # 处理 `num_frames` 维度，将 `[B, C, T, H, W]` 变为 `[B, C*T, H, W]`
        if img.ndim == 5:
            img = img.view(batch_size, -1, img.shape[3], img.shape[4])  # `[B, C*T, H, W]`

        # Patch embedding
        x = self.patch_embed(img)  # 现在 img 形状是 `[B, C*T, H, W]`
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # **修正 `pos_embed` 形状**
        pos_embed = self.pos_embed[:, : x.shape[1], :]

        # Add positional encoding
        x = x + pos_embed  # 现在维度匹配

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.norm(x)

        # Reshape回到图像空间
        feature_size = int(self.num_patches ** 0.5)
        x = x.permute(0, 2, 1).reshape(batch_size, -1, feature_size, feature_size)  # (B, embed_dim, H/P, W/P)

        # 上采样以匹配原始图像分辨率
        out = self.segmentation_head(x)  # (B, num_classes, H, W)

        return out




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
        self.num_classes = num_classes
        self.use_distill = use_distill
        self.image_size = image_size
        
        # 初始化教师模型
        self.teacher = self._init_teacher(
            temporal_step, image_size, freeze_backbone
        )
        
        # 确保 `self.model_args` 已正确赋值
        decoder_embed_dim = self.model_args.get("decoder_embed_dim", 512)  # 默认 512
        
        
        # 解决 segmentation_head 未定义问题
        self.segmentation_head = nn.Sequential(
            *[self._upscaling_block(decoder_embed_dim // (2**i), 
                                    decoder_embed_dim // (2**(i+1))) for i in range(4)],
            nn.Conv2d(decoder_embed_dim // 16, self.num_classes, kernel_size=1)
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
            
    def _upscaling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """上采样模块"""
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
            1, (temporal_step * (image_size // 16) ** 2) + 1, self.model_args["embed_dim"]
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
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            *[upscaling_block(512 // (2**i), 512 // (2**(i+1))) for i in range(4)],
            nn.Conv2d(512 // 16, self.num_classes, kernel_size=1)
        )
    
    # NEW: 蒸馏损失函数
    def distill_loss(self, student_out, teacher_out, labels, temp=3.0, alpha=0.7):
        labels = labels.long()
        teacher_probs = F.softmax(teacher_out / temp, dim=1)
        student_log_probs = F.log_softmax(student_out / temp, dim=1)

        # KL 散度
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp**2)

        # 交叉熵损失（显式传递权重）
        ce_loss = F.cross_entropy(
            student_out, 
            labels, 
            weight=self.teacher_criterion.weight  # 使用教师模型的权重
        )

        return alpha * kl_loss + (1 - alpha) * ce_loss


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.use_distill and self.student is not None:
            # 学生模式
            features = self.student(img)
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
            {nn.Linear},
            dtype=dtype
        )

    # NEW: 参数冻结控制
    def set_requires_grad(self, model_part: str, requires_grad: bool):
        """控制不同部分的梯度计算"""
        if model_part == 'teacher':
            for param in self.teacher.parameters():
                param.requires_grad = requires_grad
        elif model_part == 'student' and self.student is not None:
            for param in self.student.parameters():
                param.requires_grad = requires_grad
        elif model_part == 'distill_head':
            for param in self.distill_head.parameters():
                param.requires_grad = requires_grad

    # NEW: 导出ONNX
    def export_onnx(self, student_model=True, output_path="model.onnx"):
        dummy_input = torch.randn(1, 6, self.image_size, self.image_size)
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