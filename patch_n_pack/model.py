from transformers import ViTModel
from transformers.models.vit.modeling_vit import ViTEmbeddings
import torch
from typing import Optional
import torch.nn as nn
import collections
import numpy as np
import math
import cv2


class PatchPackEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (
        image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (
        patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, x_size, y_size = pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]
        patches = torch.nn.functional.unfold(pixel_values, (self.patch_size[0], self.patch_size[1]), stride=(self.patch_size[0], self.patch_size[1]))

        # To project to the right hidden embedding dim
        hidden_dim = 768
        x_projection = torch.nn.Linear(1, hidden_dim)
        y_projection = torch.nn.Linear(1, hidden_dim)

        # x_embeddings
        patches_x_embeddings = torch.arange(x_size).view(batch_size, 1, -1) / x_size
        patches_x_embeddings = x_projection(patches_x_embeddings.T).view(batch_size, x_size,
                                                                         hidden_dim)
        patches_x_embeddings = patches_x_embeddings.expand(y_size, batch_size, x_size, hidden_dim)
        patches_x_embeddings = patches_x_embeddings.reshape(batch_size, -1, hidden_dim)

        # y_embeddings
        patches_y_embeddings = torch.arange(y_size).view(batch_size, 1, -1) / y_size
        patches_y_embeddings = y_projection(patches_y_embeddings.T).view(batch_size, y_size,
                                                                         hidden_dim)
        patches_y_embeddings = patches_y_embeddings.expand(x_size, batch_size, y_size, hidden_dim)
        patches_y_embeddings = patches_y_embeddings.reshape(batch_size, -1, hidden_dim)

        patches_positional_embeddings = patches_x_embeddings + patches_y_embeddings

        return patches_positional_embeddings


class PatchPackModel(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        # Here, we need to change the patching proc
        # For now, we can just change the way the patches are generated while keeping their number the same
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTModel()
