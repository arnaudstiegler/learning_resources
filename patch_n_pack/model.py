import collections
import collections.abc
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ViTModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.vit.modeling_vit import ViTPreTrainedModel


class PatchPackPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        self.hidden_dim = hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # Calling it projection so that it maps back to the CNN from ViT
        self.projection = torch.nn.Linear(1, self.hidden_dim)
        self.y_projection = torch.nn.Linear(1, self.hidden_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        # TODO: check height and width
        batch_size, _, x_size, y_size = pixel_values.shape

        # To project to the right hidden embedding dim
        hidden_dim = 768

        import math
        # TODO: we should get patches differently than this, this is not correct
        num_x_patches = math.floor(x_size/self.patch_size[0]) + 1
        num_y_patches = math.floor(y_size/self.patch_size[1]) + 1

        # x_embeddings
        patches_x_embeddings = torch.arange(x_size/self.patch_size[0]).view(batch_size, 1, -1) / math.floor(x_size/self.patch_size[0])
        patches_x_embeddings = self.projection(patches_x_embeddings.T).view(
            batch_size, num_x_patches, hidden_dim
        )
        patches_x_embeddings = patches_x_embeddings.expand(
            num_y_patches, batch_size, num_x_patches, hidden_dim
        )
        patches_x_embeddings = patches_x_embeddings.reshape(batch_size, -1, hidden_dim)

        # y_embeddings
        patches_y_embeddings = torch.arange(y_size).view(batch_size, 1, -1) / y_size
        patches_y_embeddings = self.y_projection(patches_y_embeddings.T).view(
            batch_size, y_size, hidden_dim
        )
        patches_y_embeddings = patches_y_embeddings.expand(
            x_size, batch_size, y_size, hidden_dim
        )
        patches_y_embeddings = patches_y_embeddings.reshape(batch_size, -1, hidden_dim)

        patches_positional_embeddings = patches_x_embeddings + patches_y_embeddings

        return patches_positional_embeddings


class PatchPackEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        self.hidden_dim = hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embeddings = PatchPackPatchEmbeddings(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:

        patches = torch.nn.functional.unfold(
            pixel_values,
            (self.patch_size[0], self.patch_size[1]),
            stride=(self.patch_size[0], self.patch_size[1]),
        )
        # Need to pad to resolution
        padded_patches = torch.nn.functional.pad(patches, (0, 1024-patches.shape[-1]))
        assert padded_patches.shape[-1] == 1024
        positional_embeddings = self.patch_embeddings(pixel_values)

        return padded_patches + positional_embeddings


class PatchPackModel(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        # Here, we need to change the patching proc
        # For now, we can just change the way the patches are generated while keeping their number the same
        self.embeddings = PatchPackEmbeddings(config)
        self.encoder = ViTModel(config)


class PatchPackModelImageClassification(ViTPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = PatchPackModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    # This is a copy paste from transformers
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
