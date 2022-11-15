import os

import click
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import Trainer
from transformers import ViTFeatureExtractor, RobertaTokenizer
from transformers import ViTModel, RobertaModel

from clip_training.datagen import get_filtered_df_for_training


class ClipDataset(Dataset):
    def __init__(self, df, images_path: str, transform=None):
        self.data_df = df
        self.img_dir = images_path
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k")

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        image = read_image(
            os.path.join(self.img_dir, f'{self.data_df.loc[idx, "image_index"]}.jpg'))
        text = self.data_df.loc[idx, 'TEXT']
        if self.transform:
            image = self.transform(image)
        return {'image': self.feature_extractor(image, return_tensors='pt'),
                'text': self.tokenizer(text, return_tensors='pt')}


def collate(inputs):
    max_length = max([elem['text']['input_ids'].shape[-1] for elem in inputs])
    input_ids = torch.concat([torch.nn.functional.pad(elem['text']['input_ids'],
                                                      (0,
                                                       max_length - elem['text']['input_ids'].shape[
                                                           -1]),
                                                      value=1) for elem in inputs])
    masks = torch.concat([torch.nn.functional.pad(elem['text']['attention_mask'],
                                                  (0, max_length -
                                                   elem['text']['attention_mask'].shape[-1]),
                                                  value=0) for elem in inputs])
    image = torch.stack([elem['image']['pixel_values'][0] for elem in inputs])

    return {'image': {'pixel_values': image},
            'text': {'input_ids': input_ids, 'attention_mask': masks}}


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(768, 768)

        self.temperature = torch.nn.Parameter(torch.tensor([0.07]))  # Provided by the paper

    def forward(self, text, image):
        outputs = self.image_encoder(**image)
        image_embedding = outputs.last_hidden_state[:, 0]
        image_embedding = nn.functional.normalize(self.image_proj(image_embedding), p=2, dim=-1)

        outputs = self.text_encoder(**text)
        text_embedding = outputs.last_hidden_state[:, 0]
        text_embedding = nn.functional.normalize(self.text_proj(text_embedding), p=2, dim=-1)

        logits = torch.matmul(text_embedding, image_embedding.T) * torch.exp(self.temperature)

        return logits


class CustomTrainer(Trainer):
    @staticmethod
    def get_label(inputs):
        return torch.arange(inputs['text']['input_ids'].shape[0])

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(text=inputs['text'], image=inputs['image'])

        # # forward pass
        labels = self.get_label(inputs)

        loss_t = nn.functional.cross_entropy(outputs, labels)
        loss_i = nn.functional.cross_entropy(outputs.T, labels)
        loss = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss


@click.command()
@click.option('--filepath')
@click.option('--images_path')
def run_training(filepath: str, images_path: str) -> None:
    df = get_filtered_df_for_training(filepath, images_path)
    train_dataset = ClipDataset(df, images_path)

    model = ClipModel()
    trainer = CustomTrainer(model=model, train_dataset=train_dataset, data_collator=collate)

    trainer.train()


if __name__ == '__main__':
    run_training()
