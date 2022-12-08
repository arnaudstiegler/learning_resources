import os
from typing import Dict

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import Trainer
from transformers import ViTFeatureExtractor, RobertaTokenizer
from transformers import ViTModel, RobertaModel
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from clip_training.datagen import get_filtered_df_for_training
from dataclasses import dataclass, asdict


@dataclass
class ClipSample:
    input_ids: torch.tensor
    attention_mask: torch.tensor
    pixel_values: torch.tensor


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


def collate(inputs) -> Dict:
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

    return asdict(ClipSample(pixel_values=image, input_ids=input_ids, attention_mask=masks))


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(768, 768)

        self.temperature = torch.nn.Parameter(torch.tensor([0.07]))  # Provided by the paper

    def forward(self, text, attention_mask, image):
        outputs = self.image_encoder(image)
        image_embedding = outputs.last_hidden_state[:, 0]
        image_embedding = nn.functional.normalize(self.image_proj(image_embedding), p=2, dim=-1)

        outputs = self.text_encoder(text, attention_mask=attention_mask)
        text_embedding = outputs.last_hidden_state[:, 0]
        text_embedding = nn.functional.normalize(self.text_proj(text_embedding), p=2, dim=-1)

        logits = torch.matmul(text_embedding, image_embedding.T) * torch.exp(self.temperature)

        return logits


def get_label(outputs, n_device):
    return torch.stack([torch.arange(outputs.shape[-1]) for _ in range(n_device)])


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(text=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        image=inputs['pixel_values'])
        # If using data parallel, the effective batch size is different
        n_device = torch.cuda.device_count() if outputs.device.type == 'cuda' else 1
        device = outputs.device

        labels = get_label(outputs, n_device).to(device)
        batched_outputs = outputs.view(n_device, outputs.shape[-1], outputs.shape[-1])

        loss_t = nn.functional.cross_entropy(batched_outputs, labels)
        loss_i = nn.functional.cross_entropy(batched_outputs.permute(0, 2, 1), labels)
        loss = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss


def compute_metrics(
        predictions: EvalPrediction
) -> Dict[str, float]:
    batch_size = predictions.predictions.shape[-1]
    predicted_ids = np.argmax(predictions.predictions)
    labels = get_label(predictions.predictions, batch_size)

    import ipdb; ipdb.set_trace()

    raise NotImplementedError


@click.command()
@click.option('--filepath')
@click.option('--images_path')
def run_training(filepath: str, images_path: str) -> None:
    df = get_filtered_df_for_training(filepath, images_path)
    train, test = train_test_split(df, test_size=0.05, random_state=0)

    train_dataset = ClipDataset(train.reset_index(drop=True), images_path)
    test_dataset = ClipDataset(test.reset_index(drop=True), images_path)

    model = ClipModel()

    training_args = TrainingArguments(
        output_dir='tmp_trainer',
        per_device_train_batch_size=2,
        fp16=False,
        logging_strategy='steps',
        max_steps=1000,
        logging_steps=100,
        eval_steps=1,
        evaluation_strategy='steps',
        per_device_eval_batch_size=3,
        dataloader_drop_last=True,
        include_inputs_for_metrics=True,
        # TODO: add warmup steps
        # TODO: use Adafactor
        # gradient_checkpointing=True  # TODO: Add it back to the ClipModel before enabling it here
    )

    trainer = CustomTrainer(model=model, train_dataset=train_dataset, eval_dataset=test_dataset,
                            data_collator=collate,
                            args=training_args, compute_metrics=compute_metrics)

    trainer.train()

    # trainer.evaluate(eval_dataset=test_dataset)


if __name__ == '__main__':
    run_training()