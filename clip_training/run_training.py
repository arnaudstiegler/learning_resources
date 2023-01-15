import os
from typing import Dict

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import Trainer, IntervalStrategy
from transformers import ViTFeatureExtractor, RobertaTokenizer
from transformers import ViTModel, RobertaModel
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from clip_training.datagen import get_filtered_df_for_training


class ClipDataset(Dataset):
    def __getitem__(self, idx):
        image = read_image(
            os.path.join(self.img_dir, f'{int(self.data_df.loc[idx, "SAMPLE_ID"])}.jpg'))
        text = self.data_df.loc[idx, 'TEXT']
        if self.transform:
            image = self.transform(image)

        tokenized_text = self.tokenizer(text, return_tensors='pt', truncation=True)

        return {
            'pixel_values': self.feature_extractor(image, return_tensors='pt')['pixel_values'],
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            # Allows compute_metrics to work on the trainer side without static labels
            'return_loss': True
        }

    def __init__(self, df, images_path: str, transform=None):
        self.data_df = df
        self.img_dir = images_path
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    def __len__(self):
        return self.data_df.shape[0]


def collate(inputs) -> Dict:
    max_length = max([elem['input_ids'].shape[-1] for elem in inputs])
    input_ids = torch.concat([torch.nn.functional.pad(elem['input_ids'],
                                                      (0,
                                                       max_length - elem['input_ids'].shape[
                                                           -1]),
                                                      value=1) for elem in inputs])
    masks = torch.concat([torch.nn.functional.pad(elem['attention_mask'],
                                                  (0, max_length -
                                                   elem['attention_mask'].shape[-1]),
                                                  value=0) for elem in inputs])
    image = torch.stack([elem['pixel_values'][0] for elem in inputs])

    return {'input_ids': input_ids, 'attention_mask': masks,
            'pixel_values': image, 'return_loss': True}


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        self.text_proj = nn.Linear(768, 768)
        self.image_proj = nn.Linear(768, 768)

        self.temperature = torch.nn.Parameter(torch.tensor([0.07]))  # Value provided by the paper

    def cos_similarity(self, image_embedding, text_embedding, full_matrix=True):
        image_embedding = nn.functional.normalize(self.image_proj(image_embedding), p=2, dim=-1)
        text_embedding = nn.functional.normalize(self.text_proj(text_embedding), p=2, dim=-1)

        if full_matrix:
            # Compute the cos_similarity across all images and all labels
            logits = torch.matmul(text_embedding, image_embedding.T) * torch.exp(self.temperature)
        else:
            # Compute the cos similarity for column-wise, similar to taking the diagonal of the full matrix product
            logits = (text_embedding * image_embedding).sum(dim=1)
        return logits

    def forward(self, input_ids, attention_mask, pixel_values, return_loss=True):
        image_output = self.image_encoder(pixel_values)
        text_output = self.text_encoder(input_ids, attention_mask=attention_mask)
        # Take the CLS token as image representation
        logits = self.cos_similarity(image_output.last_hidden_state[:, 0], text_output.last_hidden_state[:, 0])

        return logits


def get_label(outputs):
    if outputs.shape[0] % outputs.shape[-1] != 0:
        raise ValueError
    return torch.arange(outputs.shape[-1]).repeat(outputs.shape[0] // outputs.shape[-1])


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'])
        device = outputs.device

        labels = get_label(outputs).to(device)

        loss_t = nn.functional.cross_entropy(outputs, labels)
        loss_i = nn.functional.cross_entropy(outputs.T, labels)
        loss = (loss_i + loss_t) / 2

        return (loss, {'pred': outputs}) if return_outputs else loss


def compute_metrics(
        predictions: EvalPrediction
) -> Dict[str, float]:
    batch_size = predictions.predictions.shape[-1]
    num_samples = predictions.predictions.shape[0]

    logits = predictions.predictions.reshape(-1, batch_size)
    encoded_preds = np.argmax(logits, axis=-1)

    labels = np.tile(np.arange(batch_size), reps=num_samples // batch_size)

    correct = encoded_preds == labels

    acc = np.sum(correct) / num_samples

    return {'accuracy': acc}


@click.command()
@click.option('--filepath')
@click.option('--images_path')
def run_training(filepath: str, images_path: str) -> None:
    df = get_filtered_df_for_training(filepath, images_path)
    # Given the overall size of the dataset, test_size is kept low (size is still well over 10k examples)
    train, test = train_test_split(df, test_size=0.01, random_state=0)

    train_dataset = ClipDataset(train.reset_index(drop=True), images_path)
    test_dataset = ClipDataset(test.reset_index(drop=True), images_path)

    model = ClipModel()

    training_args = TrainingArguments(
        output_dir='tmp_trainer',
        per_device_train_batch_size=12,
        fp16=torch.cuda.is_available(),
        logging_strategy='steps',
        max_steps=int(1e5),
        logging_steps=100,
        eval_steps=1000,
        save_steps=10000,
        evaluation_strategy=IntervalStrategy.STEPS,
        per_device_eval_batch_size=24,  # NB: in this scenario, the "accuracy" actually depends on the eval batch size
        dataloader_drop_last=True,
        include_inputs_for_metrics=True,
        learning_rate=1e-5,
        optim='adafactor',
        lr_scheduler_type='cosine',  # TODO: cosine
        warmup_steps=500,
        # gradient_checkpointing=True  # TODO: Add it back to the ClipModel before enabling it here
    )

    trainer = CustomTrainer(model=model, train_dataset=train_dataset, eval_dataset=test_dataset,
                            data_collator=collate,
                            args=training_args, compute_metrics=compute_metrics)

    # Initial evaluation before training
    trainer.evaluate()
    trainer.train()


if __name__ == '__main__':
    run_training()
