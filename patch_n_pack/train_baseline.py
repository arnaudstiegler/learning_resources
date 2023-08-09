import click
import torch
from typing import Optional
from torch.optim import Adam
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from functools import partial
import torchmetrics
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast


BASELINE_CONFIG = {
    'max_epochs': 8,
    'learning_rate': 1e-5,
    'image_size': 512,
    'train_steps': 100,  # Log train loss every train steps
    'val_steps': 2000  # Eval every val_steps steps
}

BATCH_SIZE_FOR_IMAGE_SIZE = {
    224: {
        'train_batch_size': 64,
        'val_batch_size': 64,
    },
    512: {
        'train_batch_size': 28,
        'val_batch_size': 56,
    },
    1024: {
        'train_batch_size': 2,
        'val_batch_size': 4,
    }
}


def collate(processor, samples):
    labels = torch.tensor([sample['label'] for sample in samples])
    images = [sample['image'].convert('RGB') for sample in samples]

    return processor(images, return_tensors="pt"), labels


@click.command()
@click.option('--image_size', default=BASELINE_CONFIG['image_size'], help='Number of greetings.')
@click.option('--train_batch_size', type=int)
@click.option('--val_batch_size', type=int)
@click.option('--local_train_data', type=str)
@click.option('--local_val_data', type=str)
@click.option('--wandb_logging', is_flag=True)
def train(image_size: int, wandb_logging: bool, train_batch_size: Optional[int], val_batch_size: Optional[int], local_train_data: Optional[str], local_val_data: Optional[str]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if wandb_logging:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project='patch_n_pack',
            # Track hyperparameters and run metadata
            config={
                'base_config': BASELINE_CONFIG,
                'image_size': image_size,
                'train_batch_size': train_batch_size,
                'val_batch_size': val_batch_size,
            }
        )

    print('Initializing the dataset')
    if local_train_data and local_val_data:
        train_dataset = load_from_disk(local_train_data)
        val_dataset = load_from_disk(local_val_data)
    else:
        train_dataset = load_dataset("aharley/rvl_cdip", split='train')
        val_dataset = load_dataset("aharley/rvl_cdip", split='validation')

    num_classes = len(set(train_dataset['label']))

    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
    train_loss_metric = torchmetrics.aggregation.MeanMetric()
    val_loss_metric = torchmetrics.aggregation.MeanMetric()

    # Initializing the model from scratch
    print('Initializing the model')
    config = ViTConfig(num_labels=num_classes, image_size=image_size)
    processor = ViTImageProcessor(config, size={"height": config.image_size, "width": config.image_size})
    model = ViTForImageClassification(config)
    model.gradient_checkpointing_enable()

    collate_fn = partial(collate, processor)

    # Need to create a dataloader here
    train_bs = train_batch_size if train_batch_size else BATCH_SIZE_FOR_IMAGE_SIZE[image_size]['train_batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, collate_fn=collate_fn, num_workers=16)
    val_bs = val_batch_size if val_batch_size else BATCH_SIZE_FOR_IMAGE_SIZE[image_size]['val_batch_size']
    val_dataloader = DataLoader(val_dataset, batch_size=val_bs, collate_fn=collate_fn, num_workers=16)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    print('Starting training')
    for epoch in range(BASELINE_CONFIG['max_epochs']):
        for i, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            images, labels = batch

            with autocast(dtype=torch.float16):
                optimizer.zero_grad()

                outputs = model(pixel_values=images.pixel_values.to(device), labels=labels.to(device))
                loss = outputs.loss
                train_loss_metric(loss.to('cpu'))
                loss.backward()
                optimizer.step()

                if i > 0 and i % BASELINE_CONFIG['train_steps'] == 0:
                    print({'train_loss': train_loss_metric.compute()})

                    if wandb_logging:
                        wandb.log({'train_loss': train_loss_metric.compute().item()})

                if i > 0 and i % BASELINE_CONFIG['val_steps'] == 0:
                    with torch.no_grad():
                        model.eval()
                        acc_metric.reset()
                        val_loss_metric.reset()

                        for val_batch in tqdm(val_dataloader, position=1):
                            images, labels = val_batch

                            out = model(pixel_values=images.pixel_values.to(device), labels=labels.to(device))
                            acc_metric(out.logits.to('cpu'), labels)
                            val_loss_metric(out.loss.to('cpu'))

                        print({'val_loss': val_loss_metric.compute(), 'accuracy': acc_metric.compute()})

                        if wandb_logging:
                            wandb.log({'val_loss': val_loss_metric.compute().item(), 'accuracy': acc_metric.compute().item()})


if __name__ == "__main__":
    train()
