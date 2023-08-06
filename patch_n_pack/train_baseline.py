import click
import torch
from torch.optim import Adam
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from functools import partial
import torchmetrics
from tqdm import tqdm
import wandb


BASELINE_CONFIG = {
    'max_epochs': 5,
    'learning_rate': 1e-5,
    'train_batch_size': 64,  # For image size 224*224
    'val_batch_size': 64,  # For image size 224*224
    'train_steps': 100,  # Log train loss every train steps
    'val_steps': 2500  # Eval every val_steps steps
}


def collate(processor, samples):
    labels = torch.tensor([sample['label'] for sample in samples])
    images = [sample['image'].convert('RGB') for sample in samples]

    return processor(images, return_tensors="pt"), labels


@click.command()
@click.option('--wandb_logging', is_flag=True)
def train(wandb_logging: bool):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if wandb_logging:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project='patch_n_pack',
            # Track hyperparameters and run metadata
            config=BASELINE_CONFIG)

    train_dataset = load_dataset("aharley/rvl_cdip", split='train')
    val_dataset = load_dataset("aharley/rvl_cdip", split='validation')
    num_classes = len(set(train_dataset['label']))

    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
    train_loss_metric = torchmetrics.aggregation.MeanMetric()
    val_loss_metric = torchmetrics.aggregation.MeanMetric()

    # Initializing the model from scratch
    config = ViTConfig(num_labels=num_classes)
    processor = ViTImageProcessor(config)
    model = ViTForImageClassification(config)

    collate_fn = partial(collate, processor)

    # Need to create a dataloader here
    train_dataloader = DataLoader(train_dataset, batch_size=BASELINE_CONFIG['train_batch_size'], collate_fn=collate_fn, num_workers=24)
    val_dataloader = DataLoader(val_dataset, batch_size=BASELINE_CONFIG['val_batch_size'], collate_fn=collate_fn, num_workers=24)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in range(BASELINE_CONFIG['max_epochs']):
        for i, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            images, labels = batch
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
