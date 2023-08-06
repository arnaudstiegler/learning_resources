import click
import torch
from torch.optim import Adam
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from functools import partial
import torchmetrics


BASELINE_CONFIG = {
    'max_epochs': 1,
    'learning_rate': 1e-5,
    'train_batch_size': 64, # For image size 224*224
    # Eval every val_steps steps
    'val_steps': 5000
}


def collate(processor, samples):
    labels = torch.tensor([sample['label'] for sample in samples])
    images = [sample['image'].convert('RGB') for sample in samples]

    return processor(images, return_tensors="pt"), labels


@click.command()
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = load_dataset("aharley/rvl_cdip", split='train')
    val_dataset = load_dataset("aharley/rvl_cdip", split='validation')
    num_classes = len(set(train_dataset['label']))

    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
    loss_metric = torchmetrics.aggregation.MeanMetric()

    # Initializing the model from scratch
    config = ViTConfig(num_labels=num_classes)
    processor = ViTImageProcessor(config)
    model = ViTForImageClassification(config)

    collate_fn = partial(collate, processor)

    # Need to create a dataloader here
    train_dataloader = DataLoader(train_dataset, batch_size=BASELINE_CONFIG['train_batch_size'], collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BASELINE_CONFIG['train_batch_size'], collate_fn=collate_fn)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in range(BASELINE_CONFIG['max_epochs']):
        for i, batch in enumerate(train_dataloader):
            model.train()
            images, labels = batch
            optimizer.zero_grad()

            outputs = model(pixel_values=images.pixel_values.to(device), labels=labels.to(device))
            loss = outputs.loss

            loss_metric(loss)

            print(loss_metric.compute())

            loss.backward()
            optimizer.step()

            if i % BASELINE_CONFIG['val_steps'] == 0:
                with torch.no_grad():
                    model.eval()
                    acc_metric.reset()
                    for batch in val_dataloader:
                        images, labels = batch
                        logits = model(pixel_values=images.pixel_values.to(device)).logits
                        acc_metric(logits.to('cpu'), labels)

                    print(metric.compute())



if __name__ == "__main__":
    train()
