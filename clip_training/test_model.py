from torchvision import datasets
import torch
from transformers import IntervalStrategy
from transformers.training_args import TrainingArguments
from clip_training.run_training import CustomTrainer, collate, compute_metrics, ClipModel


test_dataset = datasets.ImageFolder(root='/home/arnaud/imagenetv2-matched-frequency-format-val')

model = ClipModel()
model.load_state_dict(torch.load('tmp_trainer/checkpoint-4000/pytorch_model.bin'))
model.eval()

training_args = TrainingArguments(
    output_dir='tmp_trainer',
    per_device_train_batch_size=8,
    fp16=torch.cuda.is_available(),
    logging_strategy='steps',
    max_steps=5000,
    logging_steps=100,
    eval_steps=100,
    evaluation_strategy=IntervalStrategy.STEPS,
    per_device_eval_batch_size=12,  # NB: in this scenario, the "accuracy" actually depends on the eval batch size
    dataloader_drop_last=True,
    include_inputs_for_metrics=True,
    learning_rate=5e-5,
    optim='adafactor',
    lr_scheduler_type='cosine',  # TODO: cosine
    warmup_steps=100,
    # gradient_checkpointing=True  # TODO: Add it back to the ClipModel before enabling it here
)

import ipdb; ipdb.set_trace()

trainer = CustomTrainer(model=model, train_dataset=None, eval_dataset=test_dataset,
                        data_collator=collate,
                        args=training_args, compute_metrics=compute_metrics)
trainer.evaluate(test_dataset)
