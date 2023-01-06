from torchvision import datasets
import torch
from transformers import RobertaTokenizer, ViTFeatureExtractor
from clip_training.run_training import ClipModel
from tqdm import tqdm


test_dataset = datasets.ImageFolder(root='/home/arnaud/imagenetv2-matched-frequency-format-val')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ClipModel()
model.load_state_dict(torch.load('tmp_trainer/checkpoint-4000/pytorch_model.bin'))
model.eval()

with torch.no_grad():
    encoded_labels = []
    for label in ['test1', 'test2']:
        processed_text = tokenizer(label, return_tensors='pt')
        encoded_labels.append(model.text_encoder(processed_text['input_ids'], processed_text['attention_mask']))

    for elem in tqdm(test_dataset):
        processed_image = feature_extractor(elem[0], return_tensors='pt')
        image_embedding = model.image_encoder(processed_image['pixel_values'])
        for encoded_label in encoded_labels:
            model.cos_similarity(image_embedding, encoded_label)
