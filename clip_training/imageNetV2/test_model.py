from torchvision import datasets
from transformers import RobertaTokenizer, ViTFeatureExtractor
from clip_training.run_training import ClipModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

'''
This script is very messy and could be optimized to have a faster evaluation loop
'''

test_dataset = datasets.ImageFolder(root='/home/arnaud/imagenetv2-matched-frequency-format-val')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ClipModel()
model.load_state_dict(torch.load('tmp_trainer/checkpoint-100000/pytorch_model.bin'))
model.eval()

with open('clip_training/imageNetV2/idx_to_labels.json') as f:
    labels_list = json.load(f)

with torch.no_grad():
    print('Encoding targets')
    encoded_labels = []
    for label_elem in tqdm(labels_list):
        # Using the full description
        label = 'Names: ' + ', '.join(label_elem['synset']) + '. Description: ' + label_elem['gloss']
        processed_text = tokenizer(label, return_tensors='pt')
        encoded_labels.append(model.text_encoder(processed_text['input_ids'], processed_text['attention_mask']).last_hidden_state[:, 0])

    print('Encoding images')
    images = []
    targets = []
    for idx, (image, target) in tqdm(enumerate(test_dataset)):
        targets.append(target)
        images.append((idx, feature_extractor(image, return_tensors='pt')))

    image_dataloader = DataLoader(images, batch_size=32)

    image_embeddings = []
    for batch in tqdm(image_dataloader):
        idx, image = batch
        encoded_images = model.image_encoder(image['pixel_values'].squeeze(1)).last_hidden_state[:,0]
        image_embeddings.append((idx, encoded_images))

    image_idx = torch.concat([elem[0] for elem in image_embeddings])
    image_tensor = torch.concat([elem[1] for elem in image_embeddings]).view(-1, 768)
    # num samples, 768
    text_tensor = torch.stack(encoded_labels).squeeze(1)

    # Manually creating the list to keep the indices
    sample_list = []
    for idx, img_tensor in zip(image_idx, image_tensor):
        for label_idx, label in enumerate(text_tensor):
            sample_list.append((idx, img_tensor, label_idx, label))

    print('Running similarities')

    results = defaultdict(dict)
    sample_dataloader = DataLoader(sample_list, batch_size=32)
    for sample in tqdm(sample_dataloader):
        img_idxs, img_emb, label_idxs, label_emb = sample
        score = model.cos_similarity(img_emb, label_emb, full_matrix=False)
        for score_idx, (img_idx, label_idx) in enumerate(zip(img_idxs, label_idxs)):
            results[img_idx.item()][label_idx.item()] = score[score_idx]

    predictions = []
    for img_id in sorted(results.keys()):
        predictions.append(torch.argmax(torch.stack([val for val in results[img_id].values()])).item())

    import ipdb; ipdb.set_trace()

    print(f'Accuracy on ImageNet: {accuracy_score(targets, predictions)}')
