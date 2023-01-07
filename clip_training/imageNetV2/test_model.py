from torchvision import datasets
import torch
from transformers import RobertaTokenizer, ViTFeatureExtractor
from clip_training.run_training import ClipModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import torch
import itertools
from torch.utils.data import DataLoader, Dataset


test_dataset = datasets.ImageFolder(root='/home/arnaud/imagenetv2-matched-frequency-format-val')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ClipModel()
model.load_state_dict(torch.load('tmp_trainer/checkpoint-4000/pytorch_model.bin'))
model.eval()

with open('clip_training/imageNetV2/idx_to_labels.json') as f:
    labels_list = json.load(f)

# with torch.no_grad():
encoded_labels = []
for label_elem in tqdm(labels_list[0:10]):
    label = label_elem['synset'][0]
    processed_text = tokenizer(label, return_tensors='pt')
    encoded_labels.append(model.text_encoder(processed_text['input_ids'], processed_text['attention_mask']).last_hidden_state[:, 0])

# image_embeddings = []
images = []
targets = []
for image, target in test_dataset:
    targets.append(target)
    images.append(feature_extractor(image, return_tensors='pt'))
    if len(images) > 5:
        break

image_dataloader = DataLoader(images, batch_size=3)

image_embeddings = []
for batch in image_dataloader:

    encoded_images = model.image_encoder(batch['pixel_values'].squeeze(1)).last_hidden_state[:,0]
    image_embeddings.append(encoded_images)

import ipdb;ipdb.set_trace()
    # image_embeddings.append(model.image_encoder(processed_image['pixel_values']).last_hidden_state[:, 0])
    # if len(image_embeddings) > 4:
    #     break


image_tensor = torch.stack(image_embeddings)
text_tensor = torch.stack(encoded_labels)


similarity_dataloader = DataLoader(itertools.product(text_tensor, image_tensor), bat)
import ipdb;ipdb.set_trace()
    # tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    # tensor_y = torch.Tensor(my_y)
    #
    # my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # my_dataloader = DataLoader(my_dataset)  # create your dataloader
    # # Store prediction for each label
    # image_prediction = torch.cat(
    #     [model.cos_similarity(image_embedding, encoded_label) for encoded_label in encoded_labels])
    # predictions.append(torch.argmax(image_prediction).item())

print(f'Accuracy on ImageNet: {accuracy_score(targets, predictions)}')
