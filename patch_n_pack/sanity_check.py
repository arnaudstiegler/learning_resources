from PIL import Image
from patch_n_pack.utils import resize_with_aspect_ratio, set_up_model
import torch
from torch.optim import Adam
import random

MAX_EPOCHS = 10


def single_batch_overfit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = Image.open("patch_n_pack/images/10003.jpeg")
    processor, model = set_up_model()

    model.to(device)

    inputs = processor(images=image, return_tensors="pt")
    optimizer = Adam(model.parameters(), lr=1e-5)

    for i in range(MAX_EPOCHS):
        optimizer.zero_grad()

        resolution = random.choice([128, 256, 512, 1024])

        print(f'Resolution: {resolution}')

        x = resize_with_aspect_ratio(inputs.pixel_values, resolution)
        outputs = model(pixel_values=x.to(device), labels=torch.tensor([0], device=device))
        loss = outputs.loss

        print(loss.item())

        loss.backward()
        optimizer.step()
    return


if __name__ == "__main__":
    single_batch_overfit()
