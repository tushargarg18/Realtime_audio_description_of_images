import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom dataset class
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, captions_dataset, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Load captions as a list of (image_name, caption) pairs
        self.data = []
        with open(str(captions_dataset), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().lower().split(",", 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    img_name = img_name.strip()
                    caption = caption.strip()
                    image_path = os.path.join(self.image_folder, img_name)
                    if os.path.exists(image_path):  # check if image file exists
                        self.data.append((img_name, caption))
                    #else:
                        #print(f"[Warning] Skipped missing image at line: {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name, caption

# 2. Loader class
class ImageLoader:
    def __init__(self, image_folder,captions_dataset, transform, batch_size=32):
        self.dataset = ImageCaptionDataset(image_folder,captions_dataset, transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def get_dataloader(self):
        return self.dataloader

    def denormalize(self, img_tensor):
        """
        Converts a normalized image tensor back to viewable format (H, W, C) for visualization.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img_tensor.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    def visualize_batch(self, n=6):
        """
        Visualizes first 'n' images from the batch.
        """
        data_iter = iter(self.dataloader)
        images, image_names = next(data_iter)

        plt.figure(figsize=(12, 6))
        for i in range(min(n, len(images))):
            plt.subplot(2, 3, i+1)
            img = self.denormalize(images[i])
            plt.imshow(img)
            plt.title(image_names[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# image_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),         # Resize to fixed size
#         transforms.ToTensor(),                 # Convert to tensor (C, H, W)
#         transforms.Normalize(                  # Normalize using ImageNet means & stds
#             mean=[0.485, 0.456, 0.406],        # RGB mean
#             std=[0.229, 0.224, 0.225]          # RGB std
#         )
#     ])
# image_folder = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images"
# #image_folder = "/Users/tejfaster/Developer/Python/cv_project/EchoLens/DataSet/Images"
# captions_dataset = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt"
# #captions_dataset = "//Users/tejfaster/Developer/Python/cv_project/EchoLens/DataSet/captions.txt"

# dataset = ImageCaptionDataset(image_folder,captions_dataset, transform=image_transforms)

# loader = ImageLoader(image_folder,captions_dataset,transform=image_transforms)
# print(len(loader.get_dataloader()))