import torch
import torch.nn as nn
import torchvision.models as models

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import os
from PIL import Image
from torchvision import transforms

from encoder_pretrained_multi_attention import CNNEncoderWithMHSA
from decoder_transformer import TransformerDecoder
import yaml
from vocabulary import Vocabulary

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import os
from tqdm import tqdm
import yaml
import json

from imageLoader import ImageCaptionDataset, ImageLoader
#from encoder import CNNEncoder
#from encoder_pretrained import CNNEncoder
from encoder_pretrained_multi_attention import CNNEncoderWithMHSA
from vocabulary import Vocabulary
from decoder_transformer import TransformerDecoder
#from decoder import DecoderRNN

from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the configurations
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Read the vocabulary file
vocab = Vocabulary()
vocab.from_json("vocab.json")

#encoder = CNNEncoder()
encoder = CNNEncoderWithMHSA(embed_size=256, attention_heads=2).to(device)
decoder = TransformerDecoder(
    embed_size=config["training"]["embed_size"],
    vocab_size=len(vocab.word2idx),
    num_heads=2,
    num_layers=3,
    #hidden_dim=512,
    dropout=0.1
    ).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    return image.to(device)


img1 = preprocess_image("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/44856031_0d82c2c7d1.jpg")
img2 = preprocess_image("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/56489627_e1de43de34.jpg")

with torch.no_grad():
    feat1 = encoder(img1)
    feat2 = encoder(img2)
    print("Encoder output mean/std for img1:", feat1.mean().item(), feat1.std().item())
    print("Encoder output mean/std for img2:", feat2.mean().item(), feat2.std().item())
    print("L2 distance between features:", torch.norm(feat1 - feat2).item())