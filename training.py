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

if __name__ == "__main__":
    #--------------------------------------------------------------------------------------
    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),         # Resize to fixed size
        transforms.ToTensor(),                 # Convert to tensor (C, H, W)
        transforms.Normalize(                  # Normalize using ImageNet means & stds
            mean=[0.485, 0.456, 0.406],        # RGB mean
            std=[0.229, 0.224, 0.225]          # RGB std
        )
    ])

    # Read the configurations
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    image_folder = config["data"]["image_dir"]
    captions_dataset = config["data"]["captions_file"]

    # Read the vocabulary file
    vocab = Vocabulary()
    vocab.from_json("vocab.json")
    #--------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------
    # -------- CONFIGURATION --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    # -------- SETUP DATA & VOCAB --------
    image_loader = ImageLoader(image_folder, captions_dataset, transform=image_transforms, batch_size=batch_size)
    dataloader = image_loader.get_dataloader()

    # -------- MODEL INIT --------
    #encoder = CNNEncoder().to(device)
    encoder = CNNEncoderWithMHSA(embed_size=256, attention_heads=2).to(device)
    #decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).to(device)
    decoder = TransformerDecoder(
    embed_size=config["training"]["embed_size"],
    vocab_size=len(vocab.word2idx),
    num_heads=2,
    num_layers=3,
    #hidden_dim=512,
    dropout=0.1
    ).to(device)

    # -------- LOSS & OPTIMIZER --------
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.mhsa.parameters())
    #params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    # -------- TRAINING LOOP --------
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        loop = tqdm(dataloader, leave=True)
        for imgs, imgs_name, captions in loop:
            #print(imgs_name)
            #print(captions)
            imgs = imgs.to(device)
            # Convert raw captions into numerical format
            tokenized_captions = [vocab.numericalize(caption) for caption in captions]

            # Make the length of captions equal for each batch
            caption_lengths = [len(cap) for cap in tokenized_captions]
            max_len = max(caption_lengths)
            padded_captions = [cap + [vocab.word2idx["<pad>"]] * (max_len - len(cap)) for cap in tokenized_captions]
            targets = torch.tensor(padded_captions).to(device)
            #print(targets.shape)
            #print("Size of target from training loop", targets.size())

            tgt_input = targets[:, :-1]  # all tokens except last
            #print("Size of tgt_input from training loop", tgt_input.size())
            tgt_output = targets[:, 1:] # all tokens except first
            #print("Size of tgt_output from training loop", tgt_output.size())

            tgt_input = tgt_input.permute(1, 0)     # shape [T, B]
            #print("Size of tgt_input from training loop", tgt_input.size())
            tgt_output = tgt_output.permute(1, 0)  # [T, B]
            #print("Size of tgt_output from training loop", tgt_output.size())

            # Generate mask
            tgt_seq_len = tgt_input.shape[0]
            tgt_mask = TransformerDecoder.generate_square_subsequent_mask(tgt_seq_len, device)

            # Forward pass
            features = encoder(imgs)
            #print("Feature stats - Mean:", features.mean().item(), "Std:", features.std().item())

            outputs = decoder(tgt_input, features, tgt_mask)
            #print(outputs)
            predicted = outputs.argmax(dim=-1)
            #print("Predicted tokens (training):", predicted)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_output.reshape(-1))
            #print(loss)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            # for name, param in encoder.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad is not None)
        
        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), "encoder_529_"+str(epoch)+".pth")


        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # Save models
    # torch.save(encoder.state_dict(), "encoder_" + str(timestamp) + ".pth")
    # torch.save(decoder.state_dict(), "decoder_" + str(timestamp) + ".pth")
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
