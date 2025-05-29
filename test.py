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

def generate_caption(image_tensor, encoder, decoder, vocab, max_length=20):
    with torch.no_grad():
        # Encode the image to get memory (encoder output)
        memory = encoder(image_tensor)  # shape: [B, T_mem, E] or [1, seq_len, embed]
        print("Encoder output shape before permute:", memory.shape)
        # memory = memory.permute(1, 0, 2)  # [T_mem, B, E] for TransformerDecoder
        # print("Encoder output shape after permute:", memory.shape)

        # Initialize caption generation with <start>
        caption = [vocab.word2idx["<start>"]]

        for _ in range(max_length):
            
            targets = torch.tensor(caption).unsqueeze(1).to(device)  # [T, 1]

            if targets.shape[0] == 1:
                tgt_input = targets  # First time: don't remove anything
            else:
                tgt_input = targets[:-1, :]  # Remove last token (the one to predict)
            
            tgt_seq_len = tgt_input.shape[0]
            tgt_mask = TransformerDecoder.generate_square_subsequent_mask(tgt_seq_len, device)
            #print(targets.size())

            #tgt_output = targets[:, 1:] # all tokens except first

            #tgt_input = tgt_input.permute(1, 0)     # shape [T, B]
            #tgt_output = tgt_output.permute(1, 0)  # [T, B]
        
            #tgt_input = torch.tensor(caption).unsqueeze(1).to(device)  # [T, 1]
            #tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(device)  # [T, T]

            output = decoder(tgt_input, memory, tgt_mask)  # [T, 1, vocab_size]
            if output.shape[0] == 0:
                break  # Avoid indexing error if somehow empty
            next_word_logits = output[-1, 0, :]  # take last token output
            next_word_id = next_word_logits.argmax().item()
            predicted_tokens = output.argmax(dim=-1)
            print("Predicted tokens:", predicted_tokens[:, 0])  # batch first

            if vocab.idx2word[next_word_id] == "<end>":
                break

            caption.append(next_word_id)

            # embeddings = decoder.embed(word).unsqueeze(1)  # (1, 1, embed_size)
            # hiddens, states = decoder.lstm(embeddings, states)
            # outputs = decoder.linear(hiddens.squeeze(1))  # (1, vocab_size)
            # predicted = outputs.argmax(1)  # (1,)
            # word = predicted

            # predicted_word = vocab.idx2word[predicted.item()]
            # if predicted_word == "<end>":
            #     break
            # caption.append(predicted_word)
        
        # Convert caption ids to words, skip <start>
        caption_words = [vocab.idx2word[idx] for idx in caption[1:]]
        return " ".join(caption_words)

    return " ".join(caption)

# /Users/tejfaster/Developer/Python/cv_project/EchoLens/src/image/pexels-souvenirpixels-414612.jpg

image_path = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens/Test_Data/Images/"
images = os.listdir(image_path)
#test_img = ["667626_18933d713e.jpg", "3747543364_bf5b548527.jpg"]
#test_img = ["mQcKcyt3Nb25vMYBSbb47aQ9Kw.jpg", "pexels-souvenirpixels-414612.jpg"]

#3405942945_f4af2934a6.jpg
#3091912922_0d6ebc8f6a.jpg
#3463922449_f6040a2931.jpg

image_tensor = preprocess_image("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/3091912922_0d6ebc8f6a.jpg")
caption = generate_caption(image_tensor, encoder, decoder, vocab)
print("Generated Caption 3091912922_0d6ebc8f6a.jpg:", caption)

# for i in images:
#     img_path = image_path + i
#     image_tensor = preprocess_image(img_path)
#     caption = generate_caption(image_tensor, encoder, decoder, vocab)
#     print(f"Generated Caption {i}:", caption)
#     #show_image_with_caption(img_path, caption)
#     #break

    



