import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)  # Word embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)   # Project to vocab size
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        features: (batch_size, embed_size) - image feature vector from encoder
        captions: (batch_size, max_seq_len) - target caption indices (with <start> token)
        """
        embeddings = self.embed(captions[:, :-1])  # Ignore the last word for teacher forcing
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # Concatenate image feature at start
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        