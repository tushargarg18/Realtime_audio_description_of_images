import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, tgt_padding_mask=None):
        # tgt: [B, T], memory: [B, 1, embed_size] or [B, M, embed_size]
        #print("Size of target", tgt.size())
        T, B = tgt.size()

        positions = torch.arange(0, T).unsqueeze(1).expand(T, B).to(tgt.device)  # [T, B]
        #print("Size of positions", positions.size())
       
        tgt_embed = self.dropout(self.word_embedding(tgt) + self.pos_embedding(positions))
        #print("Size of tgt_embed", tgt_embed.size())

        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        #print("Size of output", self.fc_out(output).size())

        return self.fc_out(output)  # [B, T, vocab_size]
    
    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.triu(torch.ones((sz, sz), device = device), diagonal=1)
        #print(mask)
        return mask.masked_fill(mask == 1, float('-inf'))