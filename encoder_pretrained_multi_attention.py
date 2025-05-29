import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoderWithMHSA(nn.Module):
    def __init__(self, embed_size=256, attention_heads=2):
        super(CNNEncoderWithMHSA, self).__init__()

        # Load pretrained ResNet50 and remove classifier layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.backbone = nn.Sequential(*modules)  # Output: [B, 2048, 7, 7]

        self.conv_out_channels = 2048
        self.spatial_size = 7  # From ResNet50 feature map size: 7x7
        self.num_patches = self.spatial_size * self.spatial_size  # = 49
        # Multihead Self-Attention
        # Learnable positional encoding [1, 49, 2048]
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.conv_out_channels)
        )

        self.mhsa = nn.MultiheadAttention(embed_dim=self.conv_out_channels,
                                          num_heads=attention_heads,
                                          batch_first=True)

        # Linear layer to reduce dimensionality for decoder
        self.fc = nn.Linear(self.conv_out_channels, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        # CNN feature maps: [B, 2048, 7, 7]
        features = self.backbone(images)
        #print("Size of features:", features.size())

        # Flatten spatial dimensions: [B, 2048, 49] → transpose → [B, 49, 2048]
        B, C, H, W = features.size()
        #print(features.size())
        features = features.view(B, C, H*W).permute(0, 2, 1)
        #print("Size of features after resize:", features.size())
        
        # Add positional encoding
        features = features + self.positional_encoding  # [B, 49, 2048]
        #print("Size of features after positional encoding:", features.size())

        # Self-attention on spatial tokens
        attended_features, _ = self.mhsa(features, features, features)
        #print("Size of attended_features:", attended_features.size())
        #print("Attention out mean:", attended_features.mean().item(), "std:", attended_features.std().item())

        out = self.relu(self.fc(attended_features))  # [B, embed_size, Multiattention head] 3D vector
        #print("CNN features sample:", features.view(features.size(0), features.size(1), -1)[0, :, 0][:5])
        
        x = out.permute(1, 0, 2)
        #print("Encoder output size", x.size())
        return out.permute(1, 0, 2)  # [49, B, embed_size]