import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

## 컨볼루션 임베딩
class Embedding(nn.Module):
    def __init__(self, embed_dim, kernel_size, stride, padding, dropout=0.0):
        super(Embedding, self).__init__()
        self.embedding = nn.Conv2d(1, 
                                   embed_dim, 
                                   kernel_size=kernel_size, 
                                   stride=stride, 
                                   padding=padding)
        
    def forward(self, x):
        return self.embedding(x)

## 인코더
class MultiheadAttention(nn.Module):
    def __init__(self, *, dim=124, num_heads=4, dropout=0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
    
    def forward(self, x):
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x

    
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

    
class Encoder(nn.Module):
    def __init__(self, dim=124, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiheadAttention(dim=dim, num_heads=num_heads, dropout=dropout),
                FeedForwardNetwork(dim=dim, hidden_dim=dim * 4, dropout=dropout)
            ]) for _ in range(num_layers)
        ])

    def forward(self, x):
        height = x.shape[2]
        x = rearrange(x, 'b c h w -> b (h w) c')

        for attn, mlp in self.layers:
            x = attn(x)
            x = mlp(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=height)
        return x

    
class Head(nn.Module):
    def __init__(self, input_dim = 256 ,dim=128, num_classes=11):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim , dim, kernel_size=1),  
            nn.Conv2d(dim, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        return self.layers(x)
## 디코더
# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP for both max and avg pooling paths
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# Decoder with Channel and Spatial Attention
class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.channel_att = ChannelAttention(dim)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        spatial_attention = self.spatial_att(x)

        # Use spatial_attention for final output
        return spatial_attention

class ARC_Net(nn.Module):
    def __init__(
        self, 
        *,
        dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=11,
        dropout=0.1,
        kernel_stride_padding=((1, 1, 0), (3, 1, 1))
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList()

        # 각 커널 크기, 스트라이드, 패딩을 튜플로 묶어서 처리
        for (kernel_size, stride, padding) in kernel_stride_padding:
            self.stages.append(
                nn.ModuleList([
                    Embedding(embed_dim=dim, kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout),
                    Encoder(dim=dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
                ])
            )
        
        # Decoder를 통해 attention을 적용하여 결합
        self.decoder = Decoder(dim * len(kernel_stride_padding))  
        self.head = Head(input_dim=dim * len(kernel_stride_padding) + 1, dim=dim, num_classes=num_classes)

    def forward(self, x):
        all_outputs = []

        # 각 커널 크기, 스트라이드, 패딩에 대해 독립적으로 처리
        for embedding, encoder in self.stages:
            scale_x = embedding(x)
            scale_x = encoder(scale_x)
            all_outputs.append(scale_x)

        # 다양한 커널 크기에서 추출된 특징을 병합
        x = torch.cat(all_outputs, dim=1)

        # Decoder를 통해 어텐션 적용 및 결과와 결합
        ch_sp = self.decoder(x)

        # Decoder의 출력을 원래 x에 concatenate
        x = torch.cat([ch_sp,x], dim=1)

        # 병합된 특징을 헤드에 전달
        x = self.head(x)
        return x

    
if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device

    x = torch.randn(13, 1, 30, 30)
    x = x.to(device)
    model_args = {
        'dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'num_classes': 11,
        'dropout': 0.1,
        'kernel_stride_padding': ((1, 1, 0), (3, 1, 1))
    }
        
    model = ARC_Net(**model_args).to(device)
    x = model(x)
    print(x.shape)
