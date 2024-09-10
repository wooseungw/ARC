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
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class Decoder(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Decoder, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

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
        self.decoder = Decoder(gate_channels = dim * len(kernel_stride_padding))  
        self.head = Head(input_dim=dim * len(kernel_stride_padding), dim=dim, num_classes=num_classes)

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
        x = self.decoder(x)
        # print("ch_sp:",x.shape)
        
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
