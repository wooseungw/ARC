import typing
from module import Attention, PreNorm, FeedForward, Residual
import torch
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import f1_score
from torchmetrics import F1Score
import torch.distributed as dist
from collections import defaultdict
from torchvision import transforms, datasets
import timm  # PyTorch Image Models 라이브러리
import numpy as np

from torchprofile import profile_macs

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        #[0]번은 어텐션 벨류, [1]번은 어텐션 가중치
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, k):
        x = q
        y = k
        inp_x = self.layer_norm_1(x)
        inp_y = self.layer_norm_1(y)
        x = x + self.attn(inp_x, inp_y, inp_y)[0]
        #print("Cross attention",x.shape)
        x = x + self.linear(self.layer_norm_2(x))
        
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
        head_num_layers=2,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.embedding = nn.Sequential(
            nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        # Layers/Networks
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.ffn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    def get_embedding(self, x):
        # Preprocess input
        x = self.embedding(x)
        
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Perform classification prediction
        cls = x[0]
        #print("Vit의 cls 모양",cls.shape)
        return cls

    def get_value(self,x):
        
        x = self.embedding(x)
        
        B, T, _ = x.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        return x
    
    def forward(self, x):
        # Preprocess input
        #x = img_to_patch(x, self.patch_size)
        x = self.embedding(x)
        B, T, _ = x.shape
        #x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)    
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

    
# if __name__ == "__main__":
    
#     img = torch.ones([1, 4, 3, 128, 128])
#     print("Is x a list of tensors?", all(isinstance(item, torch.Tensor) for item in img))
#     print("Length of x:", len(img))
    
#     #패치 사이즈
#     p_s = 8
#     model_kwargs = {
#         'embed_dim': (128),
#         'hidden_dim': (128)*4,
#         'num_channels': 3,
#         'num_heads': 8,
#         'num_layers': 6,
#         'num_classes': 3,
#         'patch_size': p_s,
#         'num_patches': (128//p_s)**2,
#         'dropout': 0.1,
#         'head_num_layers': 2 
#     }
#     #model = ViT_QA2(model_kwargs,lr=1e-3)
#     #model = ViT_cls_cross14(model_kwargs,lr=1e-3)
#     model = ViT_QA_cos(model_kwargs,lr=1e-3)
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     print('Trainable Parameters: %.3fM' % parameters)
#     flops = profile_macs(model, img)
#     print("flops: ",flops)
#     out = model(img)
    
#     print("Shape of out :", out.shape)      # [B, num_classes]
#     import time
#     # 모델을 평가 모드로 설정
#     model.eval()

    # # 추론 시간 측정
    # start_time = time.time()
    # with torch.no_grad():
    #     output = model(img)
    # end_time = time.time()

    # # 추론에 걸린 시간 계산
    # inference_time = end_time - start_time
    # print(f"Inference Time: {inference_time} seconds")