import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from math import sqrt
'''
2. LayerNorm & PreNorm
LayerNorm은 채널 차원을 따라 정규화를 수행합니다. 이는 학습 과정을 안정화하고 가속화하는 데 도움을 줍니다.
PreNorm은 주어진 함수(fn, 예: 어텐션 또는 피드포워드 네트워크)를 적용하기 전에 입력을 정규화합니다.
'''    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
    
    
    
class Embedding(nn.Module):
    def __init__(self, embed_channel,kernel_size, stride, padding):
        super(Embedding, self).__init__()
        self.embedding = nn.Conv2d(1, 
                                   embed_channel, 
                                   kernel_size=kernel_size, 
                                   stride =stride, 
                                   padding= padding)
        
        
    def forward(self, x):
        x = self.embedding(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, embed_channel):
        super(Encoder, self).__init__()  # 부모 클래스의 __init__ 메서드 호출
        self.encoder = nn.TransformerEncoder(d_model=embed_channel, nhead=8)
        
    def forward(self, x):
        return self.encoder(x)
    
if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device  # GPU 사용 가능 여부 및 MPS 지원 여부 확인
    
    x = torch.randn(3, 1, 30, 30)
    kwargs={
        'kernel_size':3, 
        'stride':1, 
        'padding':1,
        'embed_channel':128
        }
    
    model = Embedding(**kwargs)
    x = model(x)
    print(x.shape)
    x = torch.randn(3, 900, 256)
    model = Encoder(embed_channel=256)
    x = model(x)
    print(x.shape)


 
        

class ARC_Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = Head