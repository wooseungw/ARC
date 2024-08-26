import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from math import sqrt

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
    
if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device  # GPU 사용 가능 여부 및 MPS 지원 여부 확인
    
    x = torch.randn(3, 1, 30, 30)
    kwargs={
        'kernel_size':3, 
        'stride':1, 
        'padding':1,
        'embed_channel':256
        }
    
    model = Embedding(**kwargs)
    
    print(model(x).shape)

class Encoder(nn.Module):
    def __init__():
        
    def forward(self,x):
        return x
    
# class Decoder(nn.Module):
#     def __init__(self, decoder_dim,):
    
#     def forward(self,x):
#         return x
    
# class Head(nn.Module):
#     def __init__(decoder_dim, num_classes):
        
#     def forward(self,x):
        
#         return x
        

class ARC_Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = Head