import torch
import torch.nn as nn
import torch.nn.functional as F

class FEBlock(nn.Module):
    def __init__(self, embed_size=1):
        super(FEBlock, self).__init__()
        # 1x1 ~ 30x30 Convolution layers 생성 (Padding X)
        # self.convs = nn.ModuleList([nn.Conv2d(1, embed_size, kernel_size=n, padding=0) for n in range(1, 31)])
        # self.fc = nn.ModuleList([nn.Linear((30-n)**(30-n)embed_size, 30*30*embed_size) for n in range(1, 31)])
        self.numbers = [1, 2, 3, 5, 7, 9, 11, 13, 15, 25, 30]
        self.stages = nn.ModuleList([])
        for n in self.numbers:
            self.stages.append(nn.Sequential(
                nn.Conv2d(1, n, kernel_size=n, padding=0),
                nn.Flatten(1),
                nn.Linear((30-n+1)**2*n, 30*30*embed_size)
            ))

    def forward(self, x):
        features = []
        for stage in self.stages:
            features.append(stage(x).unsqueeze(1))  # (batch, 1, 1, n*n*embed_size)
            # print(features[-1].shape)
        return torch.stack(features, dim=1)  # (batch, 30, n*n*embed_size)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size=1):
        super(SelfAttentionBlock, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 30*30*embed_size))
        self.self_attn = nn.MultiheadAttention(embed_dim=30*30*embed_size, num_heads=4)

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # print(cls_tokens.shape)
        # print(x.shape)
        x = x.squeeze(2)
        # print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 31, 30*30*embed_size)
        x, _ = self.self_attn(x, x, x)
        return x[:, 0].unsqueeze(1)  # (batch, 1, 30*30*embed_size)

class HeadBlock(nn.Module):
    def __init__(self, embed_size=1, num_classes=11):
        super(HeadBlock, self).__init__()
        self.fc1 = nn.Linear(30*30*embed_size, 30*30*embed_size*2)
        self.fc2 = nn.Linear(30*30*embed_size*2, 30*30*embed_size)
        self.fc3 = nn.Linear(30*30*embed_size, 30*30)
        self.conv = nn.Conv2d(1, num_classes, kernel_size=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 30, 30)  # Reshape to (batch, 1, 30, 30)
        x = self.conv(x)
        x = self.log_softmax(x)
        return x

class BWNet_MAML(nn.Module):
    def __init__(self, embed_size=1):
        super(BWNet_MAML, self).__init__()
        self.fe_block = FEBlock(embed_size=embed_size)
        self.self_attn_block = SelfAttentionBlock(embed_size=embed_size)
        self.head_block = HeadBlock(embed_size=embed_size)
        

    def forward(self, ex_input):
        features = self.fe_block(ex_input)  # (batch, 30, 30*30*embed_size)
        cls_feature = self.self_attn_block(features)  # (batch, 1, 30*30*embed_size)
        out = self.head_block(cls_feature)  # (batch, 1, 30, 30)
        #out = self.fc_final(out)  # (batch, 11, 30, 30)
        return out

# input_tensor = torch.randn(1, 1, 30, 30)  # 입력 텐서 예시
# example_input = torch.randn(10, 1, 30, 30)  # 입력 텐서 예시
# # example_output = torch.randn(10, 1, 30, 30)  # 출력 텐서 예시

# model = BWNet_MAML(embed_size=2)
# output = model(example_input)

# print(output.shape)  # 최종 출력 크기를 확인
