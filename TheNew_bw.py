import torch
import torch.nn as nn
import torch.nn.functional as F

class FEBlock(nn.Module):
    def __init__(self, embed_size=1):
        super(FEBlock, self).__init__()
        self.stages = nn.ModuleList([])
        
        for n in range(1, 31):  # 1부터 30까지 크기 변화
            # 1xn and nx1 convolutions with padding to maintain approximately the same size
            conv_1xn = nn.Conv2d(1, 1, kernel_size=(1, n), padding=(0, (n - 1) // 2), groups=1)  # 1xn convolution
            conv_nx1 = nn.Conv2d(1, embed_size, kernel_size=(n, 1), padding=((n - 1) // 2, 0), groups=1)  # nx1 convolution
            
            self.stages.append(nn.Sequential(
                conv_1xn,
                conv_nx1,
            ))

        self.conv1x1 = nn.Conv2d(30*embed_size, 1, kernel_size=1)  # 30채널을 1채널로 변환

    def forward(self, x):
        features = []
        for stage in self.stages:
            out = stage(x)  # 각 stage를 통해 결과를 얻음
            
            # 최종 출력 크기를 강제로 30x30으로 맞추기 위해 패딩 추가
            # out.shape[2] (height)와 out.shape[3] (width)를 확인해 30x30이 아니면 F.pad로 크기를 맞춤
            if out.shape[2] != 30 or out.shape[3] != 30:
                pad_h = 30 - out.shape[2]
                pad_w = 30 - out.shape[3]
                # 양 끝에 필요한 만큼의 패딩을 추가
                out = F.pad(out, (0, pad_w, 0, pad_h))  # (왼쪽, 오른쪽, 위쪽, 아래쪽)
            
            features.append(out)  # 각 결과를 리스트에 저장
        
        # 30개의 feature를 채널 차원에서 결합하여 30채널로 변환
        cat_features = torch.cat(features, dim=1)
        out = self.conv1x1(cat_features)
        return out



class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 30*30))
        self.layernorm1 = nn.LayerNorm(30*30)
        self.self_attn = nn.MultiheadAttention(embed_dim=30*30, num_heads=4)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # (batch, 1, 30, 30) -> (batch, 1, 30*30)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        #print(cls_tokens.shape)
        #print(x.shape)
        x = x.squeeze(2)
        #print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 31, 30*30*embed_size)
        #print(x.shape)
        x = self.layernorm1(x)
        #print(x.shape)
        x, _ = self.self_attn(x, x, x)
        #print(x.shape)
        return x[:, 0].unsqueeze(1)  # (batch, 1, 30*30*embed_size)

class HeadBlock(nn.Module):
    def __init__(self, embed_size=1, num_classes=11):
        super(HeadBlock, self).__init__()
        self.fc1 = nn.Linear(30*30, 30*30*2)
        self.fc2 = nn.Linear(30*30*2, 30*30)
        self.fc3 = nn.Linear(30*30, 30*30)
        self.conv = nn.Conv2d(1, num_classes, kernel_size=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.2)(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.2)(x)
        x = self.fc3(x)
        x = x.view(-1, 1, 30, 30)  # Reshape to (batch, 1, 30, 30)
        x = self.conv(x)
        x = self.log_softmax(x)
        return x

class BWNet_MAML(nn.Module):
    def __init__(self, embed_size=1):
        super(BWNet_MAML, self).__init__()
        self.fe_block = FEBlock(embed_size=embed_size)
        self.self_attn_block = SelfAttentionBlock()
        self.head_block = HeadBlock()
        

    def forward(self, ex_input):
        features = self.fe_block(ex_input)  # (batch, 30, 30*30*embed_size)
        cls_feature = self.self_attn_block(features)  # (batch, 1, 30*30*embed_size)
        out = self.head_block(cls_feature)  # (batch, 1, 30, 30)
        # out = self.fc_final(out)  # (batch, 11, 30, 30)
        return out

# input_tensor = torch.randn(1, 1, 30, 30)  # 입력 텐서 예시
# example_input = torch.randn(10, 1, 30, 30)  # 입력 텐서 예시
# # example_output = torch.randn(10, 1, 30, 30)  # 출력 텐서 예시

# model = BWNet_MAML(embed_size=2)
# output = model(example_input)

# print(output.shape)  # 최종 출력 크기를 확인

