import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(2)  # Flatten the spatial dimensions (H, W)
    
    def forward(self, x):
        x = self.conv(x)  # Apply convolution
        x = self.flatten(x)  # Flatten to (batch_size, embed_dim, H*W)
        x = x.transpose(1, 2)  # Transpose to (batch_size, H*W, embed_dim)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, seq_len=30*30, add_cls_token=True):
        super(FeatureExtractor, self).__init__()
        self.embedding = ConvEmbedding(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if add_cls_token else None
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len+1 if add_cls_token else seq_len, embed_dim))
        self.attention = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(self.attention, num_layers=1)
    
    def forward(self, x):
        # x shape: (batch_size, in_channels, H, W)
        x = self.embedding(x)  # Convert to tokens
        # print(x.shape)
        if self.cls_token is not None:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # Prepend cls_token
        # print(x.shape)
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)  # Apply self-attention
        # print(x.shape)
        cls_feature = x[:, 0, :]  # Extract cls token feature
        token_features = x[:, 1:, :]  # Extract other token features
        
        return cls_feature, token_features

class SelfAttentionWithThreeTokens(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4):
        super(SelfAttentionWithThreeTokens, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads)
        self.new_cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))  # New Cls token
    
    def forward(self, example_input_cls, example_output_cls):
        # example_input_cls, example_output_cls: Shape (batch_size, embed_dim)
        
        # Expand new_cls_token to match batch size
        batch_size = example_input_cls.size(0)
        new_cls_token_expanded = self.new_cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        
        # Combine all Cls tokens: shape (3, batch_size, embed_dim)
        combined_cls_tokens = torch.cat([new_cls_token_expanded, 
                                         example_input_cls.unsqueeze(1), 
                                         example_output_cls.unsqueeze(1)], dim=1)
        # print("c")
        # print(combined_cls_tokens.shape)
        # Apply self-attention
        combined_cls_tokens = combined_cls_tokens.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(combined_cls_tokens, combined_cls_tokens, combined_cls_tokens)
        
        # Return the output as the new Cls token, shape: (batch_size, embed_dim)
        return attn_output[0]  # The first token corresponds to the new_cls_token


class CombineModule(nn.Module):
    def __init__(self, feature_dim):
        super(CombineModule, self).__init__()
        
        # Self-Attention Layer for combining causals
        self.self_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.new_cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))  # New Cls token
        # Fully connected layer to produce the final causal representation
        self.fc = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, causals):
        # causals: shape (num_causals, batch_size, feature_dim)
        causals = causals.unsqueeze(1)  # Add a sequence dimension
        #print("causals",causals.shape)
        new_cls_token_expanded = self.new_cls_token.expand(1, -1, -1)
        #print("new_cls_token_expanded",new_cls_token_expanded.shape)
        
        # cls_causals = torch.cat([new_cls_token_expanded, 
        #                                  causals], dim=0)

        attn_output, _ = self.self_attention(new_cls_token_expanded, causals, causals)
        #print("attn_output",attn_output.shape)
        # Mean pooling over the sequence dimension (num_causals)
        combined_causal = attn_output.squeeze(1)  # Shape: (batch_size, feature_dim)
        
        # Pass through a fully connected layer to get the final causal representation
        final_causal = self.fc(combined_causal)  # Shape: (batch_size, feature_dim)
        
        return final_causal


class Head(nn.Module):
    def __init__(self, embed_dim=128, output_dim=1, seq_len=30*30, num_classes=11):
        super(Head, self).__init__()
        self.num_classes = num_classes
        
        # FC layers to transform features
        self.fc1 = nn.Linear(embed_dim, embed_dim)  # Project final_causal to match cls_token
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim*2, seq_len)  # Combine Cls and final_causal
        # self.fc4 = nn.Linear(embed_dim + (seq_len * embed_dim), seq_len)  # Combine with flattened token features
        
        # Convolution layer to map to class logits
        self.conv = nn.Conv2d(1, num_classes, kernel_size=1)  # 1x1 Convolution for class logits
        
        # Output reshape (no upsample since the size is already 30x30)
        self.output_reshape = nn.Sequential(
            nn.Unflatten(1, (1, int(seq_len ** 0.5), int(seq_len ** 0.5)))  # (batch_size, 1, 30, 30)
        )
        
        # LogSoftmax for multi-class classification
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, cls_token, token_features, final_causal):
        # Project final_causal to the same dimension as cls_token
        final_causal_proj = self.fc1(final_causal)
        #print("final_causal_proj",final_causal_proj.shape)
        # Combine cls_token and final_causal_proj
        #cls_combined = torch.cat((cls_token, final_causal_proj), dim=-1)
        
        cls_combined = self.fc2(cls_token)
        #print("cls_combined",cls_combined.shape)
        # Flatten token features
        #token_features_flat = token_features.view(token_features.size(0), -1)
        
        # Combine cls_combined with token_features_flat
        #combined_features = torch.cat((cls_combined, token_features_flat), dim=-1)
        combined_features = torch.cat((final_causal_proj, cls_combined), dim=-1)
        # print("combined_features",combined_features.shape)
        x = self.fc3(combined_features)  # (batch_size, seq_len)
        # print("x",x.shape)
        # Reshape to (batch_size, 1, 30, 30)
        x = self.output_reshape(x)
        # print("x",x.shape)
        # Apply convolution to get class logits
        logits = self.conv(x)  # (batch_size, num_classes, 30, 30)
        # print("logits",logits.shape)
        # Apply log_softmax to get class probabilities
        output = self.log_softmax(logits)  # (batch_size, num_classes, 30, 30)
        
        return output
    

class BWNet(nn.Module):
    def __init__(self, feature_dim=128, num_examples=5):
        super(BWNet, self).__init__()
        self.feature_extractor = FeatureExtractor(embed_dim=feature_dim)
        self.causal_inference = SelfAttentionWithThreeTokens(feature_dim=feature_dim)
        self.combine_module = CombineModule(feature_dim=feature_dim)
        self.head = Head()

    def forward(self, input_tensor, example_input, example_output):
        # 입력 및 예제 텐서를 30x30으로 패딩
        
        # Feature extraction
        cls_feature, _ = self.feature_extractor(input_tensor)
        ex_input_cls_feature, _ = self.feature_extractor(example_input)
        ex_output_cls_feature, _ = self.feature_extractor(example_output)
        
        # Causal inference
        causals = self.causal_inference(ex_input_cls_feature, ex_output_cls_feature)
        
        # Combine module
        final_causal = self.combine_module(causals)
        
        # Head
        output = self.head(cls_feature, _, final_causal)
        
        # Padding 제거
        # output = self.remove_padding(output)
        
        return output