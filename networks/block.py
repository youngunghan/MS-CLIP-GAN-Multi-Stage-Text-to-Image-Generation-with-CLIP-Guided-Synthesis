import torch
import torch.nn as nn
import torch.nn.functional as F

'''
A larger number of channels = ability to learn more features
A larger spatial size = ability to capture finer structures from the start

Advantages of kernel_size=3, stride=1, padding=1:
    - Input and output sizes remain the same (due to padding=1)
    - Provides an appropriate receptive field
    - Fewer parameters compared to 5x5 or 7x7 kernels
    - Widely used as a standard since VGGNet
'''
def CBR2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False, norm='bn', act='relu', trans=False):
    layers = []
    if not trans:
        layers += [nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    else:
        layers += [nn.ConvTranspose2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        
    if norm == 'bn': layers += [nn.BatchNorm2d(out_chans)]
    
    if act == 'relu':
        layers += [nn.ReLU()]
    elif act == 'leakyrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    elif act == 'tanh':
        layers += [nn.Tanh()]
    elif act == 'silu':
        layers += [nn.SiLU()]
        
    cbr = nn.Sequential(*layers)
    return cbr

def LBR(in_chans, out_chans, bias=False, norm='bn', act='relu'):
    layers = []
    layers += [nn.Linear(in_chans, out_chans, bias=bias)]
        
    if norm == 'bn': 
        layers += [nn.BatchNorm1d(out_chans)]
    elif norm == 'ln':
        layers += [nn.LayerNorm(out_chans)]
        
    if act == 'relu': 
        layers += [nn.ReLU()]
    elif act == 'leakyrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        
    lbr = nn.Sequential(*layers)
    return lbr

class ResBlock(nn.Module): # Resnet Module
    expansion = 1
    
    def __init__(self, in_chans, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.in_chans = in_chans
        self.layer = nn.Sequential(
            CBR2d(self.in_chans, self.in_chans),
            CBR2d(self.in_chans, self.in_chans, act=False),
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    
class EncBlock(nn.Module): # Encoder Module
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layer = CBR2d(in_chans, out_chans, kernel_size=4, stride=2, act='leakyrelu')
        
    def forward(self, x):
        return self.layer(x)
            
class DecBlock(nn.Module): # Decoder Module
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layer = CBR2d(in_chans, out_chans, act='leakyrelu')
        
    def forward(self, x):
        return self.layer(x)
    
class SemanticSpatialAwareBlock(nn.Module):
    def __init__(self, in_channels, text_dim):
        super().__init__()
        self.in_channels = in_channels
        self.text_dim = text_dim
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Text projection - 텍스트 특징을 채널 수에 맞게 변환
        self.text_proj = nn.Sequential(
            nn.Flatten(),  # 텐서를 평탄화
            nn.Linear(text_dim, in_channels),
            nn.ReLU()
        )
        
    def forward(self, x, text_embed):
        '''
        Inputs:
            x: image feature map [B, C, H, W] e.g., [32, 64, 64, 64]
            text_embed: text embedding [B, text_dim] e.g., [32, 256]
        Outputs:
            out: text-enhanced feature map [B, C, H, W]
        '''
        B, C, H, W = x.shape
        
        # Project and expand text features
        text = text_embed.reshape(B, -1)
        text = self.text_proj(text)
        text = text.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # Channel attention
        channel_map = self.channel_attn(x)
        x = x * channel_map
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        spatial_map = self.spatial_attn(spatial)
        
        # Final fusion
        out = x * spatial_map + text
        return out
    # (EIGGAN) https://www.techscience.com/cmc/v80n1/57408/html
    # (SSAGAN) https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable weight

    def forward(self, x):
        """
        inputs:
            x: input feature maps (B, C, H, W)
        returns:
            out: attention value + input feature
            attention: B, H*W, H*W
        """
        B, C, H, W = x.size()
        
        # Reshape query, key, value
        query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # B, H*W, C//8
        key = self.key(x).view(B, -1, H*W)  # B, C//8, H*W
        value = self.value(x).view(B, -1, H*W)  # B, C, H*W
        
        # Calculate attention scores
        attention = torch.bmm(query, key)  # B, H*W, H*W
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, H*W
        out = out.view(B, C, H, W)
        
        # Add residual connection with learnable weight
        out = self.gamma * out + x
        
        return out