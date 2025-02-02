import torch
import torch.nn as nn
from .block import *

class ConditioningAugmention(nn.Module):
    def __init__(self, c_txt_dim, cond_dim, device):
        super(ConditioningAugmention, self).__init__()
        self.device = device
        self.c_txt_dim = c_txt_dim  
        self.c_hat_txt_dim = cond_dim      
        self.layer = LBR(self.c_txt_dim, self.c_hat_txt_dim * 2, norm=False)

    def forward(self, x):
        '''
        Inputs:
            x: CLIP text embedding c_txt
        Outputs:
            condition: augmented text embedding c_hat_txt
            mu: mean of x extracted from self.layer. 
            log_sigma: log(sigma) of x extracted from self.layer.
        '''
        features = self.layer(x)
        mu, log_sigma = features[:, :self.c_hat_txt_dim], features[:, self.c_hat_txt_dim:]
        
        # Reparameterization trick
        epsilon = torch.randn_like(mu).to(mu.device) # z를 mu와 같은 디바이스에 생성
        condition = mu + torch.exp(log_sigma) * epsilon # c_hat_txt

        return condition, mu, log_sigma

class ImageExtractor(nn.Module):
    def __init__(self, in_chans):
        super(ImageExtractor, self).__init__()
        self.in_chans = in_chans
        self.image_net = CBR2d(self.in_chans, 3, norm=False, act='tanh', trans=True)

    def forward(self, x):
        '''
        Inputs:
            x: input tensor, shape [C, H, W]
        Outputs:
            out: output image extracted with self.image_net, shape [3, H, W]
        '''
        return self.image_net(x)

class Generator_type_1(nn.Module):
    def __init__(self, in_chans, input_dim):
        super(Generator_type_1, self).__init__()
        self.in_chans = in_chans # 1024
        self.input_dim = input_dim # cond_dim + noise_dim

        self.mapping_net = self._mapping_net()
        self.upsample_net = self._upsample_net()
        self.image_net = self._image_net()
        
        cond_dim = input_dim - 100  # noise_dim이 100이라고 가정
        self.ssa_blocks = nn.ModuleList([
            SemanticSpatialAwareBlock(in_chans, cond_dim) 
            for _ in range(2)
        ])
    
    def _mapping_net(self):
        # Change the input tensor dimension [projection_dim + noise_dim] into [Ng * 4 * 4]
        initial_dim = self.input_dim       # initial_dim
        hidden_dim = initial_dim           # intermediate_dim
        final_dim = self.in_chans * 4 * 4  # final_dim
        
        lbrs = [LBR(initial_dim, hidden_dim, norm='ln', act='leakyrelu')]
        for _ in range(6):
            lbrs.append( LBR(hidden_dim, hidden_dim, norm='ln', act='leakyrelu') )
        lbrs.append( LBR(hidden_dim, final_dim, norm='ln', act='leakyrelu') )
        return nn.Sequential(*lbrs)
    
        #return LBR(self.input_dim, self.in_chans * 4 * 4, act='leakyrelu')
        #Use StyleGAN's architecture

    def _upsample_net(self):
        '''
        # Use DCGAN's Generator architecture
        # ConvTranspose2d
            C_out = C_in // 2
            H_out = (H_in - 1) * stride - 2 * padding + kernel_size
                kernel_size=4, stride=2, padding=1:
                    4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # Conv2d
            H_out = ((H_in + 2 * padding - kernel_size) // stride) + 1
                    
        # todo
            # Pixel Shuffle
            class PixelShuffleUpsampling(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.conv = nn.Conv2d(
                        in_channels, out_channels*4,
                        kernel_size=3, stride=1, padding=1
                    )
                    self.bn = nn.BatchNorm2d(out_channels*4)
                    self.relu = nn.ReLU()
                    self.shuffle = nn.PixelShuffle(2)
                
                def forward(self, x):
                    x = self.relu(self.bn(self.conv(x)))
                    return self.shuffle(x)
            # Upsample + Conv
            class UpsampleConv(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.upsample = nn.Upsample(
                        scale_factor=2, 
                        mode='bilinear',
                        align_corners=False
                    )
                    self.conv = nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=3, stride=1, padding=1
                    )
                    self.bn = nn.BatchNorm2d(out_channels)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.upsample(x)
                    return self.relu(self.bn(self.conv(x)))

        '''
        # Change the input tensor dimension [Ng, 4, 4] into [Ng/16, 64, 64]
        cbr2ds = []
        in_chans = self.in_chans
        for _ in range(4):  
            out_chans = in_chans // 2
            cbr2ds.append( CBR2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1, act='relu', trans=True) ) 
            in_chans = out_chans
        return nn.Sequential(*cbr2ds)
        
    def _image_net(self):
        return ImageExtractor(self.in_chans // 2**4)
        
    def forward(self, cond, noise):
        '''
        Inputs:
            cond: text embedding tensor [B, projection_dim] e.g., [32, 256]
            noise: gaussian noise tensor [B, noise_dim] e.g., [32, 100]
        Outputs:
            out: upsampled feature map [B, Ng/16, 64, 64] e.g., [32, 64, 64, 64]
            out_image: generated image [B, 3, 64, 64] e.g., [32, 3, 64, 64]
        '''
        # Concatenate condition and noise
        cond_noise = torch.cat((cond, noise), dim=1)  # [B, 356]
        
        # Transform to initial feature map through mapping network
        x = self.mapping_net(cond_noise).view(-1, self.in_chans, 4, 4)  # [B, 1024, 4, 4]
        
        # Apply SSA blocks to inject text information
        for ssa_block in self.ssa_blocks:
            x = ssa_block(x, cond)
        
        # Progressive upsampling: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        out = self.upsample_net(x)
        out_image = self.image_net(out)
        return out, out_image


class Generator_type_2(nn.Module):
    def __init__(self, in_chans, cond_dim, num_res_layer, device):
        super(Generator_type_2, self).__init__()
        self.device = device

        self.in_chans = in_chans
        self.cond_dim = cond_dim
        
        self.joint_net = self._joint_net()
        self.res_net = nn.ModuleList( [self._res_net() for _ in range(num_res_layer)] )
        self.upsample_net = self._upsample_net()
        self.image_net = self._image_net()   
        
        self.ssa_blocks = nn.ModuleList([
            SemanticSpatialAwareBlock(in_chans, cond_dim)
            for _ in range(2)
        ])     

    def _joint_net(self):
        # Just change the channel size of input tensor into self.in_chans
        # The input channel of joining_layer should consider applying the condition vector as attention.
        return CBR2d(self.in_chans + self.cond_dim, self.in_chans)
    
    def _res_net(self):
        return ResBlock(self.in_chans)

    def _upsample_net(self):
        # Change the input tensor dimension [C, H, W] into [C/2, 2H, 2W]
        return CBR2d(self.in_chans, self.in_chans // 2, kernel_size=4, stride=2, padding=1, trans=True)
    
    def _image_net(self):
        return ImageExtractor(self.in_chans // 2)

    def forward(self, cond, prev_out):
        '''
        Inputs:
            cond: text embedding tensor [B, cond_dim] e.g., [32, 256]
            prev_out: previous stage feature map [B, C, H, W] e.g., [32, 64, 64, 64]
        Outputs:
            out: upsampled feature map [B, C/2, 2H, 2W] e.g., [32, 32, 128, 128]
            out_image: generated image [B, 3, 2H, 2W] e.g., [32, 3, 128, 128]
        '''
        B, _, H, W = prev_out.shape
        
        # Reshape and expand condition to spatial dimensions
        cond = cond.reshape(B, -1)
        cond_spatial = cond.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # Combine features with condition
        feat = torch.cat([prev_out, cond_spatial], dim=1)
        out = self.joint_net(feat)
        
        # Apply SSA blocks and residual processing
        for ssa in self.ssa_blocks:
            out = ssa(out, cond)
        
        for res_block in self.res_net:
            out = res_block(out)
            
        # Final upsampling and image generation
        out = self.upsample_net(out)
        out_image = self.image_net(out)
        return out, out_image

class Generator(nn.Module):
    def __init__(self, in_chans, out_chans, noise_dim, cond_dim, clip_emb_dim, num_stage, device):
        super(Generator, self).__init__()
        self.device = device
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        self.input_dim = self.noise_dim + self.cond_dim
        self.c_txt_dim = clip_emb_dim        

        self.num_stage = num_stage
        self.num_res_layer_type2 = 2  # NOTE: you can change this

        # return layers
        self.cond_aug = self._conditioning_augmentation()
        self.g_layer = nn.ModuleList( [self._stage_generator(i) for i in range(self.num_stage)] )

    def _conditioning_augmentation(self):
        # Define conditioning augmentation of conditonal vector introduced in
        # (StackGAN) https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf
        return ConditioningAugmention(self.c_txt_dim, self.cond_dim, self.device)

    def _stage_generator(self, i):
        '''
        Return the class instance of Generator_type_1 or Generator_type_2 class
        Stage i generator's self.in_chans = stage i-1 generator's 'out' tensor's channel size
        '''
        if i == 0:
            return Generator_type_1(self.in_chans, self.input_dim)
        else:
            prev_chans = self.in_chans // (2 ** 4 << (i - 1)) # (16 * (2 ** (i - 1)))
            return Generator_type_2(prev_chans, self.cond_dim, self.num_res_layer_type2, self.device)

    def forward(self, txt_emb, noise):
        '''
        Inputs:
            text_embedding: c_txt
            z: gaussian noise sampled from N(0, 1)
        Outputs:
            fake_images: List that containing the all fake images generated from each stage's Generator
            mu: mean of c_txt extracted from CANet
            log_sigma: log(sigma) of c_txt extracted from CANet
        '''   
        cond, mu, log_sigma = self.cond_aug(txt_emb)
        
        prev_out = None
        fake_images = []
        for i in range(self.num_stage):
            if i == 0:
                out, out_image = self.g_layer[i](cond, noise)
            else:
                out, out_image = self.g_layer[i](cond, prev_out)
            prev_out = out
            fake_images.append(out_image)
        return fake_images, mu, log_sigma