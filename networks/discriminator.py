import torch
import torch.nn as nn
from .block import *

class UncondDiscriminator(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UncondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Change the input tensor dimension [8Nd, 4, 4] into [1, 1, 1]
        self.uncond_layer = CBR2d(self.in_chans * 8, self.out_chans, kernel_size=4, stride=4, padding=0, norm=False, act=False)

    def forward(self, x):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
        Outputs:
            uncond_out: output tensor extracted frm self.uncond_layer, shape [1, 1, 1]
        '''
        #print(f'uncond_in.shape: {x.shape}')
        uncond_out = self.uncond_layer(x)
        #print(f'uncond_out.shape: {uncond_out.shape}')
        return uncond_out


class CondDiscriminator(nn.Module):
    def __init__(self, in_chans, cond_dim, out_chans):
        super(CondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.cond_dim = cond_dim
        self.out_chans = out_chans
        
        # Change the input tensor dimension [8Nd + cond_dim, 4, 4] into [1, 1, 1]
        # self.cond_layer = CBR2d(self.in_chans * 8 + self.cond_dim, self.out_chans, kernel_size=4, stride=4, padding=0, act='leakyrelu')
        self.cond_layer = nn.Sequential(
            CBR2d(self.in_chans * 8 + self.cond_dim, self.in_chans * 8, act='leakyrelu'),
            CBR2d(self.in_chans * 8, self.out_chans, kernel_size=4, stride=4, padding=0, norm=False, act=False)
        )

    def forward(self, x, c):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            cond_out: output tensor extracted frm self.cond_layer, shape [1, 1, 1]
        '''
        B, _, H, W = x.shape 
        #print(f'c.view(B, self.cond_dim, 1, 1).shape: {c.view(B, self.cond_dim, 1, 1).shape}')
        #print(f'c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W).shape: {c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W).shape}')
        c = c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W) # [B, 128, 1, 1] [B, 128, 4, 4]
        x = torch.cat((x, c), dim = 1) # [B, 512, 4, 4] + [B, 128, 4, 4] -> [B, 640, 4, 4]
        cond_out = self.cond_layer(x)
        #print(f'torch.cat((x, c), dim = 1).shape: {x.shape}')
        #print(f'cond_out.shape: {cond_out.shape}')
        return cond_out

class AlignCondDiscriminator(nn.Module):
    def __init__(self, in_chans, cond_dim, text_emb_dim):
        super(AlignCondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.cond_dim = cond_dim
        self.text_emb_dim = text_emb_dim

        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]
        self.align_net = nn.Sequential(
            CBR2d(self.in_chans * 8 + self.cond_dim, self.in_chans * 8, act="silu"),
            #CBR2d(self.in_chans * 8, self.text_emb_dim, kernel_size=2, stride=2, norm=False, act=False),
            CBR2d(self.in_chans * 8, self.text_emb_dim, kernel_size=4, stride=4, norm=False, act=False),
            # nn.Identity()
        )
    def forward(self, x, c):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            align_out: output tensor extracted frm self.align_layer, shape [clip_embedding_dim]          
        '''
        B, _, H, W = x.shape
        #c = c.view(-1, self.cond_dim, 1, 1).expand(-1, -1, 4, 4)
        #print(f'c.view(B, self.cond_dim, 1, 1).shape: {c.view(B, self.cond_dim, 1, 1).shape}')
        #print(f'c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W).shape: {c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W).shape}')
        c = c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W) # [B, 128, 1, 1] [B, 128, 4, 4]        
        x = torch.cat((x, c), dim=1)
        #print(f'torch.cat((x, c), dim = 1).shape: {x.shape}')    
        x = self.align_net(x)
        #print(f'align_out.shape: {x.shape}')
        align_out = x.squeeze()
        #print(f'align_out.squeeze.shape: {align_out.shape}')
        return align_out

class Discriminator(nn.Module):
    def __init__(self, img_chans, in_chans, out_chans, condition_dim, clip_text_embedding_dim, curr_stage, device):
        super(Discriminator, self).__init__()
        self.img_chans = img_chans # g_out_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.cond_dim = condition_dim
        self.txt_emb_dim = clip_text_embedding_dim
        self.curr_stage = curr_stage
        self.device = device
        
        self.feature_net = self._feature_extractor()
        self.aec_net = self._aec_net()
        self.uncond_discriminator = self._uncond_discriminator()
        self.cond_discriminator = self._cond_discriminator()
        self.align_cond_discriminator = self._align_cond_discriminator()
        
        # Self-Attention 추가 (512 channels at middle layer)
        self.attention = SelfAttention(in_chans * 8)  # 중간 레이어에서는 채널이 8배로 증가
    
    def _feature_extractor(self):
        # Change the input tensor dimension [3, H, W] into [8Nd, H/16, W/16]
        '''
        Ex (Nd=64):
        [B, 3, 256, 256] -> [B, 64, 128, 128]  # stride=2
        -> [B, 128, 64, 64]                     # channels*2
        -> [B, 256, 32, 32]                     # channels*2
        -> [B, 512, 16, 16]                     # channels*2
        '''
        cbr2ds = []
        in_chans, out_chans = self.img_chans, self.in_chans
        for i in range(4):
            if i == 0: 
                # Initial Feature Extraction
                conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1, bias=False)
                conv = nn.utils.spectral_norm(conv)
                cbr2ds.append(conv)
                if i > 0:  # No norm in first layer
                    cbr2ds.append(nn.BatchNorm2d(out_chans))
                cbr2ds.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                # Downsample Network
                conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1, bias=False)
                conv = nn.utils.spectral_norm(conv)
                cbr2ds.append(conv)
                cbr2ds.append(nn.BatchNorm2d(out_chans))
                cbr2ds.append(nn.LeakyReLU(0.2, inplace=True))
            in_chans, out_chans = out_chans, out_chans * 2
        return nn.Sequential(*cbr2ds)

    def _aec_net(self):
        # Change the input tensor dimension [8Nd, H/16, W/16] into [8Nd, H/64, W/64]
        '''
        Ex (init_chans=64, stage=2):
            Encoder: [B, 512, 16, 16] -> [B, 1024, 8, 8] -> [B, 2048, 4, 4]
            Decoder: [B, 2048, 4, 4] -> [B, 1024, 4, 4] -> [B, 512, 4, 4]
        '''
        if self.curr_stage == 0:
            return nn.Identity()
        
        # Channel scaling factors
        base_chans = self.in_chans * 8
        chan_mults = [2**i for i in range(self.curr_stage + 1)]
        
        layers = []     
        for i in range(self.curr_stage): layers.append( EncBlock(base_chans * chan_mults[i], base_chans * chan_mults[i+1]) )
        for i in range(self.curr_stage - 1, -1, -1): layers.append( DecBlock(base_chans * chan_mults[i+1], base_chans * chan_mults[i]) )
        
    
        return nn.Sequential(*layers)
     
        # Use Encoder-Decoder architecture
    
    def _uncond_discriminator(self):
        # Calcualte conditional loss and unconditional loss like
        # (StackGAN) https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf
        # (StackGAN++) https://arxiv.org/pdf/1710.10916v3.pdf
        # (AttnGAN) https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf
        return UncondDiscriminator(self.in_chans, self.out_chans)

    def _cond_discriminator(self):
        # Calcualte conditional loss and unconditional loss like
        # (StackGAN) https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf
        # (StackGAN++) https://arxiv.org/pdf/1710.10916v3.pdf
        # (AttnGAN) https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf
        return CondDiscriminator(self.in_chans, self.cond_dim, self.out_chans)

    def _align_cond_discriminator(self):
        # Calculate semantic alignment loss like
        # (LAFITE) https://arxiv.org/pdf/2111.13792.pdf
        return AlignCondDiscriminator(self.in_chans, self.cond_dim, self.txt_emb_dim)

    def forward(self,
                img,
                condition=None,  # for conditional loss (mu)
                ):
        '''
        Inputs:
            img: fake/real image, shape [3, H, W]
            condition: mu extracted from CANet, shape [projection_dim]
        Outputs:
            out: fake/real prediction result (common output of discriminator)
            align_out: f_real/f_fake extracted from self.align_cond_discriminator for contrastive learning
        '''
        # Extract features through progressive stages
        x = img
        features = []
        
        # Feature extraction
        prev_out = self.feature_net(img)
        
        # Apply self-attention at the middle layer (16x16)
        if prev_out.shape[-1] == 16:  # Only apply attention at 16x16 resolution
            prev_out = self.attention(prev_out)
            
        prev_out = self.aec_net(prev_out)

        align_out = None
        if condition is None:
            out = self.uncond_discriminator(prev_out).view(-1)
        else:
            out = self.cond_discriminator(prev_out, condition).view(-1)
            align_out = self.align_cond_discriminator(prev_out, condition)

        out = nn.Sigmoid()(out)
        return out, align_out