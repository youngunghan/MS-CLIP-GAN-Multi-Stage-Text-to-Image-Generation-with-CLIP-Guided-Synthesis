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
    """Predict text-space features from image features.

    ``image_only`` zeroes the historical condition channels before the alignment
    head.  This removes the trivial text-to-text shortcut while deliberately
    retaining the old convolution's input shape, so old state_dicts still load.
    ``legacy_conditioned`` reproduces the historical concatenation exactly.
    """

    VALID_MODES = frozenset({'image_only', 'legacy_conditioned'})

    def __init__(self, in_chans, cond_dim, text_emb_dim, alignment_mode='image_only'):
        super(AlignCondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.cond_dim = cond_dim
        self.text_emb_dim = text_emb_dim
        self.set_alignment_mode(alignment_mode)

        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]
        self.align_net = nn.Sequential(
            CBR2d(self.in_chans * 8 + self.cond_dim, self.in_chans * 8, act="silu"),
            #CBR2d(self.in_chans * 8, self.text_emb_dim, kernel_size=2, stride=2, norm=False, act=False),
            CBR2d(self.in_chans * 8, self.text_emb_dim, kernel_size=4, stride=4, norm=False, act=False),
            # nn.Identity()
        )
    def set_alignment_mode(self, alignment_mode):
        if alignment_mode not in self.VALID_MODES:
            raise ValueError(
                f"alignment mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {alignment_mode!r}"
            )
        self.alignment_mode = alignment_mode

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
        if self.alignment_mode == 'image_only':
            # Preserve the legacy convolution shape but prevent a text-only shortcut.
            c = x.new_zeros(B, self.cond_dim, H, W)
        else:
            c = c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W) # [B, 128, 1, 1] [B, 128, 4, 4]
        x = torch.cat((x, c), dim=1)
        #print(f'torch.cat((x, c), dim = 1).shape: {x.shape}')
        x = self.align_net(x)
        #print(f'align_out.shape: {x.shape}')
        # flatten(1) -> [B, text_emb_dim]; batch-safe unlike squeeze() which would
        # collapse a batch of size 1 ([1, D, 1, 1] -> [D]) and break the contrastive loss.
        align_out = x.flatten(1)
        #print(f'align_out.squeeze.shape: {align_out.shape}')
        return align_out

class Discriminator(nn.Module):
    def __init__(self, img_chans, in_chans, out_chans, condition_dim,
                 clip_text_embedding_dim, curr_stage, device,
                 alignment_mode='image_only'):
        super(Discriminator, self).__init__()
        self.img_chans = img_chans # g_out_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.cond_dim = condition_dim
        self.txt_emb_dim = clip_text_embedding_dim
        self.curr_stage = curr_stage
        self.device = device
        self.alignment_mode = alignment_mode

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
        return AlignCondDiscriminator(
            self.in_chans, self.cond_dim, self.txt_emb_dim,
            alignment_mode=self.alignment_mode
        )

    def set_alignment_mode(self, alignment_mode):
        """Switch compatibility semantics without changing checkpoint parameters."""
        self.align_cond_discriminator.set_alignment_mode(alignment_mode)
        self.alignment_mode = alignment_mode

    def _extract_features(self, img):
        """Run the shared image trunk exactly once for one discriminator view."""
        features = self.feature_net(img)
        if features.shape[-1] == 16:
            features = self.attention(features)
        return self.aec_net(features)

    def forward(self,
                img,
                condition=None,  # for conditional loss (mu)
                compute_alignment=True,
                mismatched_condition=None,
                compute_unconditional=False,
                return_details=False,
                ):
        '''
        Inputs:
            img: fake/real image, shape [3, H, W]
            condition: mu extracted from CANet, shape [projection_dim]
            compute_alignment: skip the alignment head when its output is unused
            mismatched_condition: optional wrong-text condition for the same image
            compute_unconditional: include the unconditional head in detailed output
            return_details: return a dict containing all requested heads
        Outputs:
            out: fake/real prediction result (common output of discriminator)
            align_out: f_real/f_fake extracted from self.align_cond_discriminator for contrastive learning
            With return_details=True, a tensor-only dict of the requested heads is
            returned instead; this shape is compatible with DataParallel gather.
        '''
        prev_out = self._extract_features(img)

        if return_details:
            if mismatched_condition is not None and condition is None:
                raise ValueError('mismatched_condition requires a matched condition')
            details = {}
            if condition is not None:
                details['conditional'] = torch.sigmoid(
                    self.cond_discriminator(prev_out, condition).view(-1)
                )
                if compute_alignment:
                    details['alignment'] = self.align_cond_discriminator(
                        prev_out, condition
                    )
            if mismatched_condition is not None:
                details['mismatched_conditional'] = torch.sigmoid(
                    self.cond_discriminator(prev_out, mismatched_condition).view(-1)
                )
            if compute_unconditional or condition is None:
                details['unconditional'] = torch.sigmoid(
                    self.uncond_discriminator(prev_out).view(-1)
                )
            return details

        align_out = None
        if condition is None:
            out = self.uncond_discriminator(prev_out).view(-1)
        else:
            out = self.cond_discriminator(prev_out, condition).view(-1)
            if compute_alignment:
                align_out = self.align_cond_discriminator(prev_out, condition)

        out = torch.sigmoid(out)
        return out, align_out
