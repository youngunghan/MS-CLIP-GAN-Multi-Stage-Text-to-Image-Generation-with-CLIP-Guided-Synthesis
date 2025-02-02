import warnings
warnings.filterwarnings(action="ignore")

import torch
import torchvision
from utils.utils import *
from config.config import *
#from networks.discriminator import Discriminator
from networks.generator import Generator
from utils.utils import *
from criteria.loss import *
from options.test_options import TestOptions  

@torch.no_grad()
def main():
    # seed_fix(40)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = TestOptions().parse()
    
    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    
    G = Generator(args.g_in_chans, args.g_out_chans, args.noise_dim, args.condition_dim, 
                  args.clip_embedding_dim, args.num_stage, device).to(device)

    # D_lst = [
    #     Discriminator(args.g_out_chans, args.d_in_chans, args.d_out_chans, args.condition_dim, 
    #                   args.clip_embedding_dim, curr_stage, device).to(device)
    #     for curr_stage in range(args.num_stage)
    # ]

    load_checkpoint(args, G, [None for _ in range(args.num_stage)], 
                    optim_g=None, optim_d_lst=[None for _ in range(args.num_stage)], 
                    checkpoint_path=args.checkpoint_path, epoch=args.load_epoch)


    prompt = clip.tokenize([args.prompt]).to(device)
    txt_feature = clip_model.encode_text(prompt)
    z = torch.randn(txt_feature.shape[0], args.noise_dim).to(device)
    txt_feature = normalize(txt_feature.to(device)).type(torch.float32)

    fake_images, _, _ = G(txt_feature, z)
    fake_image_64 = CLIPConfig.denormalize_image(fake_images[-3].detach().cpu()) 
    fake_image_128 = CLIPConfig.denormalize_image(fake_images[-2].detach().cpu()) 
    fake_image_256 = CLIPConfig.denormalize_image(fake_images[-1].detach().cpu()) 
    # epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
    
    mkdirs(args.result_path)
    torchvision.utils.save_image(fake_image_64, args.result_path + "result_64.png")
    torchvision.utils.save_image(fake_image_128, args.result_path + "result_128.png")
    torchvision.utils.save_image(fake_image_256, args.result_path + "result_256.png")


if __name__ == "__main__":
    main()