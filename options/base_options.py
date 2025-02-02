import argparse
import os
import time
import torch
from pathlib import Path
from utils.utils import *

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):                         
        seed_fix(0)
        parser.add_argument('--name', type=str, default='experiment_name', 
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', 
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_workers', type=int, default=4)

        parser.add_argument('--data_path', default='./data/sample_train.zip', type=Path, help="path of directory containing training dataset")  
        parser.add_argument('--resume_checkpoint_path', default=None)
        parser.add_argument('--resume_epoch', type=int, default=-1)
        parser.add_argument('--report_interval', type=int, default=100, help='Report interval')
        parser.add_argument('--checkpoint_path', type=Path, default='./checkpoints', help='Checkpoint path')
        parser.add_argument('--result_path', type=Path, default='./output', help='Generated image path')     
        
        parser.add_argument('--noise_dim', type=int, default=100, help= 'Input noise dimension to Generator')
        parser.add_argument('--condition_dim', type=int, default=128, help= 'Noise projection dimension')
        parser.add_argument('--clip_embedding_dim', type=int, default=512, help= 'Dimension of c_txt from CLIP ViT-B/32')
        
        parser.add_argument('--g_in_chans', type=int, default=1024,
                            help='Number of input channels for generator (Ng)')                            
        parser.add_argument('--g_out_chans', type=int, default=3,
                            help='Number of output channels for generator')                            
        parser.add_argument('--d_in_chans', type=int, default=64,
                            help='Number of input channels for discriminator (Nd)')                            
        parser.add_argument('--d_out_chans', type=int, default=1,
                            help='Number of output channels for discriminator')
        parser.add_argument('--num_stage', type=int, default=3)      
        
        parser.add_argument('--clip_model', type=str, choices=(['ViT-B/32', 'ViT-L/14', 'ViT-B/16']), default='ViT-B/32')
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
   
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt #parser.parse_args() 
    
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)       
        return message

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.name = opt.name + time.strftime("-%Y_%m_%d_%H_%M_%S", time.localtime())        

        if print_options:
            msg = self.print_options(opt)
            
            # save to the disk
            expr_dir = os.path.join(opt.checkpoint_path, opt.name)
            mkdirs(expr_dir)
                
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(msg)
                opt_file.write('\n')
            
            if str(opt.checkpoint_path) == self.parser.get_default('checkpoint_path')[2:]:
                opt.checkpoint_path = os.path.join(expr_dir, 'ckpt')
                os.makedirs(opt.checkpoint_path, exist_ok=True)
            if str(opt.result_path) == self.parser.get_default('result_path')[2:]:
                opt.result_path = os.path.join(expr_dir, 'res')
                os.makedirs(opt.result_path, exist_ok=True)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
    
    
