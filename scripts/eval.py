import torch
import torchvision
from utils.utils import *
from config.config import CLIPConfig
from networks.generator import Generator
from options.test_options import TestOptions
from dataset.dataloader import MM_CelebA, get_dataloader
from criteria.metric import (
    calculate_clip_score, calculate_fid_score,
    calculate_ignite_fid_score, calculate_inception_score
)

@torch.no_grad()
def evaluate(args, G, clip_model, device, dataloader, num_samples=50):
    try:
        G.eval()
        clip_model.eval()
        
        metrics = {
            'clip_score': 0.0,
            'fid_score': 0.0,
            'ignite_fid_score': 0.0,
            'inception_score_mean': 0.0,
            'inception_score_std': 0.0,
            'samples_processed': 0
        }
        
        for i, (real_imgs, _, txt_embedding) in enumerate(dataloader):
            if i >= num_samples: break
                
            batch_size = txt_embedding.size(0)
            txt_embedding = normalize(txt_embedding.to(device))
            
            try:
                # Generate images from text embeddings
                z = torch.randn(batch_size, args.noise_dim).to(device)
                fake_images, _, _ = G(txt_embedding, z)
                
                # Save generated images at different resolutions
                # Example: batch_size=4, saving 4 images at each resolution
                for res_idx, size in enumerate(['64', '128', '256']):
                    fake_image = CLIPConfig.denormalize_image(
                        fake_images[-(3-res_idx)].detach().cpu()
                    )
                    save_path = os.path.join(
                        args.result_path, 
                        f"batch_{i}_size_{size}.png"
                    )
                    torchvision.utils.save_image(fake_image, save_path)
                
                # Calculate CLIP score
                # Input shapes:
                # fake_images[-1]: [batch_size, 3, 256, 256]
                # txt_embedding: [batch_size, 512]
                clip_score = calculate_clip_score(
                    fake_images[-1],
                    txt_embedding,
                    clip_model
                )
                metrics['clip_score'] += clip_score * batch_size
                
                # Calculate FID score (torchmetrics)
                fid_score = calculate_fid_score(
                    real_imgs[-1].to(device),
                    fake_images[-1],
                    device
                )
                metrics['fid_score'] += fid_score * batch_size

                # Calculate Ignite FID score
                ignite_fid = calculate_ignite_fid_score(
                    real_imgs[-1].to(device),
                    fake_images[-1],
                    device
                )
                metrics['ignite_fid_score'] += ignite_fid * batch_size

                # Calculate Inception Score
                is_mean, is_std = calculate_inception_score(
                    fake_images[-1],
                    device
                )
                metrics['inception_score_mean'] += is_mean * batch_size
                metrics['inception_score_std'] += is_std * batch_size
                
                metrics['samples_processed'] += batch_size
                
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue
            
            if i % args.print_freq == 0:
                print(f"Processed {metrics['samples_processed']} samples")
        
        # 평균 계산
        if metrics['samples_processed'] > 0:
            metrics['clip_score'] /= metrics['samples_processed']
            metrics['fid_score'] /= metrics['samples_processed']
            metrics['ignite_fid_score'] /= metrics['samples_processed']
            metrics['inception_score_mean'] /= metrics['samples_processed']
            metrics['inception_score_std'] /= metrics['samples_processed']
        
        print("\nEvaluation Results:")
        print(f"Processed {metrics['samples_processed']} samples")
        print(f"CLIP Score: {metrics['clip_score']:.4f}")
        print(f"FID Score (torchmetrics): {metrics['fid_score']:.4f}")
        print(f"FID Score (ignite): {metrics['ignite_fid_score']:.4f}")
        print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = TestOptions().parse()
    
    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    clip_model.eval()
    
    G = Generator(
        args.g_in_chans, args.g_out_chans, args.noise_dim, 
        args.condition_dim, args.clip_embedding_dim, 
        args.num_stage, device
    ).to(device)
    
    load_checkpoint(
        args, G, [None for _ in range(args.num_stage)],
        optim_g=None, optim_d_lst=[None for _ in range(args.num_stage)],
        checkpoint_path=args.checkpoint_path,
        epoch=args.load_epoch
    )
    
    eval_dataset = MM_CelebA(args.eval_data_path, args.num_stage)
    eval_loader = get_dataloader(args=args, dataset=eval_dataset, is_train=False)
    
    # 평가 실행
    metrics = evaluate(args, G, clip_model, device, eval_loader)
    
    # 결과 저장
    # with open(os.path.join(args.result_path, 'metrics.json'), 'w') as f:
    #     json.dump(metrics, f, indent=4)
    save_metrics_to_csv(args, metrics)
    print(metrics)

if __name__ == "__main__":
    main()