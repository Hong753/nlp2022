import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from cfg.config import get_default_config
from data_utils.datasets import CUBDataset
from metrics.preparation import LoadEvalModel, prepare_moments
from metrics.fid import frechet_inception_distance
from models.dfgan_attr import Generator, Discriminator, CondEpilogue, DFGAN
from utils import prepare_folders, MetricLogger

def get_fixed_data(train_dataset, test_dataset, batch_size, device):
    pass

def prepare_data(batch, device):
    images = batch["images"]
    # attributes = torch.zeros(images.size(0), 312)
    # for sample_idx, attr in enumerate(batch["attributes"]):
    #     for attr_idx in attr:
    #         attributes[sample_idx, attr_idx] = 1
    images = images.to(device)
    attributes = batch["attributes"].float().to(device)
    return images, attributes

def run(cfg):
    # CUDA
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(cfg.TRAINING.device))
    
    # Data
    data_path = os.path.join(cfg.DATA.data_dir, cfg.DATA.dataset_name)
    dataset_kwargs = {"data_path": data_path, 
                      "num_captions": cfg.DATA.num_captions, 
                      "num_words": cfg.DATA.num_words, 
                      "img_resolution": cfg.DATA.img_size}
    train_dataset = CUBDataset(split="train", **dataset_kwargs)
    test_dataset = CUBDataset(split=cfg.METRICS.ref_dataset, **dataset_kwargs)
    
    loader_kwargs = {"batch_size": cfg.TRAINING.batch_size, 
                     "num_workers": cfg.TRAINING.num_workers, 
                     "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    # Load real moments
    eval_model = LoadEvalModel(eval_backbone=cfg.METRICS.eval_backbone, device=device)
    eval_model.eval()
    moment_path = "./runs/moments/{}_{}_prenone_{}_postfriendly_{}_moments.npz".format(
        cfg.DATA.dataset_name, cfg.DATA.img_size, cfg.METRICS.ref_dataset, cfg.METRICS.eval_backbone)
    mu_real, sigma_real = prepare_moments(moment_path=moment_path, 
                                          loader=test_loader if cfg.METRICS.ref_dataset == "test" else train_loader, 
                                          model=eval_model, 
                                          quantize=True, 
                                          device=device)
    
    # DFGAN
    z_dim = cfg.MODEL.z_dim
    c_dim = cfg.MODEL.c_dim
    img_resolution = cfg.DATA.img_size
    img_channels = cfg.DATA.img_channels
    channel_base = cfg.MODEL.channel_base
    
    generator = Generator(z_dim, c_dim, img_resolution, img_channels, channel_base, num_attributes=312)
    generator = generator.train().requires_grad_(False).to(device)
    
    discriminator = Discriminator(img_resolution, img_channels, channel_base)
    discriminator = discriminator.train().requires_grad_(False).to(device)
    
    cond_epilogue = CondEpilogue(312, channel_base)
    cond_epilogue = cond_epilogue.train().requires_grad_(False).to(device)
    
    model = DFGAN(G=generator, D=discriminator, C=cond_epilogue)
    
    # Optimizers
    G_opt = torch.optim.Adam(generator.parameters(), lr=cfg.TRAINING.G_lr, betas=(0.0, 0.9))
    D_params = list(discriminator.parameters()) + list(cond_epilogue.parameters())
    D_opt = torch.optim.Adam(D_params, lr=cfg.TRAINING.D_lr, betas=(0.0, 0.9))
    
    # Train
    fid_score = None
    best_fid_score = np.infty
    best_checkpoint = None
    
    num_epochs = cfg.TRAINING.num_epochs
    logger = MetricLogger(["g_loss", "d_loss_gen", "d_loss_real", "d_loss_mis", "d_loss_gp"])
    for epoch in range(num_epochs):
        
        # Train
        for batch in train_loader:
            # Sample batch
            real_images, c = prepare_data(batch, device)
            batch_size = real_images.size(0)
            
            # Update discriminator and conditional epilogue
            model.G.eval().requires_grad_(False)
            model.D.train().requires_grad_(True)
            model.C.train().requires_grad_(True)
            D_opt.zero_grad()
            D_loss_gen, D_loss_real, D_loss_mis = model(real_images, c, mode="D_main")
            (D_loss_real + (D_loss_gen + D_loss_mis) / 2).backward()
            D_opt.step()
            logger.d_loss_gen.update(D_loss_gen.item(), batch_size)
            logger.d_loss_real.update(D_loss_real.item(), batch_size)
            logger.d_loss_mis.update(D_loss_mis.item(), batch_size)
            
            # Regularize discriminator
            model.G.eval().requires_grad_(False)
            model.D.train().requires_grad_(True)
            model.C.train().requires_grad_(True)
            D_opt.zero_grad()
            D_loss_gp = model(real_images, c, mode="D_reg")
            D_loss_gp.backward()
            D_opt.step()
            logger.d_loss_gp.update(D_loss_gp.item(), batch_size)
            
            # Update generator
            model.G.train().requires_grad_(True)
            model.D.eval().requires_grad_(False)
            model.C.eval().requires_grad_(False)
            G_opt.zero_grad()
            G_loss = model(real_images, c.detach(), mode="G_main")
            G_loss.backward()
            G_opt.step()
            logger.g_loss.update(G_loss.item(), batch_size)
        
        # Save images
        if (epoch + 1) % 1 == 0:
            model.G.eval().requires_grad_(False)
            with torch.no_grad():
                gen_z = model._sample_inputs(batch_size=cfg.TRAINING.batch_size, device=device)
                gen_images = model.G(gen_z, c).cpu()[:64]
                gen_images = ((gen_images + 1) / 2).clamp(0.0, 1.0)
                save_path = "./runs/images/{}_dfgan_attr/gen_images_{}.png".format(cfg.DATA.dataset_name, epoch + 1)
                save_image(gen_images, save_path, padding=0, nrow=8)
        
        # Evaluation (FID on test data)
        if (epoch + 1) % 10 == 0:
            num_generate = len(test_loader.dataset)
            fake_activations = []
            with torch.no_grad():
                for batch in tqdm(test_loader, disable=True):
                    _, c = prepare_data(batch, device)
                    gen_z = model._sample_inputs(batch_size=c.size(0), device=device)
                    gen_images = model.G(gen_z, c)
                    features, _ = eval_model.get_outputs(gen_images, quantize=True)
                    fake_activations.append(features)
            fake_activations = torch.cat(fake_activations, dim=0).detach().cpu().numpy()[:num_generate].astype(np.float64)
            mu_fake = np.mean(fake_activations, axis=0)
            sigma_fake = np.cov(fake_activations, rowvar=False)
            fid_score = frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {"epoch": epoch, 
                          "fid": fid_score, 
                          "best_fid": best_fid_score, 
                          "G_state_dict": model.G.state_dict(), 
                          "D_state_dict": model.D.state_dict(), 
                          "C_state_dict": model.C.state_dict()}
            save_path = "./runs/checkpoints/{}_dfgan_attr/current_weights.pth".format(cfg.DATA.dataset_name)
            torch.save(checkpoint, save_path)
            if fid_score < best_fid_score:
                best_fid_score = fid_score
                best_checkpoint = checkpoint
                save_path = "./runs/checkpoints/{}_dfgan_attr/best_weights.pth".format(cfg.DATA.dataset_name)
                torch.save(best_checkpoint, save_path)
        
        # Log
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}/{}]".format(epoch + 1, num_epochs), end="\t")
            print("fid: {:.2f}".format(fid_score), end="\t")
            print("best fid: {:.2f}".format(best_fid_score))
            logger.print_progress()
            logger.reset()

if __name__ == "__main__":
    cfg = get_default_config()
    cfg.merge_from_file("./cfg/cub_dfgan.yaml")
    
    prepare_folders()
    run(cfg)