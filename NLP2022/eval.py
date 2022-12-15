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
from metrics.ins import eval_features
from models.dfgan_attr import Generator, Discriminator, CondEpilogue, DFGAN
from models.dfgan import Generator as Generator_base, Discriminator as Discriminator_base, \
    CondEpilogue as CondEpilogue_base, DFGAN as DFGAN_base
from models.damsm import RNN_ENCODER
from utils.utils import prepare_folders, MetricLogger
import wandb
import argparse

def prepare_data(batch, device, return_class=False):
    images = batch["images"]
    # attributes = torch.zeros(images.size(0), 312)
    # for sample_idx, attr in enumerate(batch["attributes"]):
    #     for attr_idx in attr:
    #         attributes[sample_idx, attr_idx] = 1
    images = images.to(device)
    attributes = batch["attributes"].float().to(device)
    if return_class:
        classes = batch["class_ids"].float().to(device)
        return images, attributes, classes
    return images, attributes

def prepare_base_data(batch, device, return_class=False):
    images = batch["images"]
    captions = batch["captions"]
    caption_lengths = batch["caption_lengths"]
    _, sorted_idx = torch.sort(caption_lengths, dim=0, descending=True)
    sorted_images = images[sorted_idx].to(device)
    sorted_captions = captions[sorted_idx].to(device)
    sorted_caption_lengths = caption_lengths[sorted_idx].to(device)
    if return_class:
        classes = batch["class_ids"].float().to(device)
        return sorted_images, sorted_captions, sorted_caption_lengths, classes
    return sorted_images, sorted_captions, sorted_caption_lengths
       
def run(cfg, model_path, mode="attr"):
    # CUDA
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(cfg.TRAINING.device))
    
    # Load Test Data
    data_path = os.path.join(cfg.DATA.data_dir, cfg.DATA.dataset_name)
    dataset_kwargs = {"data_path": data_path, 
                      "num_captions": cfg.DATA.num_captions, 
                      "num_words": cfg.DATA.num_words, 
                      "img_resolution": cfg.DATA.img_size}
    
    test_dataset = CUBDataset(split=cfg.METRICS.ref_dataset, **dataset_kwargs)
    
    loader_kwargs = {"batch_size": cfg.TRAINING.batch_size, 
                     "num_workers": cfg.TRAINING.num_workers, 
                     "pin_memory": True}
    
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    # Load real moments\
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

    if mode=="base":
        # Pre-trained DAMSM
        vocab_size = len(test_dataset.id2word)
        text_encoder = RNN_ENCODER(vocab_size, nhidden=cfg.MODEL.c_dim)
        text_encoder_state_dict = torch.load(os.path.join(data_path, "DAMSMencoder", "text_encoder200.pth"), map_location="cpu")
        text_encoder.load_state_dict(text_encoder_state_dict)
        text_encoder = text_encoder.eval().requires_grad_(False).to(device)

    if mode=="attr":
        generator = Generator(z_dim, c_dim, img_resolution, img_channels, channel_base, num_attributes=312)
    else:
        generator = Generator_base(z_dim, c_dim, img_resolution, img_channels, channel_base)
    generator = generator.train().requires_grad_(False).to(device)
    
    if mode=="attr":
        discriminator = Discriminator(img_resolution, img_channels, channel_base)
    else:
        discriminator = Discriminator_base(img_resolution, img_channels, channel_base)
    discriminator = discriminator.train().requires_grad_(False).to(device)
    
    if mode=="attr":
        cond_epilogue = CondEpilogue(312, channel_base)
    else:
        cond_epilogue = CondEpilogue_base(c_dim, channel_base)
    cond_epilogue = cond_epilogue.train().requires_grad_(False).to(device)
    
    if mode=="attr":
        model = DFGAN(G=generator, D=discriminator, C=cond_epilogue)
    else:
        model = DFGAN_base(G=generator, D=discriminator, C=cond_epilogue)
        
    state_dict = torch.load(model_path, map_location="cpu")
    
    # load to model
    model.G.load_state_dict(state_dict=state_dict["G_state_dict"])
    model.D.load_state_dict(state_dict=state_dict["D_state_dict"])
    model.C.load_state_dict(state_dict=state_dict["C_state_dict"])
    
    model.to(device)
    
    # Define FID and IS
    fid_score = None
    is_score = None
    
    # Start Eval
    model.G.eval().requires_grad_(False)
    model.D.eval().requires_grad_(False)
    model.C.eval().requires_grad_(False)
    
    num_generate = len(test_loader.dataset)
    fake_activations = []
    probs = []
    labels = []
    if mode=="attr":
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=True):
                _, c, ids = prepare_data(batch, device, return_class=True)
                gen_z = model._sample_inputs(batch_size=c.size(0), device=device)
                gen_images = model.G(gen_z, c)
                features, logits = eval_model.get_outputs(gen_images, quantize=True)
                fake_activations.append(features)
                probs.append(logits.softmax(dim=-1))
                labels.append(ids)
    else:
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=True):
                _, captions, caption_lengths, ids = prepare_base_data(batch, device, return_class=True)
                hidden = text_encoder.init_hidden(captions.size(0))
                _, c = text_encoder(captions, caption_lengths, hidden)
                gen_z = model._sample_inputs(batch_size=c.size(0), device=device)
                gen_images = model.G(gen_z, c)
                features, logits = eval_model.get_outputs(gen_images, quantize=True)
                fake_activations.append(features)        
                probs.append(logits.softmax(dim=-1))
                labels.append(ids)                
    fake_activations = torch.cat(fake_activations, dim=0).detach().cpu().numpy()[:num_generate].astype(np.float64)
    probs = torch.cat(probs, dim=0).detach()
    labels = torch.cat(labels, dim=0).detach()
    
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    fid_score = frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    kl_score, kl_std = eval_features(probs, labels, len(test_dataset), 1)
    print(fid_score, kl_score, kl_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--mode", type=str, default="attr")
    args = parser.parse_args()
    
    cfg = get_default_config()
    cfg.merge_from_file("./cfg/cub_dfgan.yaml")
    
    prepare_folders()
    run(cfg, args.model_path, mode=args.mode)