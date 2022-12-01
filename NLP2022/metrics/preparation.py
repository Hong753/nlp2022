import os
import numpy as np
import torch

from tqdm import tqdm
from torchvision import transforms

from metrics.inception import InceptionV3
from metrics.resizer import build_resizer

class LoadEvalModel():
    def __init__(self, eval_backbone, device):
        self.eval_backbone = eval_backbone
        self.device = device
        
        if self.eval_backbone == "InceptionV3_tf":
            self.res = 299
            mean = std = [0.5] * 3
            self.model = InceptionV3(resize_input=False, normalize_input=False).to(self.device)
        else:
            raise NotImplementedError
        
        self.resizer = build_resizer(resizer="friendly", backbone=self.eval_backbone, size=self.res)
        self.totensor = transforms.ToTensor()
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)
        
    def eval(self):
        self.model.eval()
        
    def get_outputs(self, x, quantize=False):
        # Preprocess
        if quantize:
            x = (x + 1) / 2
            x = (255.0 * x + 0.5).clamp(0.0, 255.0)
            x = x.detach().cpu().numpy().astype(np.uint8)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = x.transpose((0, 2, 3, 1))
        x = list(map(lambda x: self.totensor(self.resizer(x)), list(x)))
        x = torch.stack(x, 0).to(self.device)
        x = (x / 255.0 - self.mean) / self.std
        
        # Backbone
        if self.eval_backbone == "InceptionV3_tf":
            features, logits = self.model(x)
        else:
            raise NotImplementedError
        
        return features, logits

def prepare_moments(moment_path, loader, model, quantize, device, recompute=False, disable_tqdm=False):    
    if os.path.isfile(moment_path) and recompute is False:
        mu = np.load(moment_path)["mu"]
        sigma = np.load(moment_path)["sigma"]
        print("Moments loaded from path!")
    else:
        model.eval()
        real_activations = []
        for batch in tqdm(loader, disable=disable_tqdm):
            images = batch["images"].to(device)
            with torch.no_grad():
                features, _ = model.get_outputs(images, quantize=quantize)
                real_activations.append(features)
        real_activations = torch.cat(real_activations, dim=0).detach().cpu().numpy().astype(np.float64)
        mu = np.mean(real_activations, axis=0)
        sigma = np.cov(real_activations, rowvar=False)
        print("Computed moments and saved to disk!")
        np.savez(moment_path, **{"mu": mu, "sigma": sigma})
    return mu, sigma
