# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/ins.py

import math

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
import torch
import numpy as np

def inception_softmax(eval_model, images, quantize):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps


def calculate_kl_div(ps, splits):
    scores = []
    num_samples = ps.shape[0]
    with torch.no_grad():
        for j in range(splits):
            part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            kl = torch.exp(kl)
            scores.append(kl.unsqueeze(0))

        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std


def eval_features(probs, labels, num_features, split):
    probs, labels = probs[:num_features], labels[:num_features]
    m_scores, m_std = calculate_kl_div(probs, splits=split)

    return m_scores, m_std
