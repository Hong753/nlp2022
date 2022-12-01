import numpy as np

from scipy import linalg

def frechet_inception_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm(np.dot(sigma1 + offset, sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean