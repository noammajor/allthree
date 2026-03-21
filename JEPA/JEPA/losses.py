import torch
import torch.nn.functional as F

    
def _calculate_vicreg_loss(self, x: torch.Tensor):
    std = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    batch_size = x.shape[0]
    num_features = x.shape[-1]
    x_flat = x.reshape(-1, num_features) 
    x_centered = x_flat - x_flat.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features
    
    return var_loss, cov_loss