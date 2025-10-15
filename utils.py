import torch 
import torch.nn as nn

def get_svd(W):
    """ 
    maximally efficient (with gpu) way to get svd of a matrix
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U, S, Vh

def top_sv(W):
    """
    get the top singular value of a matrix W
    """
    if len(W.shape) != 2:
        raise ValueError("W must be a 2D matrix")
    with torch.no_grad():
        U, S, Vh = get_svd(W)
        return S[0].item()
    

def get_lipschitz_constant(model):
    """
    just an MLP so its spectral norm(W1)spectral norm(W2)...
    """
    lip = 1.0
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:
            lip *= top_sv(param)
    
    return lip


def get_empirical_lipschitz(model, x, eps=1e-6, max_pairs=None):
    """
    Calculate empirical Lipschitz constant using vectorized operations.
    
    Args:
        model: Model to evaluate
        x: Input tensor [batch_size, *feature_dims]
        eps: Small constant to prevent division by zero
        max_pairs: If set, randomly sample this many pairs (for very large batches)
    
    Returns:
        Empirical Lipschitz constant (float)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    batch_size = x.size(0)
    
    # For large batches, sample random pairs instead of computing all n^2 pairs
    if max_pairs and max_pairs < (batch_size * (batch_size - 1)) // 2:
        idx1 = torch.randint(0, batch_size, (max_pairs,), device=x.device)
        idx2 = torch.randint(0, batch_size, (max_pairs,), device=x.device)
        # Ensure idx1 != idx2
        same_indices = (idx1 == idx2)
        idx2[same_indices] = (idx2[same_indices] + 1) % batch_size
        
        # Compute differences
        x_diffs = x[idx1] - x[idx2]
        out_diffs = outputs[idx1] - outputs[idx2]
    else:
        # Vectorized computation of all unique pairwise differences
        idx1, idx2 = torch.triu_indices(batch_size, batch_size, 1, device=x.device)
        
        x_diffs = x[idx1] - x[idx2]
        out_diffs = outputs[idx1] - outputs[idx2]
    
    # Flatten each difference vector and compute norms
    x_norms = torch.norm(x_diffs.view(x_diffs.size(0), -1), dim=1) + eps
    out_norms = torch.norm(out_diffs.view(out_diffs.size(0), -1), dim=1)
    
    # Compute all ratios and find max
    ratios = out_norms / x_norms
    return ratios.max().item()
 

def each_matrix_lipschitz(model):
    lips = {}
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:
            lips[name] = top_sv(param)
    return lips
