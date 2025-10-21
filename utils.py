import torch 
import torch.nn as nn
import matplotlib.pyplot as plt    




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




def plot_losses(train_losses, val_losses, batches_per_epoch, title, empirical_consts=None, algebraic_consts=None, log_every=1):
    """
    Plot losses and optionally lipschitz constants side by side.
    """
    if empirical_consts is not None and algebraic_consts is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(range(0, len(train_losses), batches_per_epoch), val_losses, label='Validation Loss', color='orange')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.grid(True, alpha=0.4)
        ax1.legend()
        
        steps = range(0, len(empirical_consts) * log_every, log_every)
        ax2.plot(steps, empirical_consts, label='Empirical Lipschitz', color='green')
        ax2.plot(steps, algebraic_consts, label='Algebraic Lipschitz (Top SV)', color='purple')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Lipschitz Constant')
        ax2.set_title('Lipschitz Constants')
        ax2.grid(True, alpha=0.4)
        ax2.legend()
        
        plt.suptitle(title)
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(range(0, len(train_losses), batches_per_epoch), val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.4)
        plt.title(title)
        plt.legend()
    
    plot_dir = "/home/slaing/muon_research/toy_problems/plots/"
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{title}.pdf")
    plt.close()
