from data.mnist import get_mnist_dataloaders
from models.mlp import MLP  
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from optim.muon import Muon
import matplotlib.pyplot as plt
from utils import get_lipschitz_constant, top_sv

@dataclass
class TrainingConfig:
    batch_size: int = 512
    val_ratio: float = 0.1
    lr: float = 0.005
    hidden_dim: int = 2048
    beta1: float = 0.95
    beta2: float = 0.95
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_epochs: int = 10
    muon_enabled: bool = True
    ns_steps: int = 5
    orthogonalize: bool = False

    seperate_biases: bool = True
    use_wu_cosine: bool = False

def plot_losses(train_losses, val_losses, batches_per_epoch, title, empirical_consts=None, algebraic_consts=None):
    
    # the val losses are only every epoch and train losses are every batch
    # make val plot with skips
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(range(0, len(train_losses), batches_per_epoch), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plot_dir = "/home/slaing/muon_research/toy_problems/plots/"
    plt.savefig(f"{plot_dir}/{title}.pdf")





def train(cfg: TrainingConfig):
    print("making loaders")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=cfg.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=28*28, hidden_dim=cfg.hidden_dim, output_dim=10, seperate_bias=cfg.seperate_biases).to(device)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")


    criterion = nn.CrossEntropyLoss()


    print("making optimizers")
    if cfg.muon_enabled: 
        # select all parameters except biases for muon optimizer
        muon_params = [p for n, p in model.named_parameters() if 'bias' not in n] if cfg.seperate_biases else model.parameters()
        adamw_params = [p for n, p in model.named_parameters() if 'bias' in n] if cfg.seperate_biases else []
        #  For each update matrix, we can scale its learning rate by a factor of 0.2 · pmax(A, B) based on its shape.
        muon_optimizer = Muon(
            muon_params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, 
            steps=cfg.ns_steps, orthogonalize=cfg.orthogonalize)
        if adamw_params != []:
            adamw_optimizer = optim.AdamW(adamw_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        else:
            adamw_optimizer = None

    else:
        adamw_optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    train_losses = []
    val_losses = []

    print("starting training")
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            x,y = data.to(device), target.to(device)
            if adamw_optimizer is not None:
                adamw_optimizer.zero_grad()
            if cfg.muon_enabled:
                muon_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            epoch_loss += loss.item()
            loss.backward()
            if adamw_optimizer is not None:
                adamw_optimizer.step()
            if cfg.muon_enabled:
                muon_optimizer.step()

            train_losses.append(loss.item())

        val_epoch_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            model.eval()
            x,y = data.to(device), target.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_epoch_loss += loss.item()
        val_losses.append(val_epoch_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_losses[-1]:.4f}")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            x,y = data.to(device), target.to(device)
            output = model(x)
            test_loss += criterion(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


    batches_per_epoch = len(train_loader)
    title = f"MNIST_MLP"
    title += f"_muon" if cfg.muon_enabled else "_adamw"
    title += f"_hd={cfg.hidden_dim}"
    title += f"_lr={cfg.lr}_ns={cfg.ns_steps}_mom={cfg.momentum}_wd={cfg.weight_decay}_bs={cfg.batch_size}"
    if cfg.seperate_biases:
        title += f"_seperate_biases"
    if cfg.use_wu_cosine:
        title += f"_wu_cosine"
    
    #avoid overwriting plots
    import time
    title += f"_{int(time.time())}"

    plot_losses(train_losses, val_losses, batches_per_epoch, title)

    return train_losses, val_losses, test_loss, test_accuracy
    

"""   

batches_per_epoch = len(train_loader)
title = f"MNIST_MLP"
title += f"_muon" if cfg.muon_enabled else "_adamw"
title += f"_hd={cfg.hidden_dim}"
title += f"_lr={cfg.muon_lr}_ns={cfg.ns_steps}_mom={cfg.momentum}_wd={cfg.weight_decay}"
if cfg.seperate_biases:
    title += f"_seperate_biases"
if cfg.use_wu_cosine:
    title += f"_wu_cosine"

plot_losses(train_losses, val_losses, batches_per_epoch, title)
#test set
"""

if __name__ == "__main__":
    """  
    #num_seeds = 1
    results_dict = {}
    #basically store results for each config in a json type file
    #have keys as representative name based on config and values as list of (train_losses, val_losses, test_loss, test_accuracy) for each seed
    to_vary = {
        "seperate_biases": [True],
        "lr": [0.0005, 0.001, 0.003, 0.01],
        "batch_size": [128, 256, 512, 1024],
        }
    
    for seperate_bias in to_vary["seperate_biases"]:
        for lr in to_vary["lr"]:
            for batch_size in to_vary["batch_size"]:
                cfg = TrainingConfig(muon_enabled=True, seperate_biases=seperate_bias, num_epochs=10, lr=lr, batch_size=batch_size)
                print(f"Training with config: {cfg}")
                train_losses, val_losses, test_loss, test_accuracy = train(cfg)
                key = f"muon_sep={seperate_bias}_lr={lr}_bs={batch_size}"
                results_dict[key] = (train_losses, val_losses, test_loss, test_accuracy)
    import json
    import os
    results_dir = "/home/slaing/muon_research/toy_problems/results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    import time
    filename = f"{results_dir}/mnist_mlp_batch_lr_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(results_dict, f)


    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=28*28, hidden_dim=2048, output_dim=10, seperate_bias=True).to(device)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=512)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.95, 0.95), weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    #basically do a few training steps, compute the actual lipschitz constant and approximate lipschitz with the batch
    # literally by max ||f(x1) - f(x2)|| / ||x1-x2|| over the batch (each pair of points or representative if too large)

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

    empirical_consts = []
    algebraic_consts = []

    model.train()
    for epoch in range(15):

    
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                L_algebraic = get_lipschitz_constant(model)
                algebraic_consts.append(L_algebraic)
                L_empirical = get_empirical_lipschitz(model, x, max_pairs=10000)
                empirical_consts.append(L_empirical)
    # plot the two constants over time
    import matplotlib.pyplot as plt
    plt.plot(empirical_consts, label='Empirical Lipschitz Constant')
    plt.plot(algebraic_consts, label='Algebraic Lipschitz Constant (Top SV)')
    plt.xlabel('Training Steps')
    plt.ylabel('Lipschitz Constant')
    plt.legend()
    plt.title('Lipschitz Constant During Training')
    plt.savefig('/home/slaing/muon_research/toy_problems/plots/lipschitz_constants.png')

    #test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            x,y = data.to(device), target.to(device)
            output = model(x)
            test_loss += criterion(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
 




        


