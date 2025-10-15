from optim.muon import Muon, WarmupCosineScheduler
from models.procrutes import generate_procrustes_data, ProcrustesModel 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def project_to_orthogonal(R):
    U, _, Vt = torch.linalg.svd(R)
    return U @ Vt


def train_optimizer(
    model,
    A,
    B,
    R_star,
    optimizer_name,
    lr,
    steps=1000,
    batch_size=None,
    betas=(0.95, 0.95),
    momentum=0.95,
    use_wu_cosine=False
):
    n = A.shape[0]
    idx = torch.arange(n, device=device)
    loss_star = torch.norm(A @ R_star - B, 'fro') ** 2

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([model.R], lr=lr, betas=betas)
    elif optimizer_name == "muon":
        optimizer = Muon([model.R], lr=lr, momentum=momentum, nesterov=False, steps=3, orthogonalize=False)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if use_wu_cosine:
        scheduler = WarmupCosineScheduler(optimizer, total_steps=steps, base_lr=lr, warmup_ratio=0.1)

    losses = []
    dist_to_opt = []

    for _ in range(steps):
        optimizer.zero_grad()
        if batch_size and batch_size < n:
            batch = idx[torch.randperm(n, device=device)[:batch_size]]
            loss = model(A[batch], B[batch])
        else:
            loss = model(A, B)

        loss.backward()
        optimizer.step()
        if use_wu_cosine:
            scheduler.step()

        with torch.no_grad():
            model.R.copy_(project_to_orthogonal(model.R))
            current_loss = torch.norm(A @ model.R - B, 'fro') ** 2
            losses.append(current_loss.item())
            dist_to_opt.append((current_loss - loss_star).item())

    return losses, dist_to_opt


def run_experiment(
    seed=0,
    n=100,
    d=10,
    noise=0.0,
    steps=1000,
    lrs=(1e-3, 3e-3, 1e-2, 3e-2),
    batch_size=None,
    betas=(0.95, 0.95),
    momentum=0.95,
    use_wu_cosine=False
):
    A, B, R_true = generate_procrustes_data(n, d, noise=noise, device=device, seed=seed)
    U, _, Vt = torch.linalg.svd(A.T @ B)
    R_star = U @ Vt

    results = {}
    for optimizer_name in ["adam", "muon"]:
        results[optimizer_name] = {}
        for lr in lrs:
            model = ProcrustesModel(d, device=device)
            _, dist_to_opt = train_optimizer(
                model, A, B, R_star, optimizer_name, lr,
                steps=steps, batch_size=batch_size, betas=betas,
                momentum=momentum, use_wu_cosine=use_wu_cosine
            )
            results[optimizer_name][lr] = dist_to_opt
    return results


def plot_results(results, title="Orthogonal Procrustes: Loss Difference to Optimal", plot_dir=None):
    plt.figure(figsize=(10, 6))
    for opt_name, lr_results in results.items():
        palette = "Blues" if opt_name == "adam" else "Reds"
        colors = sns.color_palette(palette, n_colors=len(lr_results))
        for (lr, vals), c in zip(lr_results.items(), colors):
            plt.plot(vals, label=f"{opt_name} lr={lr}", color=c, lw=2)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss difference to optimal")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(plot_dir) if f.startswith("results")]
        plt_number = len(existing_files) + 1
        plt.savefig(os.path.join(plot_dir, f"results_{plt_number}.png"))

results = run_experiment(
    noise=0.0, steps=6000, use_wu_cosine=False, lrs=(1e-3, 3e-3, 1e-2, 3e-2)
    )
print("done")

plot_results(results, title="Orthogonal Procrustes: full batch and no scheduler", plot_dir="/home/slaing/toy_problems/plots/")


