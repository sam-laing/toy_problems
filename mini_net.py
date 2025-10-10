from data.mnist import get_mnist_dataloaders
from models.mlp import MLP  
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

from optim.muon import Muon
import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    batch_size: int = 256
    val_ratio: float = 0.1

    lr: float = 0.001
    muon_lr: float = 0.001
    hidden_dim: int = 256
    beta1: float = 0.95
    beta2: float = 0.95
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_epochs: int = 10
    muon_enabled: bool = True
    ns_steps: int = 5
    seperate_biases: bool = True
    use_wu_cosine: bool = False

def plot_losses(train_losses, val_losses, batches_per_epoch, title):
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
    model = MLP(input_dim=28*28, hidden_dim=cfg.hidden_dim, output_dim=10).to(device)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    criterion = nn.CrossEntropyLoss()


    print("making optimizers")
    if cfg.muon_enabled: 
        # select all parameters except biases for muon optimizer
        muon_params = [p for n, p in model.named_parameters() if 'bias' not in n] if cfg.seperate_biases else model.parameters()
        adamw_params = [p for n, p in model.named_parameters() if 'bias' in n] if cfg.seperate_biases else []
        muon_optimizer = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        if adamw_params != []:
            adamw_optimizer = optim.AdamW(adamw_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        else:
            adamw_optimizer = None

    else:
        adamw_optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    train_losses = []
    val_losses = []

    #will track and plot every batch loss rather than epoch loss

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
    return test_loss, test_accuracy

if __name__ == "__main__":
    cfg = TrainingConfig(muon_enabled=False)
    train(cfg)