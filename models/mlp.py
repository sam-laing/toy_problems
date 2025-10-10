import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', use_bias=True):
        """
        Very simple MLP: just stitching together two linear layers with an activation in between.

        interested in the case where we do not use biases, and instead absorb them into the weights by
        augmenting the input with a constant 1. This is mathematically equivalent, but interesting to see
        if Muon can handle this vs separate optimizer for biases.
        """
        super(MLP, self).__init__()
        self.use_bias = use_bias
        
        if use_bias:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim + 1, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim + 1, output_dim, bias=False)
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

    def forward(self, x):
        if self.use_bias:
            return self.fc2(self.activation(self.fc1(x)))
        else:
            batch_size = x.shape[0] if x.dim() > 1 else 1
            ones = torch.ones(batch_size, 1, device=x.device)
            
            x_aug = torch.cat([x, torch.ones(1, device=x.device)]) if x.dim() == 1 else torch.cat([x, ones], dim=1)
            x = self.activation(self.fc1(x_aug))
            
            x_aug = torch.cat([x, torch.ones(1, device=x.device)]) if x.dim() == 1 else torch.cat([x, ones], dim=1)
            return self.fc2(x_aug)
        

if __name__ == "__main__":

    # just want to see how the parameters look in the case of biases vs not

    model_with_bias = MLP(input_dim=10, hidden_dim=20, output_dim=5, activation='relu', use_bias=True)
    model_without_bias = MLP(input_dim=10, hidden_dim=20, output_dim=5, activation='relu', use_bias=False)

    for name, param in model_with_bias.named_parameters():
        print(f"With bias - {name}: {param.shape}")

    for name, param in model_without_bias.named_parameters():
        print(f"Without bias - {name}: {param.shape}")

    num_params_with_bias = sum(p.numel() for p in model_with_bias.parameters())
    num_params_without_bias = sum(p.numel() for p in model_without_bias.parameters())
    print(f"Total parameters with bias: {num_params_with_bias}")
    print(f"Total parameters without bias: {num_params_without_bias}")