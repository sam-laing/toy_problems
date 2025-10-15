import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', seperate_bias=True):
        super(MLP, self).__init__()
        self.seperate_biases = seperate_bias

        if seperate_bias:

            
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        else:
            # fc1: (input_dim + 1, hidden_dim)
            W1 = torch.empty(input_dim, hidden_dim)
            nn.init.kaiming_uniform_(W1, a=0, mode='fan_in', nonlinearity=activation)
            b1 = torch.zeros(1, hidden_dim)
            self.fc1 = nn.Parameter(torch.cat([W1, b1], dim=0))

            # fc2: (hidden_dim + 1, output_dim)
            W2 = torch.empty(hidden_dim, output_dim)
            nn.init.kaiming_uniform_(W2, a=0, mode='fan_in', nonlinearity=activation)
            b2 = torch.zeros(1, output_dim)
            self.fc2 = nn.Parameter(torch.cat([W2, b2], dim=0))

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

    def forward(self, x):
        if self.seperate_biases:
            return self.fc2(self.activation(self.fc1(x)))
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)  # [1, input_dim]

            batch_size = x.shape[0]

            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            h = self.activation(x_aug @ self.fc1)

            h_aug = torch.cat([h, torch.ones(batch_size, 1, device=x.device)], dim=1)
            out = h_aug @ self.fc2

            if out.shape[0] == 1:
                return out.squeeze(0)
            return out

if __name__ == "__main__":

    # just want to see how the parameters look in the case of biases vs not

    model_with_bias = MLP(input_dim=10, hidden_dim=20, output_dim=5, activation='relu', seperate_bias=False)
    model_without_bias = MLP(input_dim=10, hidden_dim=20, output_dim=5, activation='relu', seperate_bias=True)

    for name, param in model_with_bias.named_parameters():
        print(f"With bias - {name}: {param.shape}")

    for name, param in model_without_bias.named_parameters():
        print(f"Without bias - {name}: {param.shape}")

    num_params_with_bias = sum(p.numel() for p in model_with_bias.parameters())
    num_params_without_bias = sum(p.numel() for p in model_without_bias.parameters())
    print(f"Total parameters with bias: {num_params_with_bias}")
    print(f"Total parameters without bias: {num_params_without_bias}")