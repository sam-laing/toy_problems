import torch 

def generate_procrustes_data(n, d, device, noise=0.0, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    U, _, Vt = torch.linalg.svd(torch.randn(d, d, generator=generator))
    R_true = U @ Vt
    A = torch.randn(n, d, generator=generator)
    B = A @ R_true
    if noise > 0:
        B += noise * torch.randn_like(B, generator=generator)
    return A.to(device), B.to(device), R_true.to(device)


class ProcrustesModel(torch.nn.Module):
    def __init__(self, d, device):
        super().__init__()
        self.R = torch.eye(d, requires_grad=True, device=device)

    def forward(self, A, B):
        return torch.norm(A @ self.R - B, 'fro')**2
