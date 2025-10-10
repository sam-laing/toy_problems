import torch

# the muon optimizer with option to just orthogonalize rather than approximate
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def get_svd(G, eps=1e-7):
    """
    Compute the SVD of G using torch.linalg.svd. This is a wrapper around the function to ensure
    that we can use it with torch.compile.
    """
    #X = G.bfloat16()
    X = G.float()
    X /= (X.norm() + eps) # ensure top singular value <= 1, eps to prevent NaNs

    if len(X.shape) == 1:
        #throw error
        raise ValueError("G is a vector, cannot compute SVD")
    elif len(X.shape) == 2:
        if X.size(0) > X.size(1):
            X = X.T
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S = S.unsqueeze(0)
        return U, S, Vh
    else:
        raise ValueError("G is not a matrix, cannot compute SVD ... should be done in optimizer")


def orthogonalise(G):
    U, S, Vh = get_svd(G)

    return U @ Vh 


class Muon(torch.optim.Optimizer):
    def __init__(
            self, params, lr=1e-3, momentum=0, nesterov=False, steps=3, eps=1e-7,
            orthogonalize=False
            ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        self.steps = steps
        self.orthogonalize = orthogonalize
        self.eps = eps
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight

                if self.orthogonalize:
                    update= orthogonalise(g.reshape(len(g), -1)).view(g.shape)
                else:
                    update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
                update.mul_(max(1, g.size(-2) / g.size(-1))**0.5)
                p.data.add_(update, alpha=-lr) # take a step


from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, base_lr, warmup_ratio=0.1, min_lr=1e-6, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for _ in self.optimizer.param_groups:
            if step < self.warmup_steps:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs