"""
Muon optimizer (Keller Jordan, 2024).

Nesterov-momentum + Newton-Schulz orthogonalization of the update. Intended
only for 2D hidden matrices; 1D parameters (LayerNorm gains, biases, learnable
tokens) and embedding-like matrices should go to AdamW instead.

Reference: https://kellerjordan.github.io/posts/muon/
"""
import torch


def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration. Orthogonalizes a 2D matrix G.

    Operates in bfloat16 for throughput; the quintic coefficients are tuned
    so the iteration converges quickly from a unit-Frobenius-norm start.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: orthogonalized momentum SGD for 2D hidden matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, (
                    f"Muon expects 2D params; got shape {tuple(p.shape)}")
                g = p.grad
                state = self.state[p]
                if "mom" not in state:
                    state["mom"] = torch.zeros_like(g)
                buf = state["mom"]
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if nesterov else buf
                u = _zeropower_via_newtonschulz5(u, steps=ns_steps)
                # Shape-adaptive scaling: wider matrices get proportionally
                # smaller steps, consistent with spectral-norm-based scaling.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    """Multiplexes step/zero_grad/state_dict over several optimizers.

    Inherits from torch.optim.Optimizer purely to satisfy LambdaLR's isinstance
    check; real state lives in the wrapped optimizers, so we bypass
    Optimizer.__init__.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.defaults = {}

    @property
    def param_groups(self):
        # Property (not static list) so LambdaLR always sees live groups.
        return [g for o in self.optimizers for g in o.param_groups]

    @param_groups.setter
    def param_groups(self, _value):
        pass

    @property
    def state(self):
        merged = {}
        for o in self.optimizers:
            merged.update(o.state)
        return merged

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self.optimizers:
            o.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)
