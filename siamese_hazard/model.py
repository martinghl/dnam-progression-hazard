import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=128, dropout=0.1):
        super().__init__()
        layers = []
        dims = (in_dim,) + tuple(hidden)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SiameseHazardNet(nn.Module):
    """Siamese tower + 3 hazard heads + projection head (triplet)."""

    def __init__(self, in_dim, emb_dim=128, dropout=0.1):
        super().__init__()
        self.tower = MLP(in_dim, hidden=(256, 128), out_dim=emb_dim, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        def head():
            dims = (emb_dim * 2 + emb_dim + 1, 128, 64, 1)
            layers = []
            for i in range(len(dims) - 2):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
            layers += [nn.Linear(dims[-2], dims[-1])]
            return nn.Sequential(*layers)

        self.h2_head = head()
        self.h3_head = head()
        self.h5_head = head()

    def encode(self, x):
        return self.tower(x)

    def project(self, h):
        e = self.proj(h)
        return F.normalize(e, p=2, dim=-1)

    def hazard_input(self, h0, h1, dt):
        d = torch.abs(h1 - h0)
        inp = torch.cat([h0, h1, d, dt.view(-1, 1)], dim=-1)
        return inp, d

    def forward_h2(self, inp):
        return self.h2_head(inp).squeeze(-1)

    def forward(self, x0, x1, dt):
        h0 = self.encode(x0)
        h1 = self.encode(x1)
        inp, d = self.hazard_input(h0, h1, dt)
        l2 = self.forward_h2(inp)
        l3 = self.h3_head(inp).squeeze(-1)
        l5 = self.h5_head(inp).squeeze(-1)
        e0 = self.project(h0)
        e1 = self.project(h1)
        return (l2, l3, l5), (h0, h1, d), (e0, e1)


class ContextProjector(nn.Module):
    """Context-aware projector with optional residual free weights."""

    def __init__(
        self,
        context_features: torch.Tensor,
        proj_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        projector_type: str = "residual",
        residual_alpha_init: float = 0.05,
        free_weight_init_std: float = 0.01,
    ):
        super().__init__()
        if context_features.dim() != 2:
            raise ValueError("context_features must be 2D: (n_cpg, ctx_dim)")
        if projector_type not in {"residual", "context_only"}:
            raise ValueError("projector_type must be one of: residual, context_only")

        self.register_buffer("E", context_features.float())
        self.projector_type = projector_type
        ctx_dim = context_features.shape[1]
        n_cpg = context_features.shape[0]

        self.g = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.alpha_resid = nn.Parameter(torch.tensor(float(residual_alpha_init)))
        self.W_free = nn.Parameter(torch.empty(n_cpg, proj_dim))
        nn.init.normal_(self.W_free, mean=0.0, std=float(free_weight_init_std))

    def V(self):
        base = self.g(self.E)
        if self.projector_type == "context_only":
            return base
        return base + self.alpha_resid * self.W_free

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.V()


class ContextSiameseHazardNet(nn.Module):
    """Full v8 model: context projection + Siamese hazard network."""

    def __init__(
        self,
        context_features: torch.Tensor,
        proj_dim: int = 64,
        ctx_hidden: int = 32,
        emb_dim: int = 128,
        dropout: float = 0.1,
        ctx_dropout: float = 0.0,
        projector_type: str = "residual",
        residual_alpha_init: float = 0.05,
        free_weight_init_std: float = 0.01,
    ):
        super().__init__()
        self.ctx = ContextProjector(
            context_features,
            proj_dim=proj_dim,
            hidden_dim=ctx_hidden,
            dropout=ctx_dropout,
            projector_type=projector_type,
            residual_alpha_init=residual_alpha_init,
            free_weight_init_std=free_weight_init_std,
        )
        self.core = SiameseHazardNet(in_dim=proj_dim, emb_dim=emb_dim, dropout=dropout)

    def project_pair(self, x0, x1):
        V = self.ctx.V().to(dtype=x0.dtype)
        z0 = torch.matmul(x0, V)
        z1 = torch.matmul(x1, V)
        return z0, z1

    def forward(self, x0, x1, dt):
        z0, z1 = self.project_pair(x0, x1)
        return self.core(z0, z1, dt)

    def forward_h2(self, x0, x1, dt):
        z0, z1 = self.project_pair(x0, x1)
        h0 = self.core.encode(z0)
        h1 = self.core.encode(z1)
        inp, _ = self.core.hazard_input(h0, h1, dt)
        return self.core.forward_h2(inp)

    def get_V(self):
        self.ctx.eval()
        with torch.no_grad():
            V = self.ctx.V()
        return V
