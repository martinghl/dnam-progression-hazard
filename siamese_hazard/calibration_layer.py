import torch
import torch.nn as nn
import torch.nn.functional as F

class PlattLayer(nn.Module):
    """Affine calibration layer on logits: logit' = a*logit + b."""
    def __init__(self, init_a: float = 1.0, init_b: float = 0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.a * logits + self.b

class MultiHeadPlattCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.h2 = PlattLayer()
        self.h3 = PlattLayer()
        self.h5 = PlattLayer()

    def forward(self, l2: torch.Tensor, l3: torch.Tensor, l5: torch.Tensor):
        return self.h2(l2), self.h3(l3), self.h5(l5)

def _masked_bce_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = (mask == 1) & ~torch.isnan(y)
    if m.sum() == 0:
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits[m], y[m].float(), reduction="mean")

def fit_platt_calibrator(
    calibrator: MultiHeadPlattCalibrator,
    l2: torch.Tensor, l3: torch.Tensor, l5: torch.Tensor,
    y2: torch.Tensor, y3: torch.Tensor, y5: torch.Tensor,
    m2: torch.Tensor, m3: torch.Tensor, m5: torch.Tensor,
    max_iter: int = 200,
    l2_reg: float = 1e-3,
    device: str = "cpu",
):
    calibrator = calibrator.to(device)
    calibrator.train()

    l2 = l2.to(device); l3 = l3.to(device); l5 = l5.to(device)
    y2 = y2.to(device); y3 = y3.to(device); y5 = y5.to(device)
    m2 = m2.to(device); m3 = m3.to(device); m5 = m5.to(device)

    opt = torch.optim.LBFGS(list(calibrator.parameters()), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        c2, c3, c5 = calibrator(l2, l3, l5)
        loss = _masked_bce_logits(c2, y2, m2) + _masked_bce_logits(c3, y3, m3) + _masked_bce_logits(c5, y5, m5)
        # L2 regularization to keep a≈1,b≈0 unless data supports otherwise
        reg = 0.0
        for layer in [calibrator.h2, calibrator.h3, calibrator.h5]:
            reg = reg + (layer.a - 1.0).pow(2) + (layer.b).pow(2)
        loss = loss + l2_reg * reg
        loss.backward()
        return loss

    final_loss = float(opt.step(closure).item())
    calibrator.eval()
    meta = {
        "loss": final_loss,
        "h2_a": float(calibrator.h2.a.detach().cpu().item()),
        "h2_b": float(calibrator.h2.b.detach().cpu().item()),
        "h3_a": float(calibrator.h3.a.detach().cpu().item()),
        "h3_b": float(calibrator.h3.b.detach().cpu().item()),
        "h5_a": float(calibrator.h5.a.detach().cpu().item()),
        "h5_b": float(calibrator.h5.b.detach().cpu().item()),
    }
    return meta
