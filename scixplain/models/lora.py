import torch
import torch.nn as nn
from typing import Iterable, Tuple
import math


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = (self.alpha / max(1, self.r)) if self.r > 0 else 0.0
        self.do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        if self.r > 0:
            self.A = nn.Linear(in_f, self.r, bias=False)
            self.B = nn.Linear(self.r, out_f, bias=False)
            # init A small, B zero as in LoRA
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)) if hasattr(nn, 'init') else None
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None
            self.B = None
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and self.A is not None and self.B is not None:
            x_lora = x.to(self.A.weight.dtype)
            lora_out = self.B(self.do(self.A(x_lora)))
            y = y + self.scale * lora_out.to(y.dtype)
        return y

    # Expose base weight/bias for modules that expect .weight/.bias (e.g., MultiheadAttention fast path)
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, 'bias', None)


def _iter_named_linears(module: nn.Module):
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            yield name, sub


def apply_lora(modules: Iterable[Tuple[str, nn.Module]], r: int = 8, alpha: float = 16.0, dropout: float = 0.0, freeze_base: bool = True, name_filter: 'Iterable[str] | None' = None):
    """
    Replace nn.Linear in given modules with LoRALinear wrapper.
    modules: iterable of (prefix_name, module)
    name_filter: substrings; only layers whose qualified name contains any of these will be adapted; if None -> all linears.
    Returns list of adapted modules for collecting params.
    """
    adapted: list[nn.Module] = []
    filters = list(name_filter) if name_filter is not None else None
    for prefix, mod in modules:
        parent_dict = {}
        for n, m in mod.named_children():
            parent_dict[prefix + ('.' if prefix else '') + n] = (mod, n, m)
        # walk named_modules to find linears
        for qname, lin in _iter_named_linears(mod):
            fq = prefix + ('.' if prefix else '') + qname if prefix else qname
            if filters is not None and not any(f in fq for f in filters):
                continue
            # Replace on its parent
            # Find parent by splitting qname
            if qname == '':
                # root-level module is Linear; cannot assign without its parent reference
                # skip here (caller should wrap explicitly)
                continue
            parts = qname.split('.')
            parent = mod
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
            # Skip MultiheadAttention.out_proj etc., to avoid weight attribute assumptions
            import torch.nn as nn
            if isinstance(parent, nn.MultiheadAttention):
                continue
            setattr(parent, leaf, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base))
            adapted.append(getattr(parent, leaf))
    return adapted
