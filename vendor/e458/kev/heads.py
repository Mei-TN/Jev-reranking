"""Versioned, variable-candidate heads for the isolated head ablation tree."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_head_config(config=None, *, head_dim=256):
    """Missing metadata means the original pointer/end head, including old widths.

        The paired configurations set rng_anchor_dim=256. A
        legacy pointer/end configuration without that field keeps direct init.

    """
    if config is None:
        config = {"version": 1, "kind": "pointer", "dim": head_dim, "readout": "end"}
    if type(config) is not dict:
        raise ValueError("head_config must be a JSON object")
    required = {"version", "kind", "dim", "readout"}
    if not required <= config.keys():
        raise ValueError("head_config is missing required fields")
    kind = config["kind"]
    if type(kind) is not str or kind not in {"pointer", "tanh_pointer", "cosine_pointer", "residual_pair_mlp"}:
        raise ValueError("unknown head kind")
    optional = {"rng_anchor_dim"}
    if kind == "cosine_pointer":
        optional.add("cosine_scale")
    if kind == "residual_pair_mlp":
        optional.add("mlp_hidden")
    if set(config) - required - optional:
        raise ValueError("unknown head_config fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("unsupported head_config version")
    dim, readout = config["dim"], config["readout"]
    if type(dim) is not int or dim <= 0:
        raise ValueError("head dim must be a positive integer")
    if type(readout) is not str or readout not in {"end", "mean_span", "last_span"}:
        raise ValueError("unknown head readout")
    anchored = "rng_anchor_dim" in config
    if anchored and (type(config["rng_anchor_dim"]) is not int or config["rng_anchor_dim"] != 256):
        raise ValueError("campaign rng_anchor_dim must be 256")
    if not anchored and (kind != "pointer" or readout != "end"):
        raise ValueError("new head variants require rng_anchor_dim=256")
    if anchored:
        if kind == "pointer":
            if (dim, readout) not in {(256, "end"), (64, "end"), (256, "mean_span"), (256, "last_span")}:
                raise ValueError("unsupported campaign pointer configuration")
        elif dim != 256 or readout != "end":
            raise ValueError("nonlinear campaign heads require dim=256/readout=end")
    out = {"version": 1, "kind": kind, "dim": dim, "readout": readout}
    if anchored:
        out["rng_anchor_dim"] = 256
    if kind == "cosine_pointer":
        scale = config.get("cosine_scale")
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale != 16:
            raise ValueError("cosine_scale must be fixed at 16")
        out["cosine_scale"] = 16.0
    if kind == "residual_pair_mlp":
        if type(config.get("mlp_hidden")) is not int or config["mlp_hidden"] != 128:
            raise ValueError("mlp_hidden must be 128")
        out["mlp_hidden"] = 128
    return out


class PointerHead(nn.Module):
    # Preserve the pointer-head interface and operation order.
    def __init__(self, d, dp=256, dropout=0.0):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("head dropout must be finite and in [0, 1)")
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_decide, h_opts):
        if self.dropout.p > 0:
            k, q = self.dropout(self.k(h_opts)), self.dropout(self.q(h_decide))
        else:
            k, q = self.k(h_opts), self.q(h_decide)
        logits = (k @ q) * self.scale
        observer = getattr(self, "_diagnostic_observer", None)
        if observer is not None:
            observer(q=q, k=k, logits=logits)
        return logits


class TanhPointerHead(PointerHead):
    def forward(self, h_decide, h_opts):
        k, q = self.k(h_opts).tanh(), self.q(h_decide).tanh()
        if self.dropout.p > 0:
            k, q = self.dropout(k), self.dropout(q)
        return (k @ q) * self.scale


class CosinePointerHead(PointerHead):
    def forward(self, h_decide, h_opts):
        k, q = self.k(h_opts), self.q(h_decide)
        # Norm arithmetic is FP32. Restore the projection dtype before the dot
        # product so the surrounding training autocast policy stays in force.
        with torch.autocast(device_type=k.device.type, enabled=False):
            kn = F.normalize(k.float(), dim=-1, eps=1e-12).to(k.dtype)
            qn = F.normalize(q.float(), dim=-1, eps=1e-12).to(q.dtype)
        if self.dropout.p > 0:
            kn, qn = self.dropout(kn), self.dropout(qn)
        return (kn @ qn) * 16.0


class ResidualPairMLPHead(PointerHead):
    def __init__(self, d, dp=256, dropout=0.0, mlp_hidden=128):
        super().__init__(d, dp, dropout)
        self.pair_mlp = nn.Sequential(nn.Linear(4 * dp, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, 1))
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def forward(self, h_decide, h_opts):
        k, q = self.k(h_opts), self.q(h_decide)
        if self.dropout.p > 0:
            k, q = self.dropout(k), self.dropout(q)
        baseline = (k @ q) * self.scale
        queries = q.expand_as(k)
        features = torch.cat((queries, k, queries * k, (queries - k).abs()), dim=-1)
        residual = self.pair_mlp(features).squeeze(-1)
        logits = baseline + residual.to(baseline.dtype)
        observer = getattr(self, "_diagnostic_observer", None)
        if observer is not None:
            observer(q=q, k=k, logits=logits, base_logits=baseline, residual_logits=residual)
        return logits


def make_head(hidden_size, *, head_dim=256, dropout=0.0, head_config=None):
    config = normalize_head_config(head_config, head_dim=head_dim)
    if "rng_anchor_dim" not in config:
        head = PointerHead(hidden_size, dp=config["dim"], dropout=dropout)
    else:
        # Use the same 256-dimensional initialization draw for every variant
        # from the caller's CPU RNG. Extra allocation draws cannot advance it.
        canonical = PointerHead(hidden_size, dp=256, dropout=dropout)
        if config["kind"] == "pointer" and config["dim"] == 256:
            head = canonical
        else:
            cls = {"pointer": PointerHead, "tanh_pointer": TanhPointerHead,
                   "cosine_pointer": CosinePointerHead, "residual_pair_mlp": ResidualPairMLPHead}[config["kind"]]
            with torch.random.fork_rng(devices=[]):
                kwargs = {"mlp_hidden": config["mlp_hidden"]} if config["kind"] == "residual_pair_mlp" else {}
                head = cls(hidden_size, dp=config["dim"], dropout=dropout, **kwargs)
            with torch.no_grad():
                for name in ("q", "k"):
                    target, source = getattr(head, name), getattr(canonical, name)
                    target.weight.copy_(source.weight[:config["dim"]])
                    target.bias.copy_(source.bias[:config["dim"]])
    head.head_config = config
    return head


def readout_features(h, decide_idx, opt_idx, *, readout="end", opt_spans=None):
    """Read one question from [L,d]; span bounds are half-open content-only.

    Empty content falls back to its end delimiter. The encoder, not token
    values or padding heuristics, defines the spans. last_span reads the last
    content token h[end-1] of the same spans mean_span averages; the query
    is the decide token for every readout.
    """
    if readout == "end":
        return h[decide_idx], h[torch.tensor(opt_idx, device=h.device)]
    if readout not in ("mean_span", "last_span"):
        raise ValueError("mean_span requires one explicit content span per option")
    if opt_spans is None or len(opt_spans) != len(opt_idx) or not opt_idx:
        raise ValueError(readout + " requires one explicit content span per option")
    values = []
    for end_token, span in zip(opt_idx, opt_spans):
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(v) is not int for v in span)):
            raise ValueError("invalid option content span")
        start, end = span
        if not 0 <= start <= end == end_token < h.shape[0]:
            raise ValueError("option content bounds do not end at their delimiter")
        if start == end:
            values.append(h[end_token])
        elif readout == "mean_span":
            values.append(h[start:end].mean(dim=0))
        else:
            values.append(h[end - 1])
    return h[decide_idx], torch.stack(values)
