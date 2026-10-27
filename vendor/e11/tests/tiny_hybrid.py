"""A tiny stand-in for the hybrid backbone, so CPU tests run the real DecisionModel, the real PEFT LoRA and kev.train.

Module names end in the real target leaves (self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_*,out_proj},
mlp.{gate,up,down}_proj), so PEFT injects, initializes, scales and saves the adapters as on the real model. Every
position reads only itself and earlier allowed positions, in the packed (4-D additive mask) and the row (2-D
mask) form alike. Only AutoModelForCausalLM.from_pretrained and the tokenizer are replaced; no download, no GPU.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import types
from unittest import mock

import torch

from kev import model as model_module

HIDDEN, INTERMEDIATE, VOCAB, POSITIONS = 16, 24, 64, 512
LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
LEAF_SIZES = {
    "full_attention": ("self_attn", {"q_proj": (HIDDEN, HIDDEN), "k_proj": (HIDDEN, HIDDEN // 2),
                                     "v_proj": (HIDDEN, HIDDEN // 2), "o_proj": (HIDDEN, HIDDEN)}),
    "linear_attention": ("linear_attn", {"in_proj_qkv": (HIDDEN, 2 * HIDDEN), "in_proj_z": (HIDDEN, HIDDEN),
                                         "in_proj_a": (HIDDEN, 4), "in_proj_b": (HIDDEN, 4),
                                         "out_proj": (HIDDEN, HIDDEN)})}
MLP_SIZES = {"gate_proj": (HIDDEN, INTERMEDIATE), "up_proj": (HIDDEN, INTERMEDIATE), "down_proj": (INTERMEDIATE, HIDDEN)}
SPECIAL_IDS = {token: i + 1 for i, token in enumerate(model_module.SPECIAL)}


def lora_modules(layer_types=LAYER_TYPES):
    """The base model's LoRA-targetable modules as the plan's shape metadata lists them."""
    rows = []
    for kind in layer_types:
        for leaf, (i, o) in {**LEAF_SIZES[kind][1], **MLP_SIZES}.items():
            rows.append({"leaf": leaf, "in_features": i, "out_features": o})
    return rows


class TinyConfig:
    def __init__(self, layer_types):
        self.hidden_size, self.layer_types, self.use_cache = HIDDEN, list(layer_types), True
        self.model_type, self.vocab_size, self.tie_word_embeddings = "tiny_hybrid", VOCAB, False

    def to_dict(self):
        return dict(vars(self))

    def get(self, key, default=None):
        return getattr(self, key, default)


class Leaves(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        for name, (i, o) in sizes.items():
            setattr(self, name, torch.nn.Linear(i, o))


class TinyLayer(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        name, sizes = LEAF_SIZES[kind]
        setattr(self, name, Leaves(sizes))
        self.mlp = Leaves(MLP_SIZES)

    def forward(self, h, allow):
        def earlier(x):                     # mean over the positions this one may read
            return (allow @ x) / allow.sum(-1, keepdim=True)
        if self.kind == "full_attention":
            a = self.self_attn
            h = h + a.o_proj(torch.tanh(a.q_proj(h)) * earlier(torch.cat((a.k_proj(h), a.v_proj(h)), -1)))
        else:
            a = self.linear_attn
            q, v = a.in_proj_qkv(h).split(HIDDEN, dim=-1)
            decay = torch.sigmoid(a.in_proj_a(h)).mean(-1, keepdim=True)
            beta = torch.sigmoid(a.in_proj_b(h)).mean(-1, keepdim=True)
            h = h + a.out_proj(torch.tanh(q) * earlier(v * beta) * decay + torch.tanh(a.in_proj_z(h)))
        return h + self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))


class TinyHybridBackbone(torch.nn.Module):
    def __init__(self, layer_types=LAYER_TYPES):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.embed_positions = torch.nn.Embedding(POSITIONS, HIDDEN)
        self.layers = torch.nn.ModuleList(TinyLayer(kind) for kind in layer_types)
        self.config = TinyConfig(layer_types)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **unused):
        h = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        length = input_ids.shape[1]
        if attention_mask.dim() == 4:       # packed: additive [B,1,L,L], zero where reading is allowed
            allow = attention_mask[:, 0] == 0
        else:                               # rows: [B,L] valid keys, causal
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=h.device))
            allow = causal[None] & attention_mask[:, None, :].bool() | torch.eye(length, dtype=torch.bool, device=h.device)[None]
        allow = allow.to(h.dtype)
        for layer in self.layers:
            h = layer(h, allow)
        return types.SimpleNamespace(last_hidden_state=h)


class FakeAuto:
    """Stands in for AutoModelForCausalLM: fixed 'pretrained' weights that do not draw from the trainer's generator."""
    layer_types = LAYER_TYPES

    @classmethod
    def from_pretrained(cls, name, revision=None, dtype=None, attn_implementation=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261014)
            backbone = TinyHybridBackbone(cls.layer_types)
        return types.SimpleNamespace(model=backbone.to(dtype or torch.float32))


class FakeTokenizer:
    """One token per byte; the empty string has no token, as the real tokenizer gives for an empty option."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (byte % 53) for byte in text.encode()])

    def convert_tokens_to_ids(self, token):
        return SPECIAL_IDS[token]

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}\n")


def write_training_file(path, records=24):
    """Choice, noul and score records; some score levels are empty strings, so some option content spans are empty."""
    rows = []
    for i in range(records):
        if i % 3 == 0:
            keys = ["alpha", "beta", "gamma", "delta"]
            question = {"type": "choice", "instructions": f"pick the label of item {i}",
                        "criteria": {"alpha": "the first kind", "beta": None, "gamma": "a third kind", "delta": "last"},
                        "label": keys[i % 4]}
        elif i % 3 == 1:
            question = {"type": "noul", "instructions": f"is item {i} even?", "label": i % 2 == 0}
        else:
            question = {"type": "score", "instructions": f"rate item {i}",
                        "criteria": ["", "m", "high"] if i % 2 else ["low", "middle", "high"], "label": i % 3}
        rows.append({"state": f"record {i}: synthetic text {'x' * (i % 5)}", "questions": {"main": question}})
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_trainer(argv):
    """kev.train.main() with the real DecisionModel on the tiny backbone."""
    import kev.train as train_module
    with mock.patch.object(model_module, "AutoModelForCausalLM", FakeAuto), \
            mock.patch.object(train_module, "load_tokenizer", lambda name, revision=None: FakeTokenizer()), \
            mock.patch.object(sys, "argv", ["kev.train", *argv]), contextlib.redirect_stdout(io.StringIO()):
        train_module.main()


def decision_model(head_config, hybrid, seed=5):
    """A DecisionModel around the tiny backbone without adapters, assembled as runtime.load_bundle does."""
    from kev.heads import make_head, normalize_head_config
    torch.manual_seed(seed)
    model = model_module.DecisionModel.__new__(model_module.DecisionModel)
    torch.nn.Module.__init__(model)
    model.lm = TinyHybridBackbone()
    model.head_config = normalize_head_config(head_config)
    model.head = make_head(HIDDEN, head_dim=model.head_config["dim"], head_config=model.head_config)
    model.pad_id, model.hybrid, model.option_isolation, model.device = 0, hybrid, False, "cpu"
    model.eval()
    return model
