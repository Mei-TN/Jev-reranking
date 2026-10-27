"""Define readout, adapter-scope, and rank comparisons.

The matched LL_A reference uses rank-64 adapters in all target modules and
an option-ending pointer. kev.e458_reference validates and loads its saved
step-zero records for the same fold and seed.
"""

REFERENCE_ARM = "LL_A"
REFERENCE = {"rank": 64, "targets": "all", "readout": "end"}

# LoRA target leaves on the hybrid backbone, in the order DecisionModel lists them.
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
DELTANET = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
TARGET_MODULES = {"all": ATTENTION + MLP + DELTANET, "attn": ATTENTION + DELTANET}

# adapter: "reference" = every LoRA tensor of the arm has the reference's name and shape and is copied from it;
#          "ordinary"  = the rank differs, so no LoRA tensor has the reference's shape; the ordinary initialization
#                        stands and is not called matched.
ARMS = {
    "POINTER_NOFFN": {"rank": 64, "targets": "attn", "readout": "end", "adapter": "reference",
                      "roles": ["E4a:pointer/NoFFN", "E5:NoFFN-rank64"]},
    "POOL_ALL": {"rank": 64, "targets": "all", "readout": "mean_span", "adapter": "reference",
                 "roles": ["E4a:pooling/all", "E8:mean-content"]},
    "POOL_NOFFN": {"rank": 64, "targets": "attn", "readout": "mean_span", "adapter": "reference",
                   "roles": ["E4a:pooling/NoFFN"]},
    "POINTER_ALL_R28": {"rank": 28, "targets": "all", "readout": "end", "adapter": "ordinary",
                        "roles": ["E5:all-rank28"]},
    "POINTER_LAST_CONTENT": {"rank": 64, "targets": "all", "readout": "last_span", "adapter": "reference",
                             "roles": ["E8:last-content"]},
}
REFERENCE_ROLES = ["E4a:pointer/all", "E5:all-rank64", "E8:end-delimiter"]

# Trainer arguments that exist only in this tree, and recorded arguments that never define the protocol.
ARGUMENTS = ("e458_arm", "e458_reference_run", "e458_expected_head_numel", "e458_expected_adapter_numel")
ARGUMENTS_NOT_COMPARED = ("out", "campaign_timeout_seconds")
# kev files in which this tree may differ from the tree that trained the reference.
SOURCE_FILES_CHANGED = ("e458_arms.py", "e458_reference.py", "heads.py", "model.py", "train.py")


def head_config(readout):
    return {"dim": 256, "kind": "pointer", "readout": readout, "rng_anchor_dim": 256, "version": 1}


def structure(rank, targets):
    """LoRA structure label. Runs share adapter tensors, optimizer membership and dropout draw shapes only within one."""
    return f"{targets}/r{rank}"


def arm(name):
    """One arm with everything that follows from its three declared settings."""
    if name not in ARMS:
        raise ValueError(f"unknown E4a/E5/E8 arm: {name!r}")
    row = ARMS[name]
    changes = {}
    if row["rank"] != REFERENCE["rank"]:
        changes["lora"] = row["rank"]
    if row["targets"] != REFERENCE["targets"]:
        changes["lora_targets"] = row["targets"]
    if row["readout"] != REFERENCE["readout"]:
        changes["head_config"] = head_config(row["readout"])
    if (row["adapter"] == "reference") != (row["rank"] == REFERENCE["rank"]):
        raise ValueError(f"{name}: only a rank-{REFERENCE['rank']} arm has LoRA tensors of the reference's shape")
    label = structure(row["rank"], row["targets"])
    return {"name": name, "rank": row["rank"], "alpha": 2 * row["rank"], "targets": row["targets"],
            "target_modules": list(TARGET_MODULES[row["targets"]]), "readout": row["readout"],
            "head_config": head_config(row["readout"]), "adapter": row["adapter"], "structure": label,
            "same_structure_as_reference": label == structure(REFERENCE["rank"], REFERENCE["targets"]),
            "argument_changes": changes, "roles": list(row["roles"])}


def head_numel(hidden_size, dim=256):
    """q and k projections with bias."""
    return 2 * (dim * hidden_size + dim)


def adapter_numel(modules, rank, targets):
    """LoRA elements for `modules` = [{"leaf", "in_features", "out_features"}, ...] of the base model."""
    leaves = set(TARGET_MODULES[targets])
    return sum(rank * (m["in_features"] + m["out_features"]) for m in modules if m["leaf"] in leaves)
