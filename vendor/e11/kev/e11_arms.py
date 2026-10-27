"""Define two-epoch comparisons with matched one-epoch references.

Keep readout, rank-64 all-module adapters, data, rates, and augmentation fixed.
Pointer runs use LL_A; mean-span runs use POOL_ALL. kev.e11_reference checks
the corresponding saved step-zero records.
"""
from . import e458_arms

EPOCHS = 2
REFERENCE_EPOCHS = 1
RANK, TARGETS = e458_arms.REFERENCE["rank"], e458_arms.REFERENCE["targets"]
# The scheduler arguments written in kev.train: OneCycleLR(opt, max_lr=[lr, head_lr], total_steps=steps, pct_start=0.1).
# Every other argument is the installed constructor's default; kev.e11_reference reads the object back and records them.
PCT_START = 0.1

# reference: the one-epoch arm, the tree that trained it, and the value its --e458_arm argument recorded.
ARMS = {
    "POINTER_2EP": {"readout": "end", "roles": ["E11:pointer/two-epoch"],
                    "reference": {"arm": "LL_A", "block": "E1/E2", "e458_arm": ""}},
    "POOL_2EP": {"readout": "mean_span", "roles": ["E11:pooling/two-epoch"],
                 "reference": {"arm": "POOL_ALL", "block": "E4a/E5/E8", "e458_arm": "POOL_ALL"}},
}

# Trainer arguments that exist only in this tree, and the recorded protocol arguments an arm changes.
ARGUMENTS = ("e11_arm", "e11_reference_run", "e11_expected_head_numel", "e11_expected_adapter_numel",
             "e11_expected_optimizer_steps")
ARGUMENT_CHANGES = ("campaign_checkpoint_steps", "epochs")
# kev files in which this tree may differ from the tree that trained the reference.
SOURCE_FILES_CHANGED = {"E4a/E5/E8": ("e11_arms.py", "e11_reference.py", "train.py")}
SOURCE_FILES_CHANGED["E1/E2"] = tuple(sorted(set(SOURCE_FILES_CHANGED["E4a/E5/E8"]) | set(e458_arms.SOURCE_FILES_CHANGED)))


def arm(name):
    """One arm with everything that follows from its readout."""
    if name not in ARMS:
        raise ValueError(f"unknown E11 arm: {name!r}")
    row = ARMS[name]
    return {"name": name, "epochs": EPOCHS, "rank": RANK, "alpha": 2 * RANK, "targets": TARGETS,
            "target_modules": list(e458_arms.TARGET_MODULES[TARGETS]), "readout": row["readout"],
            "head_config": e458_arms.head_config(row["readout"]), "adapter": "reference",
            "structure": e458_arms.structure(RANK, TARGETS), "same_structure_as_reference": True,
            "argument_changes": list(ARGUMENT_CHANGES), "reference": dict(row["reference"]),
            "source_files_changed": list(SOURCE_FILES_CHANGED[row["reference"]["block"]]), "roles": list(row["roles"])}


def optimizer_steps(records, batch, accum, epochs=EPOCHS):
    """The trainer's own count: every epoch ends with a flush of its last, possibly shorter, accumulation group."""
    microbatches = -(-records // batch)
    return epochs * -(-microbatches // accum)


def epoch_updates(records, batch, accum):
    """Source records behind each optimizer update of one epoch, in order."""
    microbatches = -(-records // batch)
    rows = []
    for first in range(0, microbatches, accum):
        last = min(first + accum, microbatches)
        rows.append(min(last * batch, records) - first * batch)
    return rows
