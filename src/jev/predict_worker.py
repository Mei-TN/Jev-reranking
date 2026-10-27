"""Run native bundle prediction in a fresh interpreter."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_source = p.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--bundle", type=Path)
    model_source.add_argument("--checkpoint", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--base", type=Path)
    p.add_argument("--model-name")
    for name in ("requests", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-state", type=int, default=8192)
    p.add_argument("--max-branch", type=int, default=8192)
    p.add_argument("--max-packed", type=int, default=8192)
    a = p.parse_args()
    import torch
    if a.bundle:
        sys.path.insert(0, str(a.bundle / "code"))
        from runtime.load_bundle import load_bundle
        manifest = json.loads((a.bundle / "model_manifest.json").read_text())
        tok, model = load_bundle(a.bundle, a.device)
    else:
        if not a.source_root or not a.base or not a.model_name:
            raise ValueError("checkpoint prediction needs source-root, local base and model-name")
        sys.path.insert(0, str(a.source_root))
        from kev.evaluate import load
        # Same declared location rebinding used by the original native exporter.
        # Checkpoint metadata on disk remains unchanged.
        original_load = torch.load
        def local_load(path, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            value = original_load(path, *args, **kwargs)
            if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == a.checkpoint / "head.pt":
                value = dict(value)
                value["base"] = str(a.base)
            return value
        torch.load = local_load
        try:
            tok, model = load(str(a.checkpoint), a.device, dtype=torch.bfloat16, merge=True)
        finally:
            torch.load = original_load
        head_config = getattr(model, "head_config", None) or dict(
            version=1, kind="pointer", dim=model.head.q.out_features, readout="end")
        manifest = dict(name=a.model_name, head_config=head_config)
    from kev.api import SystemOneRequest, to_record
    seen = set()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.requests.open(encoding="utf-8") as source, a.output.open("x", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rid, request = row["id"], dict(row["request"])
            if rid in seen:
                raise ValueError("duplicate request id: " + str(rid))
            seen.add(rid)
            if "model" in request and request["model"] != manifest["name"]:
                raise ValueError("request model differs from bundle")
            request["model"] = manifest["name"]
            if set(request) - {"model", "state", "questions"}:
                raise ValueError("only native text request inputs are supported")
            for q in request["questions"].values():
                if set(q) - {"type", "instructions", "criteria"}:
                    raise ValueError("labels/metadata cannot be model inputs")
            rec, metadata = to_record(SystemOneRequest.model_validate(request))
            enc = model.encode(tok, rec, max_state=a.max_state, max_branch=a.max_branch, strict=True)
            if len(enc["ids"]) > a.max_packed:
                raise ValueError("packed request exceeds declared token limit")
            started = time.perf_counter()
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
                started = time.perf_counter()
            with torch.inference_mode():
                probabilities = model.probs(enc)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            if len(probabilities) != len(metadata):
                raise ValueError("model probability question count differs from the request")
            response = dict(model=manifest["name"], probability_source="classification_head",
                            head_kind=manifest["head_config"]["kind"], head_config=manifest["head_config"],
                            probabilities={}, input_tokens=len(enc["ids"]), forward_seconds=time.perf_counter()-started)
            for values, meta in zip(probabilities, metadata):
                keys = meta["keys"] if meta["type"] == "choice" else ["false", "true"] if meta["type"] == "noul" else list(meta["legend"])
                values = values.tolist()
                if len(keys) != len(values):
                    raise ValueError("model probability option count differs from the request")
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
                    raise ValueError("model probabilities must be finite, bounded and sum to one")
                response["probabilities"][meta["id"]] = dict(zip(keys, values))
            out.write(json.dumps(dict(id=rid, response=response), ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
    if not seen:
        raise ValueError("empty request population")


if __name__ == "__main__":
    main()
