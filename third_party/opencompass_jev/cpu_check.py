"""Exercise the applied dataset/scorer interfaces with synthetic records, CPU only."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="new owned synthetic-fixture directory")
    parser.add_argument("--runtime-tree", type=Path, help="optional study tree containing runtime/decision_head_api.py")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from apply_opencompass_jev import load_manifest, verify_source
    verify_source(source, load_manifest(), True)
    work = args.work_dir.absolute()
    if work.exists() or work.resolve().is_relative_to(source):
        parser.error("work-dir must be new and outside the OpenCompass source checkout")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    work.mkdir(parents=True, exist_ok=False)
    for name in ["HOME", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TMPDIR"]:
        folder = work / name.lower()
        folder.mkdir()
        os.environ[name] = str(folder)
    sys.path.insert(0, str(source))
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized")
    attempts = []

    def forbidden_cuda(*_args, **_kwargs):
        attempts.append("torch.cuda._lazy_init")
        raise RuntimeError("CUDA initialization is forbidden in this CPU fixture")

    torch.cuda._lazy_init = forbidden_cuda
    datasets = importlib.import_module("opencompass.datasets.decision_benchmarks")
    native = importlib.import_module("opencompass.datasets.decision_native")
    inferencer = importlib.import_module("opencompass.openicl.icl_inferencer")
    for module in [datasets, native, inferencer]:
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("import escaped the selected checkout")
    import urllib.request

    def forbidden_network(*_args, **_kwargs):
        raise RuntimeError("network access is forbidden in this CPU fixture")

    urllib.request.urlopen = forbidden_network
    checks = []
    cases = [
        dict(id="fixture-choice", family="choice", state="Synthetic state.",
             question=dict(type="choice", instructions="Choose.", criteria={"b": "B", "a": "A"}),
             labels=["b", "a"], expected="b", split="public"),
        dict(id="fixture-score", family="score", state="Synthetic state.",
             question=dict(type="score", instructions="Rate.", criteria=["low", "high"]),
             labels=["0", "1"], expected=1, split="public"),
        dict(id="fixture-noul", family="noul", state="Synthetic state.",
             question=dict(type="noul", instructions="Decide."), labels=["no", "yes"],
             expected="yes", split="public"),
    ]
    files = {}
    for subset, record in zip(["easy", "original", "hard"], cases):
        path = work / (subset + ".jsonl")
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        files[("jevbench", "datasets/public/" + subset + ".jsonl")] = path
    kev = [
        dict(state="Synthetic state.", _meta=dict(id="kev-1", group_id="g1", source="fixture", variant="clean"),
             questions={"q": dict(type="choice", instructions="Choose.", criteria={"b": "B", "a": "A"},
                                  label="b", src="fixture-choice")}),
        dict(state="Synthetic state.", _meta=dict(id="kev-2", group_id="g2", source="fixture", variant="clean"),
             questions={"q": dict(type="noul", instructions="Decide.", label=1, src="fixture-noul")}),
    ]
    path = work / "kev.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in kev), encoding="utf-8")
    files[("kev", "evals/v7/decision-v7/development.jsonl")] = path
    # Only the fixture's input locator is replaced in this process. Original
    # dataset/scorer files, scoring logic and installed environments stay intact.
    datasets.source_file = lambda project, relative, data_root=None: files[(project, relative)]
    selection = work / "jev.selection.json"
    selection.write_text(json.dumps({"selected_ids": [row["id"] for row in cases]}), encoding="utf-8")
    loaded = native.NativeDecisionDataset.load("jevbench", str(selection))["test"]
    assert len(loaded) == 3 and [row["id"] for row in loaded] == [row["id"] for row in cases]
    request = json.loads(loaded[0]["request_json"])["request"]
    assert list(request["questions"]["decision"]["criteria"]) == ["b", "a"]
    assert "expected" not in request["questions"]["decision"] and "labels" not in request["questions"]["decision"]
    checks.append("native dataset selection/order and unlabelled request projection")
    references = [row["reference"] for row in loaded]
    predictions = [json.dumps({"probabilities": value}) for value in
                   [{"b": 0.9, "a": 0.1}, {"0": 0.1, "1": 0.9}, {"no": 0.1, "yes": 0.9}]]
    evaluator = native.NativeJevBenchEvaluator()
    score = evaluator.score(predictions, references)
    assert score["accuracy"] == score["macro_accuracy"] == score["schema_validity"] == 100
    assert score["n_planned"] == 3
    checks.append("three native JevBench question types and scorer aggregation")
    tie = evaluator.score([json.dumps({"probabilities": {"b": 0.5, "a": 0.5}})], references[:1])
    assert tie["accuracy"] == 0 and tie["details"][0]["predicted"] == "a"
    invalid = evaluator.score(["{}", *predictions[1:]], references)
    assert invalid["n_planned"] == 3 and math.isclose(invalid["accuracy"], 200 / 3)
    assert math.isclose(invalid["schema_validity"], 200 / 3)
    checks.append("JevBench lexical tie rule and invalid-output denominator")
    selection = work / "kev.selection.json"
    selection.write_text(json.dumps({"selected_ids": [row["_meta"]["id"] for row in kev]}), encoding="utf-8")
    loaded = native.NativeDecisionDataset.load("kev", str(selection), "decision-v7", "development")["test"]
    references = [row["reference"] for row in loaded]
    predictions = [json.dumps({"probabilities": {"q": value}}) for value in
                   [{"b": 0.9, "a": 0.1}, {"false": 0.1, "true": 0.9}]]
    evaluator = native.NativeKevEvaluator()
    score = evaluator.score(predictions, references)
    assert score["requested_records"] == 2 and score["invalid_records"] == 0 and score["accuracy"] == 100
    tie = evaluator.score([json.dumps({"probabilities": {"q": {"b": 0.5, "a": 0.5}}})], references[:1])
    assert tie["accuracy"] == 100
    invalid = evaluator.score(["{}", predictions[1]], references)
    assert invalid["requested_records"] == 2 and invalid["invalid_records"] == 1 and invalid["accuracy"] is None
    checks.append("Kev option-order tie rule, native scorer and invalid-output rejection")
    assert "batch_size" in inspect.signature(inferencer.ParallelGenInferencer.__init__).parameters
    if args.runtime_tree:
        tree = args.runtime_tree.resolve(strict=True)
        sys.path.insert(0, str(tree))
        runtime = importlib.import_module("runtime.decision_head_api")
        assert Path(runtime.__file__).resolve().is_relative_to(tree)
        assert runtime.HeadJevBenchEvaluator.probability_source == "classification_head"
        assert runtime.HeadKevEvaluator.probability_source == "classification_head"
        checks.append("paper runtime client and evaluator imports")
    assert not torch.cuda.is_initialized() and not attempts
    print(json.dumps({"status": "CPU_SYNTHETIC_INTERFACE_CHECKS_PASS", "checks": checks,
                      "CUDA_initialization_attempts": attempts, "model_or_HTTP_requests": 0,
                      "scope": "Synthetic loader/scorer boundaries only; no full frozen population or model benchmark-equivalence claim."}))


if __name__ == "__main__":
    main()
