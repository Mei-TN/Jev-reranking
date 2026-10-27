"""Command-line interface for configuration, training, export, inference, and scoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from . import __version__
from .config import load_site, read, root, tree_path, worker_env, write_new
from . import recipes, workflow


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def recipe_arguments(p):
    p.add_argument("--site", required=True)
    p.add_argument("--study", choices=["e1e2", "e458", "e11"], default="e1e2")
    p.add_argument("--recipe", help="custom jev.recipe.v1 JSON; records a new experiment")
    p.add_argument("--run-id")
    p.add_argument("--arm")
    p.add_argument("--fold", default="F2")
    p.add_argument("--seed", type=int, default=44)


def parser():
    p = argparse.ArgumentParser(description="JEV reproducible research workflows")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init", help="write a portable site configuration")
    q.add_argument("--site", default="configs/site.local.json")
    q = sub.add_parser("list", help="list recorded paper runs or the paper design map")
    q.add_argument("--study", choices=["e1e2", "e458", "e11"])
    sub.add_parser("verify", help="verify preserved source bytes")
    q = sub.add_parser("fork", help="copy a preserved implementation to an editable extension directory")
    q.add_argument("--tree", default="e11")
    q.add_argument("--output", required=True)
    q = sub.add_parser("plan", help="print a complete native command without model imports")
    recipe_arguments(q)
    q.add_argument("--save")
    q = sub.add_parser("doctor", help="check source, pinned inputs and training interpreter without loading a model")
    recipe_arguments(q)
    q.add_argument("--imports", action="store_true", help="import heavy training libraries in a fresh interpreter")
    q = sub.add_parser("train", help="run one exact paper recipe or an explicit custom recipe")
    recipe_arguments(q)
    q.add_argument("--gpu", required=True)
    q.add_argument("--allow-environment-drift", action="store_true")
    q = sub.add_parser("export", help="merge a checkpoint and verify a fresh offline bundle reload")
    q.add_argument("--site", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--tree", default="e11")
    q.add_argument("--source-root", help="custom extension with its own native runtime exporter")
    q.add_argument("--gpu", required=True)
    q.add_argument("--verification-examples")
    q = sub.add_parser("predict", help="native bundle prediction on unlabelled JSONL requests")
    q.add_argument("--site", required=True)
    model_source = q.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--bundle")
    model_source.add_argument("--checkpoint")
    q.add_argument("--tree", default="e11")
    q.add_argument("--source-root")
    q.add_argument("--model-name")
    q.add_argument("--requests", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--gpu")
    q.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    q = sub.add_parser("serve", help="start the native text decision service in the foreground")
    q.add_argument("--site", required=True)
    q.add_argument("--bundle", required=True)
    q.add_argument("--port", type=int, default=32100)
    q.add_argument("--log-dir", required=True)
    q.add_argument("--gpu", required=True)
    q = sub.add_parser("pipeline", help="train, verified export, native prediction and offline scoring")
    recipe_arguments(q)
    q.add_argument("--gpu", required=True)
    q.add_argument("--requests", required=True)
    q.add_argument("--labels", required=True)
    q.add_argument("--verification-examples", required=True)
    q.add_argument("--inference-format", choices=["auto", "bundle", "checkpoint"], default="auto")
    q.add_argument("--allow-environment-drift", action="store_true")
    q.add_argument("--resume", action="store_true", help="reuse only stages proven complete by native receipts and artifact bytes")
    q = sub.add_parser("assets", help="audit original full pointer/pooling-pair asset manifest")
    q.add_argument("--asset-root", required=True)
    q.add_argument("--group", choices=["train", "export", "score"])
    q = sub.add_parser("reproduce-scores", help="reconstruct original 72-model stored scores and supported appendix values on CPU")
    q.add_argument("--archive", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("evaluate", help="score native predictions or a live endpoint; see jev evaluate --help")
    q.add_argument("arguments", nargs=argparse.REMAINDER)
    q = sub.add_parser("benchmark", help="run the preserved JevBench/Kev and MC11 protocols; see jev benchmark --help")
    q.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # Forward the scorer's complete argument parser, including its --help.
    if argv and argv[0] == "evaluate":
        from .evaluate import main as evaluate
        return evaluate(argv[1:])
    if argv and argv[0] == "benchmark":
        from .benchmarks import main as benchmark
        return benchmark(argv[1:])
    p = parser()
    a = p.parse_args(argv)
    try:
        if a.command == "init":
            value = read(root() / "configs/site.example.json")
            target = Path(a.site).resolve()
            # Explicit absolute defaults keep a custom destination meaningful.
            for key, suffix in (("work_root", "work"), ("base_model", "assets/NeoHorse-1-4B-hf"), ("data_root", "assets/training")):
                value[key] = str(Path.cwd() / suffix)
            write_new(target, value)
            emit(dict(site=str(target), next="edit paths/interpreters, then run jev doctor"))
        elif a.command == "verify":
            emit(dict(status="PASS", preserved_files=workflow.verify_sources()))
        elif a.command == "fork":
            workflow.verify_sources(a.tree)
            source = tree_path(a.tree)
            destination = Path(a.output).resolve()
            if destination.exists():
                raise ValueError("extension destination already exists")
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            from .config import sha
            write_new(destination / "JEV_ORIGIN.json", dict(tree=a.tree,
                files={p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*.py")}))
            emit(dict(source_root=str(destination), recipe_field="source_root", original_tree=a.tree))
        elif a.command == "list":
            if a.study in recipes.STUDIES:
                plan = read(root() / "provenance/plans" / (a.study + ".json"))
                emit([dict(run_id=r["run_id"], arm=r["arm"], fold=r["fold"], seed=r["seed"], roles=r["roles"]) for r in plan["runs"]])
            else:
                emit(read(root() / "configs/catalog.json"))
        elif a.command in ("plan", "doctor", "train", "pipeline"):
            plan = recipes.build(a)
            if a.command == "plan":
                if a.save:
                    write_new(a.save, plan)
                emit(plan)
            elif a.command == "doctor":
                count = workflow.verify_sources(plan["tree"]) if not plan.get("custom_source") else len(list(Path(plan["source_root"]).rglob("*.py")))
                problems = workflow.validate_inputs(plan)
                python = plan["site"]["python"]["training"]
                interpreter = shutil.which(python) or (python if Path(python).is_file() else None)
                if not interpreter:
                    problems.append("training interpreter is unavailable: " + python)
                if a.imports and interpreter:
                    from .validation import environment_report
                    environment = environment_report(python, plan["source_root"], plan["site"].get("environment"))
                    if environment["status"] not in ("MATCH_RECORDED_VERSIONS", "MATCH_WITH_TORCH_CUDA_SUFFIX_NORMALIZATION"):
                        problems.append("environment: " + environment["status"])
                else:
                    environment = None
                emit(dict(status="READY" if not problems else "INCOMPLETE", source_files_verified=count,
                          run_id=plan["run_id"], problems=problems, environment=environment,
                          scope="CPU preparation only; GPU model execution and numeric reproduction are separate"))
                return 0 if not problems else 2
            else:
                if a.command == "train":
                    checkpoint = workflow.train(plan, a.gpu, a.allow_environment_drift)
                    emit(dict(checkpoint=str(checkpoint)))
                else:
                    from .pipeline import run_pipeline
                    emit(run_pipeline(plan, a.gpu, a.requests, a.labels, a.verification_examples,
                        inference_format=a.inference_format, allow_environment_drift=a.allow_environment_drift,
                        resume=a.resume))
        elif a.command == "export":
            site = load_site(a.site)
            workflow.gpu_check(a.gpu, site)
            emit(dict(bundle=str(workflow.export_checkpoint(a.checkpoint, a.output, site, a.tree, a.gpu, a.verification_examples, a.source_root))))
        elif a.command == "predict":
            site = load_site(a.site)
            if a.device == "cuda":
                workflow.gpu_check(a.gpu, site)
            if a.checkpoint:
                source = Path(a.source_root).resolve() if a.source_root else tree_path(a.tree)
                emit(dict(predictions=str(workflow.predict_checkpoint(a.checkpoint, source, a.requests, a.output, site, a.gpu, a.device, a.model_name))))
            else:
                emit(dict(predictions=str(workflow.predict(a.bundle, a.requests, a.output, site, a.gpu, a.device))))
        elif a.command == "serve":
            site, bundle = load_site(a.site), Path(a.bundle).resolve()
            workflow.gpu_check(a.gpu, site)
            meta = read(bundle / "model_manifest.json")
            head = bundle / "model_head_config.json"
            # Exported bundles already contain their authoritative head config.
            candidates = [bundle / "head_config.json", bundle / "classification_head_config.json"]
            head = next((p for p in candidates if p.is_file()), None)
            if head is None:
                head = Path(a.log_dir).resolve().with_suffix(".head.json")
                write_new(head, meta["head_config"])
            cmd = [site["python"]["serving"], "-P", "-m", "runtime.serve_bundle", "--bundle", str(bundle),
                   "--model-id", meta["name"], "--head-config", str(head), "--port", str(a.port), "--log-dir", str(Path(a.log_dir).resolve())]
            return subprocess.call(cmd, env=worker_env(bundle / "code", a.gpu, site.get("environment")))
        elif a.command == "assets":
            manifest = read(root() / "provenance/portable_pair/ASSET_MANIFEST.json")
            from .config import sha
            rows = []
            for key, item in manifest["assets"].items():
                if a.group and a.group not in item["groups"]:
                    continue
                path = Path(a.asset_root) / item["path"]
                state = "MISSING" if not path.is_file() else "PASS" if path.stat().st_size == item["bytes"] and sha(path) == item["sha256"] else "MISMATCH"
                rows.append(dict(asset=key, status=state, groups=item["groups"]))
            emit(dict(scope="original full F2s44 pointer/pooling pair asset pins; not a complete all-design release", assets=rows))
            return 0 if all(r["status"] == "PASS" for r in rows) else 2
        elif a.command == "reproduce-scores":
            from .scoring import reproduce_scores
            emit(reproduce_scores(a.archive, a.output))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("JEV: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
