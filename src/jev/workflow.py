"""Run training and evaluation commands, recording their exit codes and outputs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from .config import read, root, sha, tree_path, worker_env, write_new
from .recipes import options


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(tree=None):
    repository = root()
    manifest = read(repository / "vendor/SOURCES.json")
    checked, errors = 0, []
    folders = {name: repository / snapshot["import_root"]
               for name, snapshot in manifest["snapshots"].items()}
    for name, snapshot in manifest["snapshots"].items():
        if tree and name != tree:
            continue
        folder = folders[name]
        # Resource snapshots can surround independently inventoried source
        # trees. Those declared children have their own exact file closure.
        nested = [other for key, other in folders.items()
                  if key != name and other != folder and other.is_relative_to(folder)]
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo") and p.name != "PROVENANCE.json"
                  and not any(p.is_relative_to(child) for child in nested)}
        expected = set(snapshot["files"])
        if actual != expected:
            errors.append(name + ": file inventory changed")
        for path, entry in snapshot["files"].items():
            target = folder / path
            if not target.is_file() or sha(target) != entry["sha256"]:
                errors.append(name + "/" + path)
            checked += 1
    if tree is None:
        for path, entry in manifest.get("provenance_files", {}).items():
            target = repository / path
            if not target.is_file() or sha(target) != entry["sha256"]:
                errors.append(path)
            checked += 1
        protocols = read(repository / "provenance/PROTOCOLS.json")
        for entry in protocols["files"]:
            target = repository / entry["path"]
            if not target.is_file() or sha(target) != entry["sha256"]:
                errors.append(entry["path"])
            checked += 1
        requirements = read(repository / "requirements/manifest.json")
        for name, entry in requirements["files"].items():
            target = repository / "requirements" / name
            if (not target.resolve().is_relative_to((repository / "requirements").resolve())
                    or target.is_symlink() or not target.is_file()
                    or target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]):
                errors.append("requirements/" + name)
            checked += 1
    if not checked:
        raise ValueError("source manifest contains no matching files")
    if errors:
        raise ValueError("source verification failed: " + ", ".join(errors[:12]))
    return checked


def validate_inputs(plan):
    problems = []
    for path, expected in plan["input_pins"].items():
        if not Path(path).is_file():
            problems.append("missing input: " + path)
        elif sha(path) != expected:
            problems.append("input bytes differ from recorded experiment: " + path)
    args = options(plan["argv"])
    for key in ("--base", "--data"):
        if not Path(args[key]).exists() and not any(args[key] in p for p in problems):
            problems.append("missing " + key + ": " + args[key])
    for ref in plan.get("references", []):
        if not (Path(ref["path"]) / "campaign/initial_receipt.json").is_file():
            problems.append("reference needs genuine initialization records: " + ref["run_id"] + " at " + ref["path"])
    from .validation import validate_reference_locations
    problems.extend(validate_reference_locations(plan))
    return problems


def gpu_check(gpu, site):
    if not gpu or "," in gpu:
        raise ValueError("select exactly one allocated GPU index or UUID")
    query = ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, text=True, capture_output=True, check=True)
    values = result.stdout.strip().split(",")
    if len(values) != 3:
        raise ValueError("GPU query did not resolve exactly one device")
    uuid, name, memory = [v.strip() for v in values]
    expected_name = site.get("hardware", {}).get("gpu_name_contains", "H20")
    if not isinstance(expected_name, str) or not expected_name or expected_name not in name:
        raise ValueError("site GPU policy requires " + str(expected_name) + "; got " + name)
    limit = site.get("hardware", {}).get("idle_memory_mib_below", 500)
    if int(memory) >= limit:
        raise ValueError("GPU is occupied: " + memory + " MiB; queue work on an allocated device")
    processes = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"],
                               text=True, capture_output=True, check=True).stdout.strip()
    if processes:
        raise ValueError("selected GPU has compute processes; choose an allocated idle device")
    return dict(uuid=uuid, name=name, memory_used_mib=int(memory))


def execute(argv, env, log_dir, stage):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    record = dict(schema="jev.stage.v1", stage=stage, argv=argv, started_utc=now(),
                  host=socket.gethostname(), python=platform.python_version(),
                  selected_environment={k: env.get(k) for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")})
    write_new(log_dir / (stage + ".command.json"), record)
    started = time.monotonic()
    with (log_dir / (stage + ".stdout.log")).open("x", encoding="utf-8") as out, (log_dir / (stage + ".stderr.log")).open("x", encoding="utf-8") as err:
        print(json.dumps(dict(stage=stage, log_dir=str(log_dir), started_utc=record["started_utc"])), flush=True)
        interruption = None
        process = subprocess.Popen(argv, env=env, stdout=out, stderr=err)
        try:
            code = process.wait()
        except BaseException as error:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = process.returncode
            interruption = error
    record.update(exit_code=code, ended_utc=now(), elapsed_seconds=time.monotonic() - started)
    write_new(log_dir / (stage + ".result.json"), record)
    if interruption is not None:
        raise interruption
    if code:
        raise RuntimeError(f"{stage} failed with native exit {code}; inspect {log_dir / (stage + '.stderr.log')}")
    return record


def train(plan, gpu, allow_environment_drift=False):
    if not plan.get("custom_source"):
        verify_sources(plan["tree"])
    problems = validate_inputs(plan)
    if problems:
        raise ValueError("\n".join(problems))
    from .validation import environment_report
    environment = environment_report(plan["site"]["python"]["training"], plan["source_root"],
                                     {**plan.get("environment", {}), **plan["site"].get("environment", {})})
    matching = environment["status"] in ("MATCH_RECORDED_VERSIONS", "MATCH_WITH_TORCH_CUDA_SUFFIX_NORMALIZATION")
    if not matching and not plan["scientific_overrides"] and not allow_environment_drift:
        raise ValueError("training environment differs from the recorded stack; run doctor --imports for details. An explicit --allow-environment-drift records a new environment rather than certifying reproduction")
    args = options(plan["argv"])
    cpu = args.get("--device") == "cpu"
    if cpu and not plan["scientific_overrides"]:
        raise ValueError("recorded paper recipes require the recorded GPU execution policy")
    device = None if cpu else gpu_check(gpu, plan["site"])
    output = Path(plan["output"])
    run_dir = output.parent
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    write_new(run_dir / "execution.json", plan)
    write_new(run_dir / "hardware.json", device or {"device": "cpu"})
    write_new(run_dir / "environment_report.json", environment)
    sources = {p.relative_to(plan["source_root"]).as_posix(): sha(p)
               for p in Path(plan["source_root"]).rglob("*.py") if "__pycache__" not in p.parts}
    write_new(run_dir / "source_files.json", sources)
    env = worker_env(plan["source_root"], gpu, {**plan.get("environment", {}), **plan["site"].get("environment", {})})
    execute(plan["argv"], env, run_dir, "train")
    if sources != {p.relative_to(plan["source_root"]).as_posix(): sha(p)
                   for p in Path(plan["source_root"]).rglob("*.py") if "__pycache__" not in p.parts}:
        raise ValueError("training source changed during the run")
    metrics = read(output / "training_metrics.json")
    if metrics.get("status") != "complete" or metrics.get("stopped_early") is not False:
        raise ValueError("native trainer exited without a complete full-horizon checkpoint")
    for key in ("optimizer_steps", "planned_optimizer_steps"):
        expected = plan["expected"].get(key)
        if expected is not None and metrics.get(key) != expected:
            raise ValueError("training metric differs from recorded plan: " + key)
    expected_records = plan["expected"].get("source_records")
    if expected_records is not None and metrics.get("source_records_seen") != expected_records:
        raise ValueError("native trainer consumed a different number of source records")
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_new(run_dir / "verified_checkpoint.json", dict(metrics=metrics, artifacts_sha256=artifacts, verified_utc=now()))
    return output


def export_checkpoint(checkpoint, destination, site, tree="e11", gpu=None, examples=None, source_root=None):
    gpu_check(gpu, site)
    source = Path(source_root).resolve() if source_root else tree_path(tree)
    if not source_root:
        verify_sources(tree)
    if not (source / "runtime/export_bundle.py").is_file():
        raise ValueError("selected implementation has no native bundle exporter; use checkpoint prediction or supply a compatible custom runtime")
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("export destination already exists")
    cfg = read(checkpoint / "training_config.json")
    saved_args = cfg.get("args", {})
    head = cfg.get("head_config") or saved_args.get("head_config")
    if isinstance(head, str):
        head = json.loads(head) if head else None
    head = head or dict(version=1, kind="pointer", dim=saved_args.get("head_dim", 256), readout="end")
    examples = examples or site.get("verification_examples")
    if not examples or not Path(examples).is_file():
        raise ValueError("export requires --verification-examples or site.verification_examples with real unlabelled requests")
    logs = destination.parent / (destination.name + ".export")
    logs.mkdir(parents=True, exist_ok=False)
    head_file = logs / "head.json"
    write_new(head_file, head)
    argv = [site["python"]["training"], "-P", "-m", "runtime.export_bundle", "--run", str(checkpoint),
            "--out", str(destination), "--name", destination.name, "--base", site["base_model"],
            "--base-revision", cfg.get("base_revision") or saved_args.get("base_revision") or "",
            "--head-config", str(head_file), "--verification-examples", str(Path(examples).resolve())]
    execute(argv, worker_env(source, gpu, site.get("environment")), logs, "export")
    return destination


def predict(bundle, requests, output, site, gpu=None, device="cuda"):
    if device == "cuda":
        gpu_check(gpu, site)
    bundle, requests, output = Path(bundle).resolve(), Path(requests).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("prediction output already exists")
    logs = output.parent / (output.name + ".logs")
    argv = [site["python"]["serving"], str(Path(__file__).with_name("predict_worker.py")),
            "--bundle", str(bundle), "--requests", str(requests), "--output", str(output), "--device", device]
    execute(argv, worker_env(bundle / "code", gpu, site.get("environment")), logs, "predict")
    return output


def predict_checkpoint(checkpoint, source, requests, output, site, gpu=None, device="cuda", model_name=None):
    if device == "cuda":
        gpu_check(gpu, site)
    checkpoint, source = Path(checkpoint).resolve(), Path(source).resolve()
    output, requests = Path(output).resolve(), Path(requests).resolve()
    if output.exists():
        raise ValueError("prediction output already exists")
    model_name = model_name or checkpoint.parent.name
    argv = [site["python"]["serving"], str(Path(__file__).with_name("predict_worker.py")),
            "--checkpoint", str(checkpoint), "--source-root", str(source), "--base", site["base_model"],
            "--model-name", model_name, "--requests", str(requests), "--output", str(output), "--device", device]
    env = worker_env(source, gpu, {**site.get("environment", {}), "KEV_LORA_SCALE": "1", "KEV_MERGE": "1"})
    execute(argv, env, output.parent / (output.name + ".logs"), "predict")
    return output
