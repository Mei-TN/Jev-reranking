"""Portable locations, immutable experiment inputs and source selection."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


def root():
    bundled = Path(__file__).parent / "_resources"
    if bundled.is_dir():
        from .resources import resource_root
        return resource_root(bundled)
    source = Path(__file__).resolve().parents[2]
    if (source / "vendor.zip").is_file():
        from .resources import resource_root
        return resource_root(source, source_layout=True)
    return source


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,149}", value):
        raise ValueError("identifier must be a single safe path component")
    return value


def tree_path(name):
    name = safe_id(name)
    repository = root()
    snapshot = read(repository / "vendor/SOURCES.json").get("snapshots", {}).get(name)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("import_root"), str):
        raise ValueError("unknown training tree: " + name)
    relative = PurePosixPath(snapshot["import_root"])
    if (relative.is_absolute() or ".." in relative.parts or "\\" in snapshot["import_root"]
            or ":" in snapshot["import_root"] or not relative.parts or relative.parts[0] != "vendor"):
        raise ValueError("training tree import_root must be a relative vendor path: " + name)
    path = repository.joinpath(*relative.parts)
    if not path.resolve().is_relative_to((repository / "vendor").resolve()):
        raise ValueError("training tree must be inside the preserved vendor sources: " + name)
    if not (path / "kev/train.py").is_file():
        raise ValueError("unknown training tree: " + name)
    return path


def load_site(path):
    path = Path(path).resolve()
    site = read(path)
    if site.get("schema") != "jev.site.v1":
        raise ValueError("site.schema must be jev.site.v1")
    for key in ("work_root", "base_model", "data_root"):
        value = Path(site[key]).expanduser()
        site[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    for role, value in site.get("python", {}).items():
        # Bare executable names (python3) resolve on PATH; paths resolve relative to site.
        if "/" in value or "\\" in value:
            p = Path(value).expanduser()
            # A venv interpreter is often a symlink. Resolving it changes the environment.
            site["python"][role] = os.path.abspath(path.parent / p if not p.is_absolute() else p)
    for key in ("verification_examples",):
        if site.get(key):
            p = Path(site[key]).expanduser()
            site[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key, value in site.get("reference_outputs", {}).items():
        p = Path(value).expanduser()
        site["reference_outputs"][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    site.setdefault("python", {})
    site["python"].setdefault("training", sys.executable)
    site["python"].setdefault("serving", site["python"]["training"])
    site["site_file"] = str(path)
    return site


def worker_env(tree, gpu=None, extra=None):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(name, None)
    env.update(PYTHONPATH=str(tree), PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    # A site may supply library locations, but cannot replace the selected implementation.
    env["PYTHONPATH"] = str(tree)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env
