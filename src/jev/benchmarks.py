"""Configure local resources for JevBench, Kev, and MC11 evaluation."""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .config import read, root, sha, write_new


SCHEMA = "jev.benchmark-site.v1"
ORIGINAL_RUNNER_SHA256 = "2960be8ea6921784d3842257e6d06dce4061571f9328452a1ff639df9230369c"
E11_RUNNER_SHA256 = "7e2a4bdc97647b54fc0e0b775eb7699a6db2c229ecaf42af76dd3305eae211a2"
PROFILES = {
    "one_epoch": dict(directory="", runner_sha256=ORIGINAL_RUNNER_SHA256),
    "e11": dict(directory="followup/evaluation", runner_sha256=E11_RUNNER_SHA256),
}
NATIVE_MODULES = ("common", "jk_audit", "population", "replica_cache", "scores")


def _lexical_python(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError("controller/role Python must be a nonempty executable name or path")
    if "/" not in value and "\\" not in value:
        return value
    path = Path(value).expanduser()
    return os.path.abspath(parent / path if not path.is_absolute() else path)


def _site(path):
    path = Path(path).expanduser().resolve()
    site = read(path)
    if site.get("schema") != SCHEMA:
        raise ValueError("benchmark site.schema must be " + SCHEMA)
    guard = site.get("guard")
    if not isinstance(guard, dict):
        raise ValueError("benchmark site needs an explicit guard")
    for key in ("hostnames", "accounts"):
        if not isinstance(guard.get(key), list) or not guard[key] or any(not isinstance(value, str) or not value for value in guard[key]):
            raise ValueError("guard." + key + " must be a nonempty list")
    if not isinstance(guard.get("uids"), list) or not guard["uids"] or any(type(value) is not int or value < 0 for value in guard["uids"]):
        raise ValueError("guard.uids must explicitly list allowed numeric accounts")
    if not isinstance(guard.get("gpu_name_contains"), str) or not guard["gpu_name_contains"]:
        raise ValueError("guard.gpu_name_contains must name the expected actual hardware")
    if type(guard.get("idle_memory_mib_below")) is not int or guard["idle_memory_mib_below"] <= 0:
        raise ValueError("guard.idle_memory_mib_below must be a positive integer")
    site["controller_python"] = _lexical_python(site.get("controller_python", sys.executable), path.parent)
    native = site.setdefault("native_options", {})
    if not isinstance(native, dict):
        raise ValueError("native_options must be an object")
    for key, value in native.items():
        if not isinstance(value, str) or not value:
            raise ValueError("native_options values must be nonempty strings")
        if key.endswith("_python"):
            native[key] = _lexical_python(value, path.parent)
        elif key != "base_revision":
            item = Path(value).expanduser()
            native[key] = str((path.parent / item).resolve() if not item.is_absolute() else item.resolve())
    if site.get("role_bootstraps"):
        raise ValueError("use native_options role Python paths; historical role_bootstraps are not part of this public package")
    return path, site


def _profile(protocol):
    if protocol not in PROFILES:
        raise ValueError("unknown native benchmark protocol: " + str(protocol))
    return PROFILES[protocol]


def _verify_original(protocol="one_epoch"):
    profile = _profile(protocol)
    prefix = profile["directory"] + "/" if profile["directory"] else ""
    folder = root() / "vendor/evaluation" / profile["directory"]
    original = folder / "run_evaluation.py"
    if not original.is_file() or sha(original) != profile["runner_sha256"]:
        raise ValueError("native evaluation runner differs from the pinned public source")
    snapshot = read(root() / "vendor/SOURCES.json")["snapshots"]["evaluation"]
    identities = {}
    required = {"run_evaluation.py", "mc_worker.py", "EXPECTED_MC_POPULATION_V2.json"} | {name + ".py" for name in NATIVE_MODULES}
    if not {prefix + name for name in required} <= set(snapshot["files"]):
        raise ValueError("native evaluation source manifest lacks required workers")
    for name in sorted(required):
        path, expected = folder / name, snapshot["files"][prefix + name]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("native benchmark source changed: " + name)
        if protocol == "e11" and name != "run_evaluation.py" and expected["sha256"] != snapshot["files"][name]["sha256"]:
            raise ValueError("E11 benchmark worker/population differs from the retained original: " + name)
        identities[name] = expected["sha256"]
    return original, identities


def _load_original(protocol="one_epoch"):
    original, identities = _verify_original(protocol)
    folder = original.parent.resolve()
    for name in NATIVE_MODULES:
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, "__file__", "")).resolve() != folder / (name + ".py"):
            raise ValueError("native benchmark controller requires a fresh interpreter; foreign module: " + name)
    old = list(sys.path)
    try:
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("jev_original_benchmark_runner", original)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path[:] = old
    for name in NATIVE_MODULES:
        if Path(sys.modules[name].__file__).resolve() != folder / (name + ".py"):
            raise ValueError("native benchmark worker imported from another source: " + name)
    return runner, identities


def _checkpoint_continuity(plan):
    config = read(Path(plan["checkpoint"]) / "training_config.json")
    recorded = config.get("dse", {}).get("source_sha256")
    source = Path(plan["code_tree"]) / "kev"
    actual = {path.name: sha(path) for path in sorted(source.glob("*.py")) if not path.name.startswith("._")}
    if not isinstance(recorded, dict) or not recorded or recorded != actual:
        raise ValueError("checkpoint training source hashes do not equal the selected kev code tree; preserve historical metadata and select its actual source")
    return dict(training_config_sha256=sha(Path(plan["checkpoint"]) / "training_config.json"),
                kev_source_sha256=actual, source_tree=str(source.parent))


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _machine_and_gpu(selector, guard):
    """Resource-only derivative of the recorded native machine guard."""
    if (sys.platform != "linux" or socket.gethostname() not in guard["hostnames"]
            or getpass.getuser() not in guard["accounts"] or os.geteuid() not in guard["uids"]):
        raise ValueError("actual host/account is outside the explicit benchmark site guard")
    query = subprocess.run(["nvidia-smi", "--id=" + selector,
        "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    fields = [part.strip() for part in query.stdout.strip().split(",")]
    if (len(fields) != 4 or guard["gpu_name_contains"] not in fields[2]
            or not fields[0].isdigit() or not fields[1].startswith("GPU-")):
        raise ValueError("selected GPU does not match the explicit site's actual hardware class")
    memory_mib = int(fields[3])
    if memory_mib < 0:
        raise ValueError("selected GPU returned an invalid memory-occupancy reading")
    if memory_mib >= guard["idle_memory_mib_below"]:
        raise ValueError("selected GPU is occupied; no benchmark was launched")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    if any(line.split(",")[0].strip() == fields[1] for line in apps.stdout.splitlines()):
        raise ValueError("selected GPU has an existing compute process")
    return dict(index=int(fields[0]), uuid=fields[1], name=fields[2], memory_mib=memory_mib,
        host=socket.gethostname(), boot_id=_boot_id(), account=getpass.getuser(),
        sampled_epoch=time.time(), gpu_query=query.stdout, compute_query=apps.stdout)


def _sidecar(output, suffix):
    output = Path(output)
    return output.with_name(output.name + "." + suffix)




def _controller_capabilities():
    """Check actual Linux APIs before creating the native output or adapter sidecars."""
    result = dict(platform=sys.platform, python_version=sys.version,
                  pidfd_open=callable(getattr(os, "pidfd_open", None)),
                  pidfd_send_signal=callable(getattr(signal, "pidfd_send_signal", None)))
    if sys.platform != "linux":
        result["status"] = "NON_LINUX_CPU_PREPARATION_ONLY"
        return result
    if not result["pidfd_open"] or not result["pidfd_send_signal"]:
        raise ValueError("controller_python lacks Linux os.pidfd_open/signal.pidfd_send_signal; "
                         "select a Linux controller Python providing both APIs in the benchmark site; "
                         "the model role may use a different Python")
    try:
        descriptor = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(descriptor, 0)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError("controller_python cannot use the native Linux pidfd APIs: " + str(error)) from error
    result["status"] = "SELF_PIDFD_PROBE_PASSED"
    return result


def _worker(site_path, site, forwarded, expected_site_sha=None, protocol="one_epoch"):
    if expected_site_sha and sha(site_path) != expected_site_sha:
        raise ValueError("benchmark site file changed between launcher and controller")
    profile = _profile(protocol)
    runner, native_sources = _load_original(protocol)
    if set(site["native_options"]) - set(runner.DEFAULTS):
        raise ValueError("unknown native_options: " + ", ".join(sorted(set(site["native_options"]) - set(runner.DEFAULTS))))
    native_args = [value for key, value in site["native_options"].items() for value in ("--" + key.replace("_", "-"), value)] + list(forwarded)
    args = runner.parser().parse_args(native_args)
    if protocol == "e11" and args.training_contract not in ("two_epoch_F2", "two_epoch_F3"):
        raise ValueError("--protocol e11 requires the native --training-contract two_epoch_F2 or two_epoch_F3")
    plan = runner.prepare(args)
    continuity = _checkpoint_continuity(plan)
    controller_capabilities = _controller_capabilities()
    adapter = dict(schema="jev.benchmark-adapter.v1", adapter_source_sha256=sha(__file__),
        original_runner_sha256=profile["runner_sha256"], native_sources=native_sources,
        native_profile=dict(protocol=protocol, directory=profile["directory"] or ".",
            runner_sha256=profile["runner_sha256"], training_contract=plan.get("training_contract")),
        site_file=str(site_path), site_sha256=sha(site_path), site_label=site.get("label"),
        resource_guard=site["guard"], checkpoint_source_continuity=continuity,
        modified_hooks=["machine_and_gpu"], controller_python=sys.executable,
        controller_capabilities=controller_capabilities,
        historical_hardware_reproduction_claim=False,
        scope="New execution using unchanged native benchmark protocols and honest site hardware records.")
    if args.prepare_only:
        print(json.dumps(dict(status="CPU_PREPARED_NOT_LAUNCHED", plan=plan, adapter=adapter), indent=2))
        return 0
    observations = []
    def resource_guard(selector):
        result = _machine_and_gpu(selector, site["guard"])
        observations.append(result)
        return result
    runner.machine_and_gpu = resource_guard
    output = Path(plan["output"])
    write_new(_sidecar(output, "SITE_ADAPTER.json"), adapter)
    terminal = dict(schema=adapter["schema"], site_sha256=adapter["site_sha256"],
        adapter_source_sha256=adapter["adapter_source_sha256"], native_profile=adapter["native_profile"],
        actual_gpu_observations=observations)
    try:
        code = runner.evaluate(plan)
        terminal["native_function_return"] = code
        return code
    except BaseException as error:
        terminal["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result = output / "RESULT.json"
        terminal["native_result_sha256"] = sha(result) if result.is_file() else None
        source_error = None
        try:
            terminal["native_sources_unchanged"] = _verify_original(protocol)[1] == native_sources
        except (ValueError, KeyError, OSError) as error:
            source_error = error
            terminal["native_sources_unchanged"] = False
            terminal["native_source_validation_error"] = str(error)
        write_new(_sidecar(output, "SITE_ADAPTER_RESULT.json"), terminal)
        if source_error:
            raise ValueError("native benchmark sources failed post-execution validation; retained adapter-result receipt: " + str(source_error)) from source_error


def _output_argument(arguments):
    value = None
    for index, item in enumerate(arguments):
        if item.startswith("--output="):
            value = item.split("=", 1)[1]
        elif item == "--output" and index + 1 < len(arguments):
            value = arguments[index + 1]
    return Path(value).expanduser().resolve() if value else None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        selection = argparse.ArgumentParser(add_help=False)
        selection.add_argument("--protocol", choices=sorted(PROFILES), default="one_epoch")
        selected, _ = selection.parse_known_args(argv)
        if "--help" in argv or "-h" in argv:
            native, _ = _load_original(selected.protocol)
            help_parser = native.parser()
            help_parser.prog = "jev benchmark"
            help_parser.description = __doc__
            help_parser.epilog = ("Native arguments are taken from the verified preserved controller. "
                                  "Site values supply role and asset defaults; explicit native arguments take precedence. "
                                  "See docs/EVALUATION.md for prerequisites and CPU preparation.")
            help_parser.add_argument("--site", required=True, help="jev.benchmark-site.v1 resource/role configuration")
            help_parser.add_argument("--protocol", choices=sorted(PROFILES), default="one_epoch",
                help="preserved native controller profile; e11 requires an explicit two_epoch_F2/F3 training contract (default: one_epoch)")
            help_parser.print_help()
            return 0
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--site", required=True, help="jev.benchmark-site.v1 resource/role configuration")
        parser.add_argument("--protocol", choices=sorted(PROFILES), default="one_epoch")
        parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--_site-sha", help=argparse.SUPPRESS)
        args, forwarded = parser.parse_known_args(argv)
        site_path, site = _site(args.site)
        if args._worker:
            return _worker(site_path, site, forwarded, args._site_sha, args.protocol)
        site_digest = sha(site_path)
        command = [site["controller_python"], "-P", "-m", "jev.benchmarks", "--_worker", "--site", str(site_path),
                   "--_site-sha", site_digest, "--protocol", args.protocol, *forwarded]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        output = _output_argument(forwarded)
        if "--prepare-only" in forwarded or output is None:
            return subprocess.run(command, env=env, check=False).returncode
        stdout, stderr = _sidecar(output, "controller.stdout.log"), _sidecar(output, "controller.stderr.log")
        output.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("xb") as out, stderr.open("xb") as err:
            completed = subprocess.run(command, env=env, check=False, stdout=out, stderr=err)
        native_result = output / "RESULT.json"
        adapter_result = _sidecar(output, "SITE_ADAPTER_RESULT.json")
        observed_site_digest = sha(site_path) if site_path.is_file() else None
        receipt = dict(schema="jev.benchmark-controller-exit.v1", argv=command, native_exit=completed.returncode,
            site_sha256=site_digest, site_sha256_after=observed_site_digest,
            site_file_unchanged=observed_site_digest == site_digest,
            adapter_source_sha256=sha(__file__), protocol=args.protocol,
            original_runner_sha256=_profile(args.protocol)["runner_sha256"],
            native_result_sha256=sha(native_result) if native_result.is_file() else None,
            adapter_result_sha256=sha(adapter_result) if adapter_result.is_file() else None,
            stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
        write_new(_sidecar(output, "CONTROLLER_EXIT.json"), receipt)
        print(json.dumps(dict(native_exit=completed.returncode, output=str(output),
            controller_exit=str(_sidecar(output, "CONTROLLER_EXIT.json")), stderr=str(stderr)), indent=2))
        return completed.returncode
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("JEV benchmark: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
