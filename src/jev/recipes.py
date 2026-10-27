"""Build commands from recorded experiments; only deployment locations are rebound."""
from pathlib import Path
import re
from .config import load_site, read, root, safe_id, sha, tree_path

STUDIES = {"e1e2": "e12", "e458": "e458", "e11": "e11"}


def options(argv):
    try:
        module_index = argv.index("-m") + 1
        trainer_module(argv[module_index])
    except (IndexError, ValueError) as error:
        raise ValueError("expected a native kev trainer module after -m") from error
    start = module_index + 1
    values = argv[start:]
    result, index = {}, 0
    while index < len(values):
        key = values[index]
        if not isinstance(key, str) or not key.startswith("--") or key == "--" or "=" in key:
            raise ValueError("expected distinct trainer flags followed by values or standalone switches")
        if key in result:
            raise ValueError("duplicate trainer argument: " + key)
        if index + 1 == len(values) or str(values[index + 1]).startswith("--"):
            result[key] = True
            index += 1
        else:
            result[key] = values[index + 1]
            index += 2
    return result


def trainer_module(value):
    if not isinstance(value, str) or not re.fullmatch(r"kev\.[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("trainer_module must name one native kev.<module>")
    return value


def command(python, args, module="kev.train"):
    argv = [python, "-P", "-m", trainer_module(module)]
    for key, value in args.items():
        if not isinstance(key, str) or not key.startswith("--") or key == "--" or "=" in key:
            raise ValueError("trainer argument names must be complete --flags")
        if isinstance(value, bool):
            if value:
                argv.append(key)
        else:
            argv.extend((key, str(value)))
    return argv


def output_for(site, study, run_id):
    return str(Path(site["work_root"]) / "runs" / study / safe_id(run_id) / "output")


def paper_plan(study, site, run_id=None, arm=None, fold="F2", seed=44):
    plan_path = root() / "provenance/plans" / (study + ".json")
    from .validation import verify_protocol
    verify_protocol(plan_path)
    plan = read(plan_path)
    matches = [r for r in plan["runs"] if r["run_id"] == run_id] if run_id else [
        r for r in plan["runs"] if r["arm"] == arm and r["fold"] == fold and r["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("choose one recorded run with --run-id or --arm/--fold/--seed; see jev list --study " + study)
    run = matches[0]
    args = options(run["argv"])
    original = dict(args)
    args["--base"] = site["base_model"]
    data_rel = site.get("data_files", {}).get(run["fold"], run["fold"] + ".train.jsonl")
    args["--data"] = str(Path(site["data_root"]) / data_rel)
    args["--out"] = output_for(site, study, run["run_id"])
    references = []
    for flag in ("--e458_reference_run", "--e11_reference_run"):
        if flag in args:
            ref = run["reference"]
            rid = ref["run_id"]
            ref_study = "e458" if rid.startswith("E458_") else "e1e2"
            args[flag] = site.get("reference_outputs", {}).get(rid, output_for(site, ref_study, rid))
            references.append({"run_id": rid, "path": args[flag]})
    changed = {k for k in args if args[k] != original[k]}
    allowed = {"--base", "--data", "--out", "--e458_reference_run", "--e11_reference_run"}
    if changed - allowed:
        raise ValueError("scientific arguments changed during site binding")
    return dict(schema="jev.execution.v1", study=study, run_id=run["run_id"], tree=STUDIES[study],
                argv=command(site["python"]["training"], args), output=args["--out"],
                input_pins={args["--data"]: run["data_sha256"],
                            str(Path(site["base_model"]) / "config.json"): plan["base_config_sha256"]},
                references=references, expected=run["expected"], head_config=run["head_config"],
                archived_plan_sha256=sha(plan_path), changed_location_arguments=sorted(changed),
                scientific_overrides=False, environment={})


def custom_plan(path, site):
    recipe = read(path)
    if recipe.get("schema") != "jev.recipe.v1":
        raise ValueError("recipe.schema must be jev.recipe.v1")
    rid = safe_id(recipe["id"])
    module = trainer_module(recipe.get("trainer_module", "kev.train"))
    source = None
    if recipe.get("source_root"):
        source = (Path(path).resolve().parent / recipe["source_root"]).resolve()
        if not (source / "kev/train.py").is_file():
            raise ValueError("custom source_root must contain kev/train.py")
        source_tree = source
    else:
        source_tree = tree_path(recipe["tree"])
    module_path = source_tree / "kev" / (module.split(".")[1] + ".py")
    if (not module_path.is_file() or module_path.is_symlink()
            or not module_path.resolve().is_relative_to(source_tree.resolve())):
        raise ValueError("selected training source lacks trainer_module: " + module)
    output = output_for(site, "custom", rid)
    fill = {"{base_model}": site["base_model"], "{data_root}": site["data_root"], "{output}": output}
    def bind(value):
        if isinstance(value, bool):
            return value
        value = str(value)
        for token, replacement in fill.items():
            value = value.replace(token, replacement)
        return value
    args = {"--" + k.removeprefix("--"): bind(v) for k, v in recipe["trainer_args"].items()}
    if args.get("--out") != output:
        raise ValueError("custom recipe --out must use {output}")
    references = []
    for reference in recipe.get("references", []):
        if (not isinstance(reference, dict) or not isinstance(reference.get("run_id"), str)
                or not isinstance(reference.get("path"), str) or not reference["path"]):
            raise ValueError("active recipe references need run_id and path; put explanatory lineage in reference_notes")
        references.append(dict(reference, path=bind(reference["path"])))
    return dict(schema="jev.execution.v1", study="custom", run_id=rid, tree=recipe.get("tree", "custom"),
                argv=command(site["python"]["training"], args, module), output=output,
                input_pins={bind(k): v for k, v in recipe.get("input_pins", {}).items()},
                references=references, expected=recipe.get("expected", {}),
                scientific_overrides=True, environment={}, source_root=str(source) if source else None,
                trainer_module=module,
                recipe_sha256=sha(path), custom_source=source is not None)


def build(args):
    site = load_site(args.site)
    if args.recipe:
        plan = custom_plan(args.recipe, site)
    else:
        plan = paper_plan(args.study, site, args.run_id, args.arm, args.fold, args.seed)
    if not plan.get("source_root"):
        plan["source_root"] = str(tree_path(plan["tree"]))
    if not plan["scientific_overrides"]:
        from .validation import base_asset_pins
        plan["input_pins"].update(base_asset_pins(site))
    plan["site"] = site
    return plan
