"""Run one fixed native MC task in its own process/output, using the new head client."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from .bundle_contract import digest, read_config

MC_TASKS = ('mmlu_fewshot','mmlu','hellaswag','gsm8k_mc10','arc_easy','gsm8k_mc4',
            'winogrande','arc_challenge','chess','gpqa_fewshot','gpqa')


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--endpoints', required=True, nargs='+')
    p.add_argument('--head-config', required=True, type=Path)
    p.add_argument('--task', required=True, choices=MC_TASKS)
    p.add_argument('--rw-root', required=True, type=Path)
    p.add_argument('--runtime-cache', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-legacy-h0', action='store_true')
    a = p.parse_args()
    sys.dont_write_bytecode = True
    os.environ['RUNWEAVER_OPENJEV_CACHE'] = str(a.runtime_cache.resolve())
    rw = a.rw_root.resolve()
    sys.path.insert(0, str(rw/'src'))
    from agentic_evaluate.benchmarks import openjev_adapter as adapter_module
    from agentic_evaluate.benchmarks import openjev_bridge as bridge
    from agentic_evaluate.benchmarks import openjev_pointer as pointer
    from agentic_evaluate.benchmarks.native_task_runtime import locked
    from .openjev_head import ClassificationNLI
    for module in (adapter_module, bridge, pointer):
        if not Path(module.__file__).resolve().is_relative_to(rw/'src'):
            raise ValueError('Unexpected RunWeaver source location: '+str(module.__file__))
    revision_path = rw/'scripts/openjev/data_revisions.json'
    revisions = json.loads(revision_path.read_text())
    # MC11 excludes auxiliary-candidate "gsm8k" and vision/game tasks.
    native = SimpleNamespace(environment='static', user=None, seed=None,
        task_ids=[a.task], output_dir=str(a.output.resolve()), max_concurrency=len(a.endpoints),
        agent=SimpleNamespace(model=a.model, endpoint=a.endpoints[0], api_key_env=None, args={}),
        options=dict(backend='pointer', pointer_endpoints=a.endpoints,
                     data_revisions=revisions, request_timeout_s=300))
    adapter = adapter_module.OpenJevAdapter()
    config = adapter.effective_config(native)
    if a.task in ('gpqa','gpqa_fewshot') and not config.get('gpqa_path'):
        raise ValueError('The existing frozen GPQA asset is absent from runtime-cache')
    expected = read_config(a.head_config)
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config.update(head_config=expected, head_kind=expected['kind'],
                  allow_legacy_h0=a.allow_legacy_h0,
                  transport='native_decision_http',
                  protocol_status='classification_head_nli_adapter')
    # Shared installation lock matches the native adapter; no install/prepare
    # is performed and this worker never enters or changes an external queue.
    with locked(adapter.cache/'install.lock', exclusive=False), locked(output/'.run.lock', nonblocking=True):
        remote = ClassificationNLI(config, output)
        config['probability_source'] = remote.probability_source
        if remote.probability_source=='pointer_head': config['protocol_status']='legacy_h0_pointer_head_nli_adapter'
        dump(output/'effective_config.json', config)
        dump(output/'source_binding.json', dict(
            native_modules={Path(m.__file__).name:digest(m.__file__) for m in (adapter_module,bridge,pointer)},
            overlay_modules={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            data_revisions_sha256=digest(revision_path), native_source=adapter_module.verify_source(),
            executable=sys.executable, original_request_generator=str(rw/'scripts/openjev/make_handoff_requests.py')))
        original = pointer.PointerNLI
        old_argv = sys.argv
        def isolated_client(bound_config, bound_output):
            if bound_config != config or Path(bound_output).resolve() != output:
                raise ValueError('Unexpected native bridge configuration')
            return remote
        try:
            # Process-local class substitution only. The complete actual bridge
            # main keeps its native argv, loaders, dataset tracking and scorer.
            pointer.PointerNLI = isolated_client
            sys.argv = ['openjev_bridge', str(output)]
            bridge.main()
            raw = json.loads((output/'native_result.json').read_text())
            observed = raw.get(a.model, {})
            complete = set(observed)=={a.task} and observed[a.task].get('n', observed[a.task].get('n_questions',0)) > 0
            if not complete: raise ValueError('Native MC task did not produce a complete positive-size result')
            dump(output/'summary.json', dict(complete=True, task_ids=[a.task],
                probability_source=remote.probability_source, head_config=expected,
                population=config['population'], metrics=adapter_module.flatten_metrics(raw),
                native_result_sha256=digest(output/'native_result.json'), calls=remote.calls,
                started_at=started, finished_at=time.time()))
        except BaseException as exc:
            if not (output/'summary.json').exists():
                dump(output/'summary.json', dict(complete=False, task_ids=[a.task],
                    probability_source=remote.probability_source, head_config=expected,
                    error=repr(exc), started_at=started, finished_at=time.time(), calls=remote.calls))
            raise
        finally:
            pointer.PointerNLI = original
            sys.argv = old_argv


if __name__ == '__main__': main()
