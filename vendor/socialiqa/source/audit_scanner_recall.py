"""Recall-only channel: the pinned skill scanner with its original NFKC/casefold normalizer.

The study decides on the case-preserving scan (audit_scanner.py). Anything this
channel finds is a candidate only; build_mask.py never records it as detected.
Only build-registry and scan are exposed. No apply, deletion, or mutation mode.
"""
import importlib.util
from pathlib import Path
import sys


if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] not in ('build-registry','scan'):
        raise SystemExit('only immutable registry/scan operations allowed')
    source=Path(__file__).with_name('skill_decontaminate.py')
    spec=importlib.util.spec_from_file_location('skill_decontaminate',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.TOOL_VERSION += '+jev-e10-skill-nfkc-casefold-recall-only-v1'
    raise SystemExit(module.main())
