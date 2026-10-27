"""Use pinned skill scanner with the study's case-preserving normalization.

Only build-registry and scan are exposed. No apply, deletion, or mutation mode.
"""
import importlib.util
from pathlib import Path
import sys
import unicodedata


def normalize(value):
    return ' '.join(unicodedata.normalize('NFC', value).split())


if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] not in ('build-registry','scan'):
        raise SystemExit('only immutable registry/scan operations allowed')
    source=Path(__file__).with_name('skill_decontaminate.py')
    spec=importlib.util.spec_from_file_location('skill_decontaminate',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.normalize_conservative=normalize
    module.TOOL_VERSION += '+jev-e10-nfc-case-preserving-v1'
    raise SystemExit(module.main())
