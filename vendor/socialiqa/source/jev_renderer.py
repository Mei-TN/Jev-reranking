"""Pinned loader for the actual JEV request renderer and runtime encoder source.

Nothing here builds a model, reads a tokenizer file or touches a GPU. The
training and serving modules import torch, transformers and datasets at module
level, so only the code that turns a request into model-visible text is taken
from the pinned files:

  kev/api.py    executed whole (pydantic only): render, option_text, to_record
  kev/data.py   `materialize` only, bound to the api objects above
  kev/model.py  `encode`, `user_tokens` and their constants only

Every extraction is closed: each global the extracted source refers to must be
supplied explicitly, defined by the extraction itself, or be a builtin.
"""
import ast
import builtins
import hashlib
import importlib.util
from pathlib import Path
import re
import symtable
import sys

# Same identities as RENDERER_SOURCE_IDENTITIES.json and campaign/PLAN.json source_sha256.
PINS = {
    'kev/api.py': 'c9eebbdb6c625563d33f030299ce0fbe8b50493dd1e92592df9514c30f5720af',
    'kev/data.py': 'e5081aa01debda6652ec631aa89f5a22e11a235c5d6db821628d5dfccf5d8614',
    'kev/model.py': '902427ae67c6cc6d07da4bf0143643cfd1c5d80e58a0f2b3dc1fbb765be0dfc7'}
ENCODER_NAMES = ('SPECIAL', 'MAX_STATE', 'MAX_BRANCH', '_SPECIAL_RE', 'user_tokens',
                 'OPT_NONE', 'OPT_DECIDE', 'encode')
# Pin request validation to the renderer version.
PYDANTIC_VERSION = '2.13.5'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned(code_tree, relative):
    path = Path(code_tree) / relative
    actual = sha256_file(path)
    if actual != PINS[relative]:
        raise ValueError(f'renderer source differs from pin: {relative} {actual}')
    return path


def _targets(node):
    for target in node.targets:
        for leaf in (target.elts if isinstance(target, ast.Tuple) else [target]):
            if not isinstance(leaf, ast.Name):
                raise ValueError('unsupported assignment target in pinned source')
            yield leaf.id


def _global_references(table):
    names = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()} \
        if table.get_type() == 'function' else {s.get_name() for s in table.get_symbols()
                                                if s.is_referenced() and not s.is_assigned()}
    for child in table.get_children():
        names |= _global_references(child)
    return names


def extract(path, names, supplied):
    """Compile exactly the named top-level definitions, in source order, from the original text."""
    source = Path(path).read_text(encoding='utf-8')
    wanted, chosen, defined = set(names), [], set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found = {node.name}
        elif isinstance(node, ast.Assign) and set(_targets(node)) & wanted:
            found = set(_targets(node))
            if not found <= wanted:
                raise ValueError('pinned assignment binds an unrequested name')
        else:
            continue
        if found & defined:
            raise ValueError('pinned source defines a requested name twice')
        defined |= found
        chosen.append(ast.get_source_segment(source, node))
    if defined != wanted:
        raise ValueError(f'pinned source lacks {sorted(wanted - defined)}')
    text = '\n\n'.join(chosen) + '\n'
    free = _global_references(symtable.symtable(text, str(path), 'exec')) - defined - set(dir(builtins))
    if free - set(supplied):
        raise ValueError(f'extracted source needs unsupplied globals {sorted(free - set(supplied))}')
    scope = {name: supplied[name] for name in free}
    exec(compile(text, str(path), 'exec'), scope)
    return {name: scope[name] for name in names}, hashlib.sha256(text.encode()).hexdigest()


def load_api(code_tree):
    path = pinned(code_tree, 'kev/api.py')
    import pydantic
    if pydantic.VERSION != PYDANTIC_VERSION:
        raise ValueError(f'pydantic {pydantic.VERSION} is not the training environment version {PYDANTIC_VERSION}')
    name = 'e10_pinned_kev_api_' + PINS['kev/api.py'][:16]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_renderer(code_tree):
    """-> (api module, materialize, identities). The trainer's own clean-record route."""
    api = load_api(code_tree)
    functions, text_sha = extract(pinned(code_tree, 'kev/data.py'), ('materialize',),
                                  {'SystemOneRequest': api.SystemOneRequest, 'to_record': api.to_record})
    identities = dict(source_sha256={k: PINS[k] for k in ('kev/api.py', 'kev/data.py')},
                      extracted_materialize_sha256=text_sha, pydantic=PYDANTIC_VERSION,
                      route='kev.data.materialize -> SystemOneRequest.model_validate -> kev.api.to_record')
    return api, functions['materialize'], identities


def load_encoder(code_tree):
    """Torch-free runtime packer checked against the imported encoder during preflight."""
    functions, text_sha = extract(pinned(code_tree, 'kev/model.py'), ENCODER_NAMES, {'re': re})
    return functions, dict(source_sha256=PINS['kev/model.py'], extracted_encoder_sha256=text_sha)
