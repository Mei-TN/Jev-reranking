"""Build a common auxiliary evaluation mask from overlap scans.

Verify source and registry digests and reconstruct the scan inputs before
building one mask shared across models. The full 1,954-question source remains
the primary evaluation population; this script leaves source data intact.

  detected    confirmed overlap, excluded from the auxiliary subset
  unresolved  candidates unresolved within the review budget, also excluded
  yellow      reported candidates retained in the subset
"""
import argparse
import bisect
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings stay importable under python -I
import audit_scanner
import jev_audit_adapter
from jev_audit_adapter import ACTUAL, SCHEMA, canonical, digest
from jev_renderer import sha256_file
from socialiqa_requests import SOURCE_FREEZE_SHA256

HERE = Path(__file__).resolve().parent
# The unmodified skill script and the two wrappers; a mask is never built with other scanner code present.
TOOL_PINS = {
    'skill_decontaminate.py': 'b37decf7110e19503de10c5c3dab6b73bc575c08b7470fae73bd95879b54ec33',
    'audit_scanner.py': 'f789b3e039ce6a1cde16b4cc4005083782f6219be32e979f7f24c62b81638201',
    'audit_scanner_recall.py': '1030ab922ed8761e30b78a3a843bd034348b05869095ce11f31483f095483fb3'}
WRAPPER = {'conservative': 'audit_scanner.py', 'recall': 'audit_scanner_recall.py'}
SUFFIX = {'conservative': '+jev-e10-nfc-case-preserving-v1', 'recall': '+jev-e10-skill-nfkc-casefold-recall-only-v1'}
POLICY = dict(shingle_width=13, cjk_shingle_width=15, red_threshold=0.8, orange_threshold=0.2)
ZONE_RANK = {'yellow': 1, 'orange': 2, 'red': 3}
REVIEW_KEYS = {'stable_key', 'benchmark', 'eval_split', 'item_id', 'decision', 'reviewer', 'reason'}
CLEAR, DETECTED, UNRESOLVED = ('no_training_overlap_detected_in_audited_scope', 'detected_training_overlap',
                               'unresolved_training_overlap_candidate')
UNCOVERED = (
    'lineage: the official SocialIQA archive has no upstream item identifier, so explicit-ID matching cannot fire; '
    'this is limited coverage, not evidence of no lineage overlap',
    'wrong answer alternatives are not a separate lexical channel (registry: premise as question, gold answer text)',
    'approximate near-duplicate (MinHash-LSH), option-permutation structure keys and semantic similarity were not run',
    'thresholds are the skill priors; no labelled calibration exists for this corpus, so no precision or recall is claimed',
    'historical development or evaluation exposure of SocialIQA is a separate question and is not measured here')


def load_skill():
    spec = importlib.util.spec_from_file_location('skill_decontaminate_for_mask', HERE / 'skill_decontaminate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with Path(path).open('rb') as stream:
        return [json.loads(line) for line in stream]


def pinned_tool(channel, *argv):
    """Run one of the two pinned wrappers in a fresh interpreter; they expose only build-registry and scan."""
    result = subprocess.run([sys.executable, '-I', '-B', str(HERE / WRAPPER[channel]), *map(str, argv)],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'{channel} wrapper ended with native exit {result.returncode}: {result.stderr[-2000:]}')


def check_frozen(frozen):
    """The handed-over freeze by its own pinned digest, then every file it records."""
    if sha256_file(Path(frozen) / 'SOURCE_FREEZE.json') != SOURCE_FREEZE_SHA256:
        raise ValueError('SOURCE_FREEZE.json is not the frozen SocialIQA contract')
    files = json.loads((Path(frozen) / 'SOURCE_FREEZE.json').read_text(encoding='utf-8'))['files']
    if not {'items.jsonl', 'eval-assets.json'} <= set(files):
        raise ValueError('SOURCE_FREEZE.json does not pin the item list and the registry configuration')
    for name, expected in files.items():
        if sha256_file(Path(frozen) / name) != expected:
            raise ValueError('frozen SocialIQA asset changed: ' + name)


def reproduce_registry(channel, frozen, registry, scratch):
    """The registry must be byte for byte what the pinned wrapper builds from the frozen eval-assets.json,
    which fixes its field mapping (premise as question, gold answer text) and its normalized contents."""
    rebuilt, registry = scratch / (channel + '_registry'), Path(registry)
    pinned_tool(channel, 'build-registry', '--config', Path(frozen) / 'eval-assets.json', '--out', rebuilt)
    if {p.name for p in registry.iterdir()} != {'MANIFEST.json', 'items.jsonl'} or any(
            (rebuilt / name).read_bytes() != (registry / name).read_bytes() for name in ('MANIFEST.json', 'items.jsonl')):
        raise ValueError(channel + ' registry is not what the pinned wrapper builds from the frozen eval-assets.json')


def reproduce_scan(channel, release, registry, scan, summary, workers, scratch):
    """Re-execute the pinned scanner now. A stored scan is accepted only as a record of what this
    execution also finds: the same match bytes and the same summary apart from the worker count."""
    rerun = scratch / (channel + '_scan')
    pinned_tool(channel, 'scan', '--release', Path(release).resolve(), '--registry', Path(registry).resolve(),
                '--out', rerun, '--workers', workers, '--progress-every', 0, '--shingle-width', POLICY['shingle_width'],
                '--cjk-shingle-width', POLICY['cjk_shingle_width'], '--red-threshold', POLICY['red_threshold'],
                '--orange-threshold', POLICY['orange_threshold'])
    again = json.loads((rerun / 'summary.json').read_text(encoding='utf-8'))
    claimed = summary.get('runtime')
    if not isinstance(claimed, dict) or set(claimed) != {'workers'} or type(claimed['workers']) is not int:
        raise ValueError(channel + ' scan: unexpected runtime record')
    if sha256_file(rerun / 'matches.jsonl') != sha256_file(Path(scan) / 'matches.jsonl') or \
            {k: v for k, v in again.items() if k != 'runtime'} != {k: v for k, v in summary.items() if k != 'runtime'}:
        raise ValueError(channel + ' scan: the pinned scanner does not reproduce this evidence')
    # The stored worker count is the only field not reproduced; it is the stored summary's own claim.
    return dict(matches_sha256=sha256_file(rerun / 'matches.jsonl'),
                stored_summary_claimed_workers=claimed['workers'], re_execution_workers=again['runtime']['workers'])


def load_scan(skill, scan, registry, release, channel, provenance, items):
    """A complete, self-consistent scan of exactly this release and registry, at the frozen policy values."""
    scan, registry = Path(scan), Path(registry)
    complete = json.loads((scan / 'SCAN_COMPLETE').read_text(encoding='utf-8'))
    summary = json.loads((scan / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((registry / 'MANIFEST.json').read_text(encoding='utf-8'))
    matches = read_jsonl(scan / 'matches.jsonl')
    unique = defaultdict(set)
    for match in matches:
        unique[match['zone']].add(match['stable_key'])
    counts = summary['counts']
    checks = (
        (complete == dict(matches_sha256=sha256_file(scan / 'matches.jsonl'),
                          summary_sha256=sha256_file(scan / 'summary.json')), 'scan files differ from SCAN_COMPLETE'),
        (summary['schema'] == skill.SCAN_SCHEMA and summary['matches_file'] == skill.file_facts(scan / 'matches.jsonl'),
         'matches.jsonl is not the file this scan summarized'),
        (counts['match_records_by_zone'] == dict(sorted(Counter(m['zone'] for m in matches).items()))
         and counts['by_rule'] == dict(sorted(Counter(m['rule'] for m in matches).items()))
         and counts['unique_rows_by_zone'] == {zone: len(keys) for zone, keys in sorted(unique.items())},
         'match records disagree with the counts in the scan summary'),
        (summary['source'] == skill.source_inventory(Path(release).resolve()), 'scan was not run on this release'),
        (summary['registry_manifest_sha256'] == sha256_file(registry / 'MANIFEST.json'), 'scan used another registry'),
        (manifest['items_file']['sha256'] == sha256_file(registry / 'items.jsonl') == summary['registry_items_sha256'],
         'registry items differ from their manifest'),
        (summary['tool_version'].endswith(SUFFIX[channel]) and manifest['tool_version'] == summary['tool_version'],
         'scan or registry was not produced by the ' + channel + ' wrapper'),
        (all(summary['parameters'][key] == value for key, value in POLICY.items()), 'scan thresholds differ from policy'),
        (counts['rows_scanned'] == counts['unique_stable_keys_scanned'] == len(provenance)
         and counts['duplicate_stable_key_rows'] == 0, 'scan did not cover every physical row exactly once'))
    for passed, reason in checks:
        if not passed:
            raise ValueError(f'{channel} scan: {reason}')
    for match in matches:
        row, item = provenance.get(match['stable_key']), items.get(match['item_id'])
        if row is None or item is None or match['zone'] not in ZONE_RANK or \
                (match['split'], match['canonical_row_sha256']) != (row['split'], row['canonical_row_sha256']) or \
                (match['benchmark'], match['eval_split']) != (item['benchmark'], item['split']):
            raise ValueError(f'{channel} scan: a match record does not belong to this release and registry')
        if match['rule'] == 'lineage_cluster':
            raise ValueError(f'{channel} scan: lineage propagation is not part of this audit')
    return matches, summary, manifest


def ngrams(tokens, width):
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)} if 0 < width <= len(tokens) else set()


def longest_common_run(needle, haystack):
    """Longest contiguous token run of `needle` that also occurs in `haystack` (monotone in the run length)."""
    low, high = 0, len(needle)
    while low < high:
        width = (low + high + 1) // 2
        if ngrams(needle, width) & ngrams(haystack, width):
            low = width
        else:
            high = width - 1
    return low


def text_confirmation(token_re, normalize, item, user, assistant, match):
    """Re-derive the evidence from the token sequences themselves; a digest hit alone is not evidence."""
    fields = item['fields']
    eval_q = token_re.findall('\n'.join(fields[k] for k in ('question', 'choices') if fields.get(k)))
    eval_a = token_re.findall('\n'.join(fields[k] for k in ('answer', 'rationale') if fields.get(k)))
    train_q, train_a = token_re.findall(normalize(user)), token_re.findall(normalize(assistant))
    width = min(POLICY['shingle_width'], len(eval_q))
    shared = dict(question=len(ngrams(eval_q, width) & ngrams(train_q, width)),
                  question_answer=len(ngrams(eval_q + eval_a, POLICY['shingle_width'])
                                      & ngrams(train_q + train_a, POLICY['shingle_width'])),
                  answer=len(ngrams(eval_a, POLICY['shingle_width']) & ngrams(train_a, POLICY['shingle_width'])))
    channel = {'question_containment': 'question', 'question_answer_containment': 'question_answer',
               'answer_or_test_candidate': 'answer'}.get(match['rule'])
    needed = match['scores']['shared_nonfrequent_shingles']
    available = shared[channel] if channel else max(shared.values())
    run = longest_common_run(eval_q, train_q)
    return dict(eval_question_tokens=len(eval_q), eval_question_shingle_width=width,
                eval_question_shingles=len(ngrams(eval_q, width)), shared_token_shingles=shared,
                longest_common_question_token_run=run, full_question_token_sequence_present=run == len(eval_q) > 0,
                gold_answer_token_sequence_present_in_supervised_answer=bool(eval_a) and bool(
                    ngrams(eval_a, len(eval_a)) & ngrams(train_a, len(eval_a))),
                text_confirms_scanner_evidence=needed > 0 and available >= needed)


def exact_hits(needles, texts):
    """Whole-string occurrences of each needle in each text, at token boundaries. needles: {key: string}."""
    offsets, total = [], 0
    for text in texts:
        offsets.append(total)
        total += len(text) + 1
    haystack = '\x00'.join(texts)
    hits = defaultdict(set)
    for key, needle in needles.items():
        if not needle or '\x00' in needle:
            raise ValueError('unusable exact-string needle')
        position = haystack.find(needle)
        while position >= 0:
            end = position + len(needle)
            left = not needle[0].isalnum() or position == 0 or not haystack[position - 1].isalnum()
            right = not needle[-1].isalnum() or end == len(haystack) or not haystack[end].isalnum()
            if left and right:
                hits[key].add(bisect.bisect_right(offsets, position) - 1)
            position = haystack.find(needle, position + 1)
    return hits


def review_budget(logical_training_samples, floor):
    return min(100000, max(floor, int(1e-4 * logical_training_samples)))


def load_reviews(path, candidates, group_of, items):
    """Skill review-file schema, bound to this benchmark and split. A decision on one physical row
    covers its identically rendered copies; every record needs a named reviewer and a reason."""
    reviews = {}
    for line, review in enumerate(read_jsonl(path), 1):
        where = f'{path}:{line}: '
        if REVIEW_KEYS - set(review) or set(review) - REVIEW_KEYS - {'logical_row_sha256'} or \
                review['decision'] not in ('remove', 'keep', 'uncertain') or \
                not all(isinstance(review[k], str) and review[k].strip() for k in ('reviewer', 'reason')):
            raise ValueError(where + 'invalid or unattributed review record')
        item = items.get(review['item_id'])
        if review['stable_key'] not in group_of or item is None or \
                (review['benchmark'], review['eval_split']) != (item['benchmark'], item['split']) or \
                review.get('logical_row_sha256', group_of[review['stable_key']]) != group_of[review['stable_key']]:
            raise ValueError(where + 'review names an unknown training row or another benchmark/split')
        key = (group_of[review['stable_key']], review['item_id'])
        if key not in candidates or candidates[key]['detected_by_text'] or candidates[key]['yellow_only']:
            raise ValueError(where + 'review does not refer to an open candidate')
        if key in reviews:
            raise ValueError(where + 'duplicate or conflicting review')
        reviews[key] = review
    return reviews


def build(args):
    for name, pin in TOOL_PINS.items():
        if sha256_file(HERE / name) != pin:
            raise ValueError('scanner code differs from its pinned identity: ' + name)
    scratch = Path(tempfile.mkdtemp(prefix='e10_mask_'))
    try:
        return assemble(args, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def assemble(args, scratch):
    skill = load_skill()
    check_frozen(args.frozen)
    release = Path(args.release)
    # Scanning a release proves nothing about where its rows came from; the adapter's verifier does.
    verification = jev_audit_adapter.verify(release, args.code_tree)
    if verification['status'] != 'PASS':
        raise ValueError('release does not verify against the pinned sources: ' + repr(verification['failures'][:3]))
    release_manifest = json.loads((release / 'MANIFEST.json').read_text(encoding='utf-8'))
    if release_manifest['schema'] != SCHEMA:
        raise ValueError('not a v2 audit release')
    frozen_items = read_jsonl(Path(args.frozen) / 'items.jsonl')
    order = [item['item_id'] for item in frozen_items]
    normalizers = {'conservative': audit_scanner.normalize, 'recall': skill.normalize_conservative}
    scans = {'conservative': (args.scan, args.registry), 'recall': (args.recall_scan, args.recall_registry)}

    rows = read_jsonl(release / 'ledgers/provenance.jsonl')
    provenance = {p['stable_key']: p for p in rows}
    group_of = {key: p['canonical_row_sha256'] for key, p in provenance.items()}
    members = defaultdict(list)
    for p in rows:
        members[p['canonical_row_sha256']].append(p)
    if len(provenance) != len(rows) or len(rows) != release_manifest['physical_rows'] or \
            any('lineage_group_id' in p for p in rows):
        raise ValueError('provenance is not one unique stable key per physical row without lineage groups')
    in_actual = {sha for sha, copies in members.items() if any(p['source_role'] == ACTUAL for p in copies)}
    logical = dict(
        basis='identical canonical message row (same model-visible text and supervised option text)',
        physical_rows=len(rows), logical_rows_all_scanned_sources=len(members),
        logical_rows_in_actual_training_inputs=len(in_actual),
        explicit_item_ids_in_actual_training_inputs=len({p['source_metadata'].get('item_id') for p in rows
                                                         if p['source_role'] == ACTUAL}))

    loaded, registries, reproduced = {}, {}, {}
    for channel, (scan, registry) in scans.items():
        reproduce_registry(channel, args.frozen, registry, scratch)
        registries[channel] = {item['item_id']: item for item in read_jsonl(Path(registry) / 'items.jsonl')}
        if list(registries[channel]) != sorted(order) or len(order) != len(set(order)):
            raise ValueError(channel + ' registry is not the frozen SocialIQA item list')
        loaded[channel] = load_scan(skill, scan, registry, release, channel, provenance, registries[channel])
        reproduced[channel] = reproduce_scan(channel, release, registry, scan, loaded[channel][1], args.workers, scratch)

    # Model-visible text of the first physical copy of every logical row.
    wanted, texts = defaultdict(dict), {}
    for sha, copies in members.items():
        wanted[copies[0]['split']][copies[0]['original_line']] = sha
    for split, lines in wanted.items():
        with (release / 'messages' / (split + '.jsonl')).open('rb') as stream:
            for line, raw in enumerate(stream, 1):
                if line in lines:
                    if digest(raw.rstrip(b'\n')) != lines[line]:
                        raise ValueError('message row does not match its provenance hash')
                    message = json.loads(raw)['messages']
                    texts[lines[line]] = (message[0]['content'], message[1]['content'])
    if set(texts) != set(members):
        raise ValueError('could not read every logical row')

    candidates = {}

    def candidate(sha, item_id):
        return candidates.setdefault((sha, item_id), dict(
            logical_row_sha256=sha, item_id=item_id, evidence=[], detected_by_text=False, yellow_only=True))

    scan_counts = {}
    for channel, (matches, summary, _) in loaded.items():
        strongest = {}
        for match in matches:
            # Identically rendered copies carry the same direct evidence; an exact-duplicate record adds none.
            if match['rule'] == 'exact_duplicate_cluster':
                continue
            key = (group_of[match['stable_key']], match['item_id'])
            if key not in strongest or ZONE_RANK[match['zone']] > ZONE_RANK[strongest[key]['zone']]:
                strongest[key] = match
        if any((group_of[m['stable_key']], m['item_id']) not in strongest for m in matches):
            raise ValueError(channel + ' scan: an inherited record has no direct evidence on the same text')
        for (sha, item_id), match in sorted(strongest.items()):
            confirmation = text_confirmation(skill.TOKEN_RE, normalizers[channel], registries[channel][item_id],
                                             *texts[sha], match)
            entry = candidate(sha, item_id)
            entry['evidence'].append(dict(channel=channel + '_scan', zone=match['zone'], rule=match['rule'],
                                          scores=match['scores'], confirmation=confirmation))
            if match['zone'] != 'yellow':
                entry['yellow_only'] = False
            if channel == 'conservative' and match['zone'] == 'red' and confirmation['text_confirms_scanner_evidence']:
                entry['detected_by_text'] = True
        scan_counts[channel] = dict(scanner_counts=summary['counts'], re_executed_at_mask_build=reproduced[channel],
                                    logical_row_item_pairs_by_zone=dict(sorted(
                                        Counter(m['zone'] for m in strongest.values()).items())))

    # Supplementary exact-string channel: full premise and context alone, in both normalizations.
    shas = sorted(texts)
    exact_counts = {}
    for name, normalize in normalizers.items():
        needles = {}
        for item in frozen_items:
            needles[(item['item_id'], 'premise')] = normalize(item['premise'])
            needles[(item['item_id'], 'context')] = normalize(item['context'])
        hits = exact_hits(needles, [normalize(texts[sha][0]) for sha in shas])
        for (item_id, kind), found in sorted(hits.items()):
            for index in sorted(found):
                entry = candidate(shas[index], item_id)
                entry['evidence'].append(dict(channel='exact_string', normalization=name, matched=kind,
                                              needle_tokens=len(skill.TOKEN_RE.findall(needles[(item_id, kind)]))))
                entry['yellow_only'] = False
        exact_counts[name] = dict(needles=len(needles), needles_with_a_hit=len(hits),
                                  shortest_context_tokens=min(len(skill.TOKEN_RE.findall(v))
                                                              for (_, kind), v in needles.items() if kind == 'context'))

    open_keys = {key for key, entry in candidates.items() if not entry['detected_by_text'] and not entry['yellow_only']}
    open_rows = {sha for sha, _ in open_keys}
    budgets = {basis: review_budget(logical[basis], args.min_review_budget) for basis in (
        'logical_rows_in_actual_training_inputs', 'logical_rows_all_scanned_sources', 'physical_rows')}
    applied = budgets['logical_rows_in_actual_training_inputs']
    exceeded = len(open_rows) > applied
    reviews = load_reviews(args.reviews, candidates, group_of, registries['conservative']) if args.reviews else {}
    if reviews and exceeded:
        raise ValueError('candidates exceed the review budget; a partial review cannot resolve them')
    if reviews and set(reviews) != open_keys:
        raise ValueError('a bounded review must decide every open candidate pair, not a chosen subset')

    by_item = defaultdict(list)
    evidence_rows = []
    for key in sorted(candidates):
        entry = candidates[key]
        review = reviews.get(key)
        if entry['yellow_only']:
            disposition = 'yellow_reported_not_excluded'
        elif entry['detected_by_text']:
            disposition = 'detected_strong_lexical_or_lineage'
        elif review and review['decision'] == 'remove':
            disposition = 'detected_by_bounded_review'
        elif review and review['decision'] == 'keep':
            disposition = 'reviewed_not_the_same_item'
        else:
            disposition = 'unresolved'
        user, assistant = texts[key[0]]
        record = dict(entry, disposition=disposition, review=review,
                      in_actual_training_inputs=key[0] in in_actual,
                      physical_rows=[dict(stable_key=p['stable_key'], split=p['split'], line=p['original_line'],
                                          source_role=p['source_role'],
                                          explicit_item_id=p['source_metadata'].get('item_id'),
                                          explicit_group_id=p['source_metadata'].get('group_id'))
                                     for p in members[key[0]]],
                      train_user_excerpt=user[:1200], train_supervised_answer_excerpt=assistant[:400])
        evidence_rows.append(record)
        by_item[key[1]].append(record)

    mask, tally, roles = [], Counter(), Counter()
    for item in frozen_items:
        records = by_item.get(item['item_id'], [])
        dispositions = Counter(r['disposition'] for r in records)
        if dispositions['detected_strong_lexical_or_lineage'] or dispositions['detected_by_bounded_review']:
            status = DETECTED
        elif dispositions['unresolved']:
            status = UNRESOLVED
        else:
            status = CLEAR
        tally[status] += 1
        if status != CLEAR:
            deciding = [r for r in records if r['disposition'].startswith('detected' if status == DETECTED else 'unres')]
            roles[(status, 'touches_actual_training_inputs' if any(r['in_actual_training_inputs'] for r in deciding)
                   else 'retained_derivation_inputs_only')] += 1
        tally['items_with_yellow_evidence'] += any(e.get('zone') == 'yellow' for r in records for e in r['evidence'])
        mask.append(dict(item_id=item['item_id'], source_ordinal=item['source_ordinal'], status=status,
                         in_auxiliary_subset=status == CLEAR, candidate_pairs=dict(sorted(dispositions.items()))))

    output = Path(args.out).resolve()
    incomplete = output.with_name(output.name + '.incomplete')
    if output.exists() or incomplete.exists():
        raise ValueError('output or its .incomplete sibling already exists; use a new versioned path')
    incomplete.mkdir(parents=True)
    for name, records in (('common_eval_mask.jsonl', mask), ('candidate_evidence.jsonl', evidence_rows),
                          ('review_queue.jsonl', [dict(logical_row_sha256=sha, item_id=item_id,
                                                       stable_key=members[sha][0]['stable_key'])
                                                  for sha, item_id in sorted(open_keys - set(reviews))])):
        with (incomplete / name).open('xb') as stream:
            for record in records:
                stream.write(canonical(record) + b'\n')
    summary = dict(
        schema='jev-e10-common-eval-mask-v1', benchmark='socialiqa', split='dev',
        subset_name='subset with no training overlap detected within the audited scope',
        not_a_contamination_free_claim=True, training_or_evaluation_data_changed=False,
        main_denominator=len(frozen_items), main_denominator_unchanged_by_this_mask=True,
        one_mask_for_all_models=True, mask_basis='union of every scanned source, actual and retained',
        counts=dict(auxiliary_subset=tally[CLEAR], excluded_detected=tally[DETECTED],
                    excluded_unresolved=tally[UNRESOLVED], items_with_yellow_evidence=tally['items_with_yellow_evidence']),
        excluded_by_source_reach={status + ':' + reach: n for (status, reach), n in sorted(roles.items())},
        candidate_pairs_by_disposition=dict(sorted(Counter(r['disposition'] for r in evidence_rows).items())),
        logical_accounting=logical,
        review=dict(
            unit='distinct logical training rows with an open orange, recall-channel or exact-string candidate',
            open_logical_rows=len(open_rows), open_candidate_pairs=len(open_keys),
            formula='min(100000, max(floor, int(1e-4 * N_logical_training_samples)))', floor=args.min_review_budget,
            floor_approval=args.budget_approval, budget_by_basis=budgets,
            applied_basis='logical_rows_in_actual_training_inputs', applied_budget=applied,
            budget_exceeded=exceeded, reviews_applied=len(reviews),
            status='BUDGET_EXCEEDED_CANDIDATES_LEFT_UNRESOLVED' if exceeded else
                   'NO_OPEN_CANDIDATES' if not open_keys else
                   'ALL_OPEN_CANDIDATES_REVIEWED' if reviews else 'WITHIN_BUDGET_NOT_YET_REVIEWED'),
        channels=dict(
            conservative_scan=dict(decides=True, normalization='NFC + whitespace folding; case, digits, operators and '
                                   'negation preserved', **scan_counts['conservative']),
            recall_scan=dict(decides=False, normalization='skill NFKC + casefold + HTML unescape + quote folding',
                             **scan_counts['recall']),
            exact_string=dict(decides=False, needles='full premise and context alone, at token boundaries',
                              by_normalization=exact_counts),
            lineage='explicit training item identifiers only; evaluation identifiers are local ordinals; grouped-split '
                    'units are recorded with each candidate but never propagate a decision'),
        policy=POLICY, uncovered=list(UNCOVERED) + release_manifest['uncovered'],
        identities=dict(
            release_manifest_sha256=sha256_file(release / 'MANIFEST.json'),
            release_verified_against_pinned_sources={key: verification[key] for key in (
                'status', 'physical_rows', 'source_files_reproduced_from_ledger',
                'every_stored_row_and_the_manifest_rebuilt_byte_identically')},
            source_freeze_sha256=sha256_file(Path(args.frozen) / 'SOURCE_FREEZE.json'),
            frozen_items_sha256=sha256_file(Path(args.frozen) / 'items.jsonl'),
            eval_assets_sha256=sha256_file(Path(args.frozen) / 'eval-assets.json'),
            scans={channel: dict(scan_complete_sha256=sha256_file(Path(scan) / 'SCAN_COMPLETE'),
                                 registry_manifest_sha256=sha256_file(Path(registry) / 'MANIFEST.json'),
                                 registry_rebuilt_byte_identically=True,
                                 tool_version=loaded[channel][1]['tool_version'])
                   for channel, (scan, registry) in scans.items()},
            reviews_sha256=sha256_file(args.reviews) if args.reviews else None,
            scanner_code_pinned=TOOL_PINS,
            mask_code={name: sha256_file(HERE / name) for name in ('build_mask.py', 'jev_audit_adapter.py', 'jev_renderer.py')}),
        files={name: dict(sha256=sha256_file(incomplete / name), rows=len(read_jsonl(incomplete / name)))
               for name in ('common_eval_mask.jsonl', 'candidate_evidence.jsonl', 'review_queue.jsonl')})
    (incomplete / 'MASK_SUMMARY.json').write_bytes(canonical(summary) + b'\n')
    incomplete.replace(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('release', 'code-tree', 'frozen', 'registry', 'scan', 'recall-registry', 'recall-scan', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--workers', type=int, default=1, help='workers for re-executing each scan (1 on Windows)')
    parser.add_argument('--min-review-budget', type=int, default=1)
    parser.add_argument('--budget-approval', help='who approved a floor other than the skill default of 1')
    args = parser.parse_args()
    if not 1 <= args.min_review_budget <= 100000 or (args.min_review_budget != 1) != bool(args.budget_approval):
        parser.error('a review floor other than 1 needs --budget-approval, and only then')
    summary = build(args)
    print(json.dumps(dict(counts=summary['counts'], review=summary['review'],
                          candidate_pairs_by_disposition=summary['candidate_pairs_by_disposition'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
