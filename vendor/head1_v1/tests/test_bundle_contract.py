"""Stdlib-only artifact and consumer identity tests; no Torch, data, network or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # head1/, which holds runtime/
from runtime.bundle_contract import (SCHEMA, config_identity, digest,
    native_prediction, validate_manifest, validate_probabilities, validate_response)


H0 = dict(version=1, kind='pointer', dim=256, readout='end', rng_anchor_dim=256)
TANH = dict(H0, kind='tanh_pointer')
LEGACY = {k:v for k,v in H0.items() if k!='rng_anchor_dim'}


class BundleContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = ['classification_head.safetensors','code/kev/__init__.py',
                 'code/kev/heads.py','code/kev/model.py','code/kev/api.py',
                 'code/runtime/__init__.py','code/runtime/bundle_contract.py',
                 'code/runtime/load_bundle.py','backbone/model.safetensors',
                 'tokenizer/tokenizer.json']
        for name in names:
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture data, never executed\n')
        (self.root/'head_config.json').write_text(json.dumps(H0))
        self.manifest = dict(schema=SCHEMA, name='fixture', probability_source='classification_head',
            head_kind='pointer', head_config=H0, backbone_dtype='bfloat16', head_dtype='float32',
            merge_dtype='float32', temperature=1, lora_scale=1, hybrid=True, option_isolation=False,
            files={p.relative_to(self.root).as_posix():digest(p) for p in self.root.rglob('*') if p.is_file()})
        self.write_manifest()

    def write_manifest(self):
        (self.root/'model_manifest.json').write_text(json.dumps(self.manifest))
        self.report = dict(status='PASS', head_config=H0, requests=3, max_probability_delta=1e-6,
                           argmax_equal=True, source_unchanged=True, checkpoint_unchanged=True,
                           manifest_sha256=digest(self.root/'model_manifest.json'))
        self.write_report()

    def write_report(self):
        (self.root/'validation_report.json').write_text(json.dumps(self.report))

    def test_six_variants_and_explicit_legacy(self):
        variants = [H0, dict(H0, dim=64), TANH,
                    dict(H0, kind='cosine_pointer', cosine_scale=16),
                    dict(H0, kind='residual_pair_mlp', mlp_hidden=128),
                    dict(H0, readout='mean_span'), LEGACY]
        for config in variants:
            with self.subTest(config=config): self.assertEqual(config_identity(config), config)

    def test_bad_variant_metadata(self):
        bad = [dict(H0, dim=True), dict(H0, version=True), dict(H0, surprise=1),
               dict(H0, rng_anchor_dim=64), dict(H0, kind='cosine_pointer', cosine_scale=15),
               dict(H0, kind='residual_pair_mlp', mlp_hidden=64),
               dict(LEGACY, kind='tanh_pointer'), dict(H0, kind='tanh_pointer', readout='mean_span')]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError): config_identity(config)

    def test_same_shaped_wrong_head_is_rejected_before_loading(self):
        self.assertEqual(validate_manifest(self.root, H0)['head_config'], H0)
        # Pointer/tanh share q/k shapes; state_dict strict=True alone cannot catch this.
        with self.assertRaisesRegex(ValueError, 'requested variant'): validate_manifest(self.root, TANH)

    def test_changed_weight_is_rejected(self):
        (self.root/'classification_head.safetensors').write_bytes(b'different head')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): validate_manifest(self.root, H0)

    def test_unbound_source_is_rejected(self):
        (self.root/'code/kev/unbound.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError, 'unbound'): validate_manifest(self.root, H0)

    def test_source_symlink_escape_is_rejected(self):
        source = self.root/'code/kev/model.py'
        source.unlink(); source.symlink_to('/etc/hosts')
        with self.assertRaises(ValueError): validate_manifest(self.root, H0)

    def test_manifest_and_config_file_must_agree(self):
        (self.root/'head_config.json').write_text(json.dumps(TANH))
        self.manifest['files']['head_config.json'] = digest(self.root/'head_config.json')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'): validate_manifest(self.root, H0)

    def test_only_reload_success_is_not_finished_export(self):
        self.report.pop('source_unchanged'); self.write_report()
        with self.assertRaises(ValueError): validate_manifest(self.root, H0)

    def test_old_report_cannot_approve_changed_manifest(self):
        self.manifest['name'] = 'changed'
        (self.root/'model_manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError): validate_manifest(self.root, H0)

    def test_distribution_metadata_and_legacy_scope(self):
        result = dict(model='fixture', probability_source='classification_head',
                      head_kind='pointer', head_config=H0)
        validate_response(result, 'fixture', H0)
        with self.assertRaises(ValueError): validate_response(result, 'fixture', TANH)
        with self.assertRaises(ValueError): validate_response(result, 'other', H0)
        legacy = dict(model='fixture', probability_source='pointer_head')
        with self.assertRaises(ValueError): validate_response(legacy, 'fixture', LEGACY)
        validate_response(legacy, 'fixture', LEGACY, allow_legacy_h0=True)
        with self.assertRaises(ValueError): validate_response(legacy, 'fixture', H0, allow_legacy_h0=True)

    def test_probabilities_cover_choice_noul_score_without_rounding(self):
        request = dict(questions={'c':dict(type='choice', criteria={'a':'A','b':'B'}),
                                  'n':dict(type='noul'), 's':dict(type='score', criteria=['low','high'])})
        result = dict(probabilities={'c':{'a':.123456789,'b':.876543211},
                                     'n':{'false':.2,'true':.8},'s':{'0':.25,'1':.75}})
        self.assertEqual(validate_probabilities(result, request), result['probabilities'])
        bad = copy.deepcopy(result); del bad['probabilities']['s']
        with self.assertRaises(ValueError): validate_probabilities(bad, request)
        bad = copy.deepcopy(result); bad['probabilities']['c']['a'] = float('nan')
        with self.assertRaises(ValueError): validate_probabilities(bad, request)
        bad = copy.deepcopy(result); bad['probabilities']['n'] = {'no':.2,'yes':.8}
        with self.assertRaises(ValueError): validate_probabilities(bad, request)

    def test_official_parser_envelope_and_noul_mapping(self):
        request = dict(questions={'decision':dict(type='noul')})
        result = dict(model='fixture', probability_source='classification_head',
                      head_kind='pointer', head_config=H0,
                      probabilities={'decision':{'false':.123456789,'true':.876543211}})
        encoded = json.loads(native_prediction(result, request, 'fixture', H0, 'jevbench'))
        self.assertEqual(set(encoded), {'probabilities'})
        self.assertEqual(encoded['probabilities'], {'no':.123456789,'yes':.876543211})
        kev = json.loads(native_prediction(result, request, 'fixture', H0, 'kev'))
        self.assertEqual(kev, {'probabilities':result['probabilities']})


if __name__ == '__main__': unittest.main()
