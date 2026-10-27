"""CPU tests of the explicit last_span readout (E8) and of the unchanged end / mean_span readouts.

Run in the training environment with CUDA_VISIBLE_DEVICES='' and PYTHONPATH at this source tree. The frozen
E1/E2 tree is compared as well when it is available (JEV_E458_BASIS_SOURCE, or ../../../campaign/source).
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from kev import heads
from kev import model as model_module
from runtime import bundle_contract

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_hybrid import SPECIAL_IDS, FakeTokenizer, decision_model  # noqa: E402

READOUTS = ("end", "mean_span", "last_span")
# Options with no, one and several content tokens (the stand-in tokenizer gives one token per byte).
RECORD = {"state": "the state text", "questions": [
    {"instr": "first question", "options": ["", "a", "abc", "two words"], "label": 2},
    {"instr": "second", "options": ["yes", ""], "label": 0}]}
OTHER = {"state": "a longer state, so that a batch needs padding", "questions": [
    {"instr": "only one", "options": ["x", "", "longer option text"], "label": 1}]}


def head(readout):
    return {"version": 1, "kind": "pointer", "dim": 256, "readout": readout, "rng_anchor_dim": 256}


def original_heads():
    """Reference head implementation for compatibility tests."""
    path = Path(__file__).with_name("heads_original_reference.py")
    spec = importlib.util.spec_from_file_location("unchanged_head1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def basis_model():
    """kev.model of the frozen E1/E2 tree, imported under another package name; None when the tree is not here."""
    root = Path(os.environ.get("JEV_E458_BASIS_SOURCE") or Path(__file__).resolve().parents[3] / "campaign/source")
    directory = root / "kev"
    if not (directory / "model.py").is_file():
        return None
    if "basis_kev" not in sys.modules:
        spec = importlib.util.spec_from_file_location("basis_kev", directory / "__init__.py",
                                                      submodule_search_locations=[str(directory)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["basis_kev"] = package
        spec.loader.exec_module(package)
    return importlib.import_module("basis_kev.model")


def expected_rows(enc, question, readout):
    """Positions the rule reads for each option, written out independently of the implementation."""
    rows = []
    for option, end_token in enumerate(enc["opt_idx"][question]):
        if readout == "end":
            rows.append([end_token])
            continue
        start, end = enc["opt_spans"][question][option]
        rows.append([end_token] if start == end else list(range(start, end)) if readout == "mean_span" else [end - 1])
    return rows


class EncodingTest(unittest.TestCase):
    def test_tokens_positions_and_masks_do_not_depend_on_the_readout(self):
        tok = FakeTokenizer()
        for record in (RECORD, OTHER):
            for isolation in (False, True):
                with self.subTest(isolation=isolation):
                    encodings = {readout: model_module.encode(tok, record, strict=True, option_isolation=isolation,
                                                              readout=readout) for readout in READOUTS}
                    end = encodings["end"]
                    self.assertNotIn("opt_spans", end)
                    for readout in ("mean_span", "last_span"):
                        other = dict(encodings[readout])
                        other.pop("opt_spans")
                        # ids, seg, pos, opt, decide_idx, opt_idx, labels, option_isolation, state_truncated
                        self.assertEqual(other, end)
                    self.assertEqual(encodings["last_span"]["opt_spans"], encodings["mean_span"]["opt_spans"])
                    masks = [model_module.branch_mask_batch([enc["seg"]], "cpu", opts=[enc["opt"]] if isolation else None)
                             for enc in encodings.values()]
                    self.assertTrue(torch.equal(masks[0], masks[1]) and torch.equal(masks[0], masks[2]))

    def test_limits_and_truncation_do_not_depend_on_the_readout(self):
        tok = FakeTokenizer()
        long_state = {"state": "s" * 40, "questions": RECORD["questions"]}
        outcomes = {}
        for readout in READOUTS:
            row = []
            for kwargs in (dict(max_state=8, max_branch=1024, strict=True), dict(max_state=384, max_branch=20, strict=True)):
                with self.assertRaises(ValueError) as caught:
                    model_module.encode(tok, long_state, readout=readout, **kwargs)
                row.append(str(caught.exception))
            truncated = model_module.encode(tok, long_state, max_state=8, readout=readout)
            row.append((truncated["ids"], truncated["seg"], truncated["pos"], truncated["state_truncated"]))
            outcomes[readout] = row
        self.assertEqual(outcomes["last_span"], outcomes["end"])
        self.assertEqual(outcomes["mean_span"], outcomes["end"])
        self.assertTrue(outcomes["end"][2][3])

    def test_frozen_tree_encodes_the_old_readouts_identically_and_rejects_the_new_one(self):
        basis = basis_model()
        if basis is None:
            self.skipTest("frozen E1/E2 source tree not available")
        tok = FakeTokenizer()
        for record in (RECORD, OTHER):
            for isolation in (False, True):
                for readout in ("end", "mean_span"):
                    kwargs = dict(strict=True, option_isolation=isolation, readout=readout)
                    self.assertEqual(model_module.encode(tok, record, **kwargs), basis.encode(tok, record, **kwargs))
        with self.assertRaises(ValueError):
            basis.encode(tok, RECORD, readout="last_span")


class ReadoutRuleTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.enc = model_module.encode(self.tok, RECORD, strict=True, readout="last_span")
        # Row i of the hidden states holds i, so a read value names the position it came from.
        self.h = torch.arange(len(self.enc["ids"]), dtype=torch.float32)[:, None].repeat(1, 3)

    def read(self, question, readout, h=None):
        spans = None if readout == "end" else self.enc["opt_spans"][question]
        return heads.readout_features(self.h if h is None else h, self.enc["decide_idx"][question],
                                      self.enc["opt_idx"][question], readout=readout, opt_spans=spans)

    def test_last_span_reads_the_last_content_token_and_falls_back_to_the_end_delimiter(self):
        ids = self.enc["ids"]
        open_id, close_id, decide_id = (SPECIAL_IDS[token] for token in model_module.SPECIAL[2:])
        lengths = []
        for question, row in enumerate(RECORD["questions"]):
            query, options = self.read(question, "last_span")
            self.assertEqual(ids[self.enc["decide_idx"][question]], decide_id)
            self.assertTrue(torch.equal(query, self.h[self.enc["decide_idx"][question]]))    # the decide token
            for option, text in enumerate(row["options"]):
                start, end = self.enc["opt_spans"][question][option]
                tokens = self.tok(text).input_ids
                self.assertEqual(ids[start:end], tokens)                     # the half-open span is the content
                self.assertEqual((ids[start - 1], ids[end]), (open_id, close_id))
                self.assertEqual(end, self.enc["opt_idx"][question][option])
                self.assertEqual(options[option, 0].item(), end - 1 if tokens else end)
                lengths.append(len(tokens))
        self.assertEqual(sorted(set(lengths)), [0, 1, 3, 9])                 # empty, one token, several tokens

    def test_the_three_readouts_differ_only_where_the_rule_says(self):
        for question, row in enumerate(RECORD["questions"]):
            end, mean, last = (self.read(question, readout)[1] for readout in READOUTS)
            for option, text in enumerate(row["options"]):
                tokens = len(self.tok(text).input_ids)
                if tokens == 0:
                    self.assertTrue(torch.equal(last[option], end[option]) and torch.equal(mean[option], end[option]))
                elif tokens == 1:
                    self.assertTrue(torch.equal(last[option], mean[option]))
                    self.assertFalse(torch.equal(last[option], end[option]))
                else:
                    self.assertFalse(torch.equal(last[option], mean[option]))
                    self.assertEqual(last[option, 0].item(), end[option, 0].item() - 1)
            for readout in READOUTS:
                self.assertEqual(self.read(question, readout)[0][0].item(), self.enc["decide_idx"][question])

    def test_gradient_reaches_exactly_the_positions_the_rule_reads(self):
        for readout in READOUTS:
            for question in range(len(RECORD["questions"])):
                h = torch.randn(len(self.enc["ids"]), 3, requires_grad=True)
                self.read(question, readout, h)[1].sum().backward()
                touched = sorted(set(h.grad.abs().sum(1).nonzero().flatten().tolist()))
                wanted = sorted({position for row in expected_rows(self.enc, question, readout) for position in row})
                self.assertEqual(touched, wanted, (readout, question))

    def test_end_and_mean_span_are_the_unchanged_original(self):
        original = original_heads()
        torch.manual_seed(3)
        h = torch.randn(len(self.enc["ids"]), 16)
        for question in range(len(RECORD["questions"])):
            for readout in ("end", "mean_span"):
                spans = None if readout == "end" else self.enc["opt_spans"][question]
                args = (h, self.enc["decide_idx"][question], self.enc["opt_idx"][question])
                new = heads.readout_features(*args, readout=readout, opt_spans=spans)
                old = original.readout_features(*args, readout=readout, opt_spans=spans)
                self.assertTrue(torch.equal(new[0], old[0]) and torch.equal(new[1], old[1]))
        args = (h, self.enc["decide_idx"][0], self.enc["opt_idx"][0])
        for kwargs in (dict(readout="mean_span", opt_spans=None), dict(readout="mean_span", opt_spans=[(1, 2)]),
                       dict(readout="mean_span", opt_spans=[(0, 1)] * 4), dict(readout="first_span", opt_spans=None)):
            messages = []
            for module in (heads, original):
                with self.assertRaises(ValueError) as caught:
                    module.readout_features(*args, **kwargs)
                messages.append(str(caught.exception))
            self.assertEqual(messages[0], messages[1])
        with self.assertRaises(ValueError):
            original.readout_features(*args, readout="last_span", opt_spans=self.enc["opt_spans"][0])

    def test_last_span_needs_the_encoder_spans(self):
        args = (self.h, self.enc["decide_idx"][0], self.enc["opt_idx"][0])
        spans = self.enc["opt_spans"][0]
        shifted = [(start, end - 1) if end > start else (start, end) for start, end in spans]
        for bad in (None, spans[:-1], [[1.0, 2.0]] * 4, shifted, [(end + 1, end) for _, end in spans]):
            with self.assertRaises(ValueError):
                heads.readout_features(*args, readout="last_span", opt_spans=bad)


class MetadataTest(unittest.TestCase):
    def test_head_metadata_names_last_span_explicitly(self):
        self.assertEqual(heads.normalize_head_config(head("last_span")), head("last_span"))
        for bad in ({**head("last_span"), "dim": 64}, {**head("last_span"), "kind": "tanh_pointer"},
                    {k: v for k, v in head("last_span").items() if k != "rng_anchor_dim"},
                    {**head("last_span"), "readout": "first_span"}, {**head("last_span"), "version": 2}):
            with self.assertRaises(ValueError):
                heads.normalize_head_config(bad)
            with self.assertRaises(ValueError):
                bundle_contract.config_identity(bad)
        for readout in READOUTS:          # exporter, loader and artifact contract agree on every readout
            config = head(readout)
            self.assertEqual(bundle_contract.config_identity(heads.normalize_head_config(config)), config)
            self.assertEqual(heads.normalize_head_config(bundle_contract.config_identity(config)), config)
        # A reader that predates the readout refuses it; it cannot read a last_span model as something else.
        with self.assertRaises(ValueError):
            original_heads().normalize_head_config(head("last_span"))

    def test_last_span_head_is_the_same_pointer_with_the_same_initialization(self):
        made = {}
        for readout in READOUTS:
            torch.manual_seed(44)
            made[readout] = (heads.make_head(16, head_config=head(readout)), torch.get_rng_state().clone())
        for readout in ("mean_span", "last_span"):
            self.assertIs(type(made[readout][0]), heads.PointerHead)
            self.assertTrue(torch.equal(made[readout][1], made["end"][1]))
            for name, value in made["end"][0].state_dict().items():
                self.assertTrue(torch.equal(value, made[readout][0].state_dict()[name]))
            self.assertEqual(made[readout][0].head_config["readout"], readout)

    def bundle(self, root, manifest_config, file_config):
        members = ("classification_head.safetensors", "code/kev/heads.py", "code/kev/__init__.py", "code/kev/model.py",
                   "code/kev/api.py", "code/runtime/__init__.py", "code/runtime/bundle_contract.py",
                   "code/runtime/load_bundle.py")
        for name in members:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        (root / "head_config.json").write_text(json.dumps(file_config))
        files = {path.relative_to(root).as_posix(): bundle_contract.digest(path)
                 for path in sorted(root.rglob("*")) if path.is_file()}
        (root / "model_manifest.json").write_text(json.dumps(dict(
            schema=bundle_contract.SCHEMA, probability_source="classification_head", head_config=manifest_config,
            head_kind=manifest_config["kind"], backbone_dtype="bfloat16", head_dtype="float32", merge_dtype="float32",
            temperature=1.0, lora_scale=1.0, option_isolation=False, hybrid=True, files=files)))
        (root / "validation_report.json").write_text(json.dumps(dict(
            status="PASS", head_config=manifest_config, requests=1, max_probability_delta=0.0, argmax_equal=True,
            manifest_sha256=bundle_contract.digest(root / "model_manifest.json"), source_unchanged=True,
            checkpoint_unchanged=True)))

    def test_exported_bundle_metadata_carries_the_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            good, mixed = Path(directory) / "good", Path(directory) / "mixed"
            self.bundle(good, head("last_span"), head("last_span"))
            manifest = bundle_contract.validate_manifest(good, head("last_span"))
            self.assertEqual(manifest["head_config"]["readout"], "last_span")
            self.assertEqual(bundle_contract.read_config(good / "head_config.json")["readout"], "last_span")
            for other in ("end", "mean_span"):           # serving with another readout is refused, not guessed
                with self.assertRaises(ValueError):
                    bundle_contract.validate_manifest(good, head(other))
            self.bundle(mixed, head("last_span"), head("mean_span"))
            with self.assertRaises(ValueError):
                bundle_contract.validate_manifest(mixed)
        reply = dict(model="m", probability_source="classification_head", head_config=head("last_span"), head_kind="pointer")
        bundle_contract.validate_response(reply, "m", head("last_span"))
        with self.assertRaises(ValueError):
            bundle_contract.validate_response(reply, "m", head("mean_span"))


class ForwardPathTest(unittest.TestCase):
    def test_training_and_inference_forwards_read_by_the_same_rule(self):
        tok = FakeTokenizer()
        packed = {}
        for readout in READOUTS:
            model = decision_model(head(readout), hybrid=False)
            enc = model.encode(tok, RECORD, strict=True)
            self.assertEqual("opt_spans" in enc, readout != "end")
            with torch.no_grad():
                hidden = model.hidden(enc)
                wanted = [model.head(hidden[enc["decide_idx"][q]],
                                     torch.stack([hidden[row].mean(0) for row in expected_rows(enc, q, readout)]))
                          for q in range(len(RECORD["questions"]))]
                training = model.forward_batch([enc])[0]          # kev.train
                inference = model.forward(enc)                    # evaluation and serving
                probabilities = model.probs(enc)
            for q, logits in enumerate(wanted):
                self.assertTrue(torch.allclose(training[q], logits, atol=1e-6), readout)
                self.assertTrue(torch.equal(training[q], inference[q]))
                self.assertTrue(torch.allclose(probabilities[q], torch.softmax(logits, -1), atol=1e-6))
            packed[readout] = (model.state_dict(), training)
        self.assertFalse(torch.allclose(packed["last_span"][1][0], packed["mean_span"][1][0], atol=1e-4))
        self.assertFalse(torch.allclose(packed["last_span"][1][0], packed["end"][1][0], atol=1e-4))
        for readout in READOUTS:                                   # the hybrid row form, as on NeoHorse
            model = decision_model(head(readout), hybrid=True)
            model.load_state_dict(packed[readout][0])
            encodings = [model.encode(tok, RECORD, strict=True), model.encode(tok, OTHER, strict=True)]
            with torch.no_grad():
                batch = model.forward_batch(encodings)             # padded rows of two records
                single = [model.forward(enc) for enc in encodings]
                probabilities = model.probs(encodings[0])
            for q, logits in enumerate(packed[readout][1]):
                self.assertTrue(torch.allclose(batch[0][q], logits, atol=1e-4), readout)
                self.assertTrue(torch.allclose(probabilities[q], torch.softmax(logits, -1), atol=1e-4))
            for record in range(2):
                for a, b in zip(batch[record], single[record]):
                    self.assertTrue(torch.allclose(a, b, atol=1e-4), readout)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
