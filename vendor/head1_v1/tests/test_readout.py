"""Synthetic token/hidden-state tests exercise offsets, not backbone numerics."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trees" / "head1_v1"))
import torch
from torch import nn
from kev.heads import normalize_head_config, readout_features
from kev.model import DecisionModel, SPECIAL, encode, rows_of


class Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1000 + SPECIAL.index(token)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[int(word) for word in text.split()])


class Cache:
    def __init__(self, **kwargs):
        self.reorders, self.crops = [], []

    def reorder_cache(self, indices):
        self.reorders.append(indices.tolist())

    def crop(self, value):
        self.crops.append(value)


class TokenFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace()
        self.calls = []

    def forward(self, input_ids, position_ids, **kwargs):
        self.calls.append((input_ids.clone(), position_ids.clone(), kwargs))
        hidden = torch.stack((input_ids.float(), position_ids.float()), -1)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=kwargs.get("past_key_values"))


class ReadoutProbe(nn.Module):
    def forward(self, query, options):
        # Both the selected query position and option content affect scores.
        return options[:, 0] / 1000 + options[:, 1] * query[1] / 100


def toy_model(readout="mean_span", hybrid=False):
    model = DecisionModel.__new__(DecisionModel)
    nn.Module.__init__(model)
    cfg = {"version": 1, "kind": "pointer", "dim": 256, "readout": readout, "rng_anchor_dim": 256}
    model.head_config = normalize_head_config(cfg)
    model.lm, model.head = TokenFeatures(), ReadoutProbe()
    model.pad_id, model.device, model.hybrid, model.option_isolation = 0, "cpu", hybrid, False
    model.eval()
    return model


RECORD = {"state": "1 2 3", "questions": [
    {"instr": "4 5", "options": ["10 30", "", "40"], "label": 0},
    {"instr": "6", "options": ["50", "60 80 100"], "label": 1},
]}


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.enc = encode(self.tok, RECORD, readout="mean_span")

    def test_encoding_preserves_legacy_tokens_and_explicit_content_boundaries(self):
        legacy = encode(self.tok, RECORD)
        self.assertNotIn("opt_spans", legacy)
        self.assertEqual(legacy, {k: v for k, v in self.enc.items() if k != "opt_spans"})
        expected = [[[10, 30], [], [40]], [[50], [60, 80, 100]]]
        for q, ends, spans in zip(expected, self.enc["opt_idx"], self.enc["opt_spans"]):
            for tokens, end_token, (start, end) in zip(q, ends, spans):
                self.assertEqual(self.enc["ids"][start:end], tokens)
                self.assertEqual(end, end_token)
                self.assertEqual(self.enc["ids"][start - 1], self.tok.convert_tokens_to_ids(SPECIAL[2]))
                self.assertEqual(self.enc["ids"][end], self.tok.convert_tokens_to_ids(SPECIAL[3]))

    def test_mean_excludes_delimiters_padding_and_empty_falls_back(self):
        h = torch.tensor(self.enc["ids"], dtype=torch.float32)[:, None].requires_grad_()
        padded = torch.cat((h, torch.full((17, 1), 99999.0)))
        d, ends, spans = self.enc["decide_idx"][0], self.enc["opt_idx"][0], self.enc["opt_spans"][0]
        query, options = readout_features(padded, d, ends, readout="mean_span", opt_spans=spans)
        self.assertTrue(torch.equal(query, h[d]))
        self.assertEqual(options.flatten().tolist(), [20.0, 1003.0, 40.0])
        options.sum().backward()
        for start, end in spans:
            if start < end:
                self.assertEqual(float(h.grad[start - 1]), 0)
                self.assertEqual(float(h.grad[end]), 0)
                self.assertTrue(torch.equal(h.grad[start:end], torch.full_like(h.grad[start:end], 1 / (end - start))))
            else:
                self.assertEqual(float(h.grad[end]), 1)
        _, legacy = readout_features(h, d, ends)
        self.assertEqual(legacy.flatten().tolist(), [1003.0] * 3)

    def test_rows_translate_spans_and_decide_for_each_question(self):
        state, state_pos, rows = rows_of(self.enc)
        self.assertEqual(state, [1000, 1, 2, 3])
        branch_start = len(state)
        for i, row in enumerate(rows):
            self.assertEqual(row["decide"] + branch_start, self.enc["decide_idx"][i])
            self.assertEqual([v + branch_start for v in row["opts"]], self.enc["opt_idx"][i])
            self.assertEqual([(a + branch_start, b + branch_start) for a, b in row["opt_spans"]], self.enc["opt_spans"][i])
            self.assertEqual(state_pos + row["pos"], list(range(len(state) + len(row["ids"]))))
            branch_start += len(row["ids"])
        self.assertNotIn("opt_spans", rows_of(encode(self.tok, RECORD))[2][0])

    def assert_nested_close(self, got, expected):
        self.assertEqual(len(got), len(expected))
        for actual, wanted in zip(got, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_packed_hybrid_rows_and_padding_agree_on_synthetic_features(self):
        for readout in ("end", "mean_span"):
            packed, rows = toy_model(readout), toy_model(readout, hybrid=True)
            short = {"state": "7", "questions": [{"instr": "8", "options": ["9", "11 12"], "label": 1}]}
            encodings = [packed.encode(self.tok, RECORD), packed.encode(self.tok, short)]
            expected = packed.forward_batch(encodings)
            actual = rows.forward_batch(encodings)
            for a, b in zip(actual, expected):
                self.assert_nested_close(a, b)
            ids, _, kw = rows.lm.calls[-1]
            self.assertTrue(bool((kw["attention_mask"] == 0).any()))
            self.assertTrue(bool((ids[kw["attention_mask"] == 0] == rows.pad_id).all()))

    def test_prefix_routes_preserve_content_and_query_offsets(self):
        for hybrid in (False, True):
            for readout in ("end", "mean_span"):
                with self.subTest(hybrid=hybrid, readout=readout):
                    model = toy_model(readout, hybrid)
                    enc = model.encode(self.tok, RECORD)
                    expected = model.probs(enc)
                    state, positions, _ = rows_of(enc)
                    hidden_state = torch.stack((torch.tensor(state).float(), torch.tensor(positions).float()), -1)
                    cache = Cache()
                    actual = model.probs_with_prefix(enc, (len(state), cache, hidden_state))
                    self.assert_nested_close(actual, expected)
                    if hybrid:
                        self.assertEqual(cache.reorders, [])  # original cache is not consumed
                        self.assertEqual(cache.crops, [])
                    else:
                        self.assertEqual(cache.crops, [-(len(enc["ids"]) - len(state))])
                    with patch("transformers.DynamicCache", Cache):
                        actual, prefix = model.probs_and_prefix(enc)
                        self.assert_nested_close(actual, expected)
                        self.assert_nested_close(model.probs_with_prefix(enc, prefix), expected)

    def test_missing_or_invalid_span_metadata_fails_instead_of_silent_end_readout(self):
        model = toy_model()
        with self.assertRaises(ValueError):
            model.forward(encode(self.tok, RECORD))
        h = torch.randn(10, 2)
        for spans in (None, [], [(3, 6)], [(3, 5.0)], [(-1, 5)], [(6, 5)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                readout_features(h, 8, [5], readout="mean_span", opt_spans=spans)


if __name__ == "__main__":
    unittest.main()
