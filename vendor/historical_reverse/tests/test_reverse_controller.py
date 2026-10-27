"""CPU controller tests: first failure, owned identity, exact reverse receipts.

All subprocess fixtures are CPU-only, GPU queries are patched. No experiment
queue is executed, and fixture checkpoints are explicitly artificial.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import reverse_train_controller as ctl


class ReverseControllerTests(unittest.TestCase):
    def test_changed_identity_never_signals_other_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
        try:
            owner = ctl.identity(child.pid)
            wrong = {**owner, "start_ticks": owner["start_ticks"] + 1}
            result = ctl.owned_stop(child, wrong, 2)
            self.assertEqual(result["reason"], "owner_identity_changed")
            self.assertIsNone(child.poll())
            stopped = ctl.owned_stop(child, owner, 2)
            self.assertTrue(stopped["child_gone"])
            self.assertIsNotNone(stopped["native"])
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=3)

    def test_lane_first_failure_preserves_native_stderr_without_next_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "runs").mkdir()
            context = dict(deadline_epoch=time.time()+120, deadline_monotonic=time.monotonic()+120)
            first = dict(run_id="first", cell="fake", gpu_uuid="cpu-owned-fixture", expected_steps=1931,
                         argv=[sys.executable, "-c", "import sys,time;time.sleep(.05);print('native failure',file=sys.stderr);sys.exit(7)"])
            second = {**first, "run_id": "second"}
            queue = dict(root=temp, code_tree=temp, environment={}, training_cap_seconds=60)
            with patch.object(ctl, "verify_inputs"), patch.object(ctl, "gpu_check", return_value={"CPU_fixture": True}):
                result = ctl.lane(queue, context, [first, second])
            self.assertEqual(result[0]["status"], "FAILED")
            self.assertEqual(result[0]["native"], 7)
            self.assertIn("native failure", (root / "runs/first/TRAIN.stderr").read_text())
            self.assertEqual(result[1]["status"], "NOTRUN_PREVIOUS_FAILURE")
            self.assertFalse((root / "runs/second").exists())

    def test_insufficient_original_stage_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "runs").mkdir()
            queue = dict(root=temp, environment={}, training_cap_seconds=60)
            context = dict(deadline_epoch=time.time()+20, deadline_monotonic=time.monotonic()+20)
            spec = dict(run_id="no_start", gpu_uuid="cpu-owned-fixture")
            with patch.object(ctl.subprocess, "Popen", side_effect=AssertionError("must not start")):
                result = ctl.run_one(queue, context, spec)
            self.assertEqual(result["status"], "FAILED")
            self.assertNotIn("owner", result)

    def fixture_checkpoint(self, root):
        d = root / "diagnostics"; d.mkdir()
        required = ["head_initial.safetensors", "head_final.safetensors", "adapter_initial.safetensors",
                    "adapter_final.safetensors", "initial_capture.json", "fixed_sample.json",
                    "fixed_sample_initial.json", "fixed_sample_final.json", "update_norms.json"]
        for name in required:
            (d / name).write_bytes(b"explicitly artificial CPU fixture")
        for name in ["head.pt", "adapter_model.safetensors", "adapter_config.json"]:
            (root / name).write_bytes(b"explicitly artificial CPU fixture")
        spec = dict(expected_steps=6, records=48, seed=44, freeze_steps=0, head_config={"CPU_fixture": True},
                    expected_joint_updates=3, expected_head_only_updates=3, expected_first_head_only_update=4)
        boundary = dict(completed_optimizer_update=3, first_head_only_optimizer_update=4,
                        adapter_group_parameter_count=12, requires_grad_parameter_count=0,
                        non_none_grad_parameter_count=0, optimizer_group_retained=True,
                        phase="head_only", lm_training=False, head_training=True)
        (d / "reverse_freeze_boundary.json").write_text(json.dumps(boundary))
        (root / "training_metrics.json").write_text(json.dumps(dict(status="complete", stopped_early=False,
             optimizer_steps=6, planned_optimizer_steps=6, source_records_seen=48,
             reverse=dict(planned_joint_updates=3, observed_joint_updates=3,
                          planned_head_only_updates=3, observed_head_only_updates=3,
                          completed_optimizer_updates=6, current_phase="head_only", boundary_event=boundary))))
        (root / "training_config.json").write_text(json.dumps(dict(args={"seed":44, "reverse_head_only_fraction":.3},
             dse=dict(planned_optimizer_steps=6, freeze_optimizer_updates=0,
                      parameter_counts={"adapter_and_special_embeddings":12},
                      reverse_schedule=dict(enabled=True, fraction=.3, joint_optimizer_updates=3,
                          head_only_optimizer_updates=3, first_head_only_optimizer_update=4)), head_config=spec["head_config"])))
        rows = [dict(optimizer_update=i, adapter_group_parameter_count=12,
                     active_adapter_parameter_count=12 if i<=3 else 0,
                     adapter_requires_grad_parameter_count=12 if i<=3 else 0,
                     adapter_non_none_grad_parameter_count=12 if i<=3 else 0,
                     reverse_phase_for_this_update="joint" if i<=3 else "head_only") for i in range(1,7)]
        (d / "optimizer_updates.jsonl").write_text(''.join(json.dumps(row)+"\n" for row in rows))
        self.rebind_fixture_hashes(root)
        return spec

    def rebind_fixture_hashes(self, root):
        d = root / "diagnostics"
        value = dict(status="complete", optimizer_steps=6, RNG_preserved=True,
                     files_sha256={p.name:ctl.sha(p) for p in d.iterdir() if p.name != "completion.json"})
        (d / "completion.json").write_text(json.dumps(value))

    def test_semantic_tail_and_boundary_corruption_rejected_after_hash_rebind(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec = self.fixture_checkpoint(root)
            self.assertEqual(ctl.check_completed(root, spec)["actual_steps"], 6)
            with self.assertRaises(AssertionError):
                ctl.check_completed(root, {**spec, "expected_steps":1931})
            log = root / "diagnostics/optimizer_updates.jsonl"
            original = log.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[3].update(active_adapter_parameter_count=12,
                           adapter_requires_grad_parameter_count=12,
                           adapter_non_none_grad_parameter_count=12, reverse_phase_for_this_update="joint")
            log.write_text(''.join(json.dumps(row)+"\n" for row in rows))
            self.rebind_fixture_hashes(root)
            with self.assertRaises(AssertionError):
                ctl.check_completed(root, spec)
            log.write_text(original)
            boundary_file = root / "diagnostics/reverse_freeze_boundary.json"
            boundary = json.loads(boundary_file.read_text())
            boundary["completed_optimizer_update"] = 4
            boundary_file.write_text(json.dumps(boundary))
            metrics_file = root / "training_metrics.json"
            metrics = json.loads(metrics_file.read_text());metrics["reverse"]["boundary_event"] = boundary
            metrics_file.write_text(json.dumps(metrics));self.rebind_fixture_hashes(root)
            with self.assertRaises(AssertionError):
                ctl.check_completed(root, spec)

    def fixture_queue(self):
        head = dict(version=1, kind="pointer", dim=256, readout="end", rng_anchor_dim=256)
        queue = dict(root=ctl.ROOT, code_tree=ctl.ROOT+"/code/reverse-supplement-v1", python=sys.executable,
                     stage_cap_seconds=86400, training_cap_seconds=21600,
                     evaluation_and_publication_shared_cap_seconds=28800, retry_policy="NEVER",
                     cells=[f"F{fold}s{seed}" for fold,seed in ctl.CELLS],
                     data={"2":{"path":"/CPU_fixture/F2"},"3":{"path":"/CPU_fixture/F3"}}, runs=[])
        for index,(fold,seed) in enumerate(ctl.CELLS):
            total = 1932 if fold==2 else 1931
            run_id = f"REV_WINH4_f{fold}_s{seed}"
            flags = {"--reverse_head_only_fraction":".3", "--freeze_fraction":"0.0",
                     "--data":queue["data"][str(fold)]["path"], "--out":ctl.ROOT+"/runs/"+run_id+"/output",
                     "--seed":str(seed), "--lora_targets":"all", "--epochs":"1", "--batch":"1",
                     "--supplement_diagnostics":"1", "--accum":"8", "--lora":"64",
                     "--lr":"2.5e-5", "--head_lr":"2.5e-5", "--head_config":json.dumps(head)}
            argv = [sys.executable,"-P","-m","kev.train"] + [x for pair in flags.items() for x in pair]
            queue["runs"].append(dict(run_id=run_id,run_index=index,fold=fold,seed=seed,cell=f"F{fold}s{seed}",
                 gpu_index=index+4,gpu_uuid=ctl.GPUS[index],expected_steps=total,freeze_steps=0,
                 expected_joint_updates=total-580,expected_head_only_updates=580,
                 expected_first_head_only_update=total-579,evaluation_port_base=32400+index*8,
                 evaluation_port_block=list(range(32400+index*8,32408+index*8)),argv=argv,head_config=head))
        return queue

    def test_queue_rejects_old_stage_cap_and_wrong_GPU_port_or_reverse_flag(self):
        good = self.fixture_queue();ctl.validate_queue(good)
        bad = copy.deepcopy(good);bad["stage_cap_seconds"] = 259200
        with self.assertRaises(AssertionError):ctl.validate_queue(bad)
        bad = copy.deepcopy(good);bad["runs"][2]["gpu_uuid"] = ctl.GPUS[0]
        with self.assertRaises(AssertionError):ctl.validate_queue(bad)
        bad = copy.deepcopy(good);bad["runs"][1]["evaluation_port_base"] = 32400
        with self.assertRaises(AssertionError):ctl.validate_queue(bad)
        bad = copy.deepcopy(good);argv = bad["runs"][0]["argv"]
        argv[argv.index("--reverse_head_only_fraction")+1] = "0"
        with self.assertRaises(AssertionError):ctl.validate_queue(bad)

    def test_publication_after_original_clock_expiry_cannot_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            original_fsync = os.fsync
            target = Path(temp) / "late.json"
            stop = (time.time()+20, time.monotonic()+20)
            def expire_after_fsync(fd):
                original_fsync(fd)
                ctl.time.time = lambda: stop[0]+1
            with patch.object(ctl.time,"time",wraps=time.time), patch.object(ctl.os,"fsync",side_effect=expire_after_fsync):
                with self.assertRaises(TimeoutError):
                    ctl.durable(target,{"CPU_fixture": True},exclusive=True,stops=stop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
