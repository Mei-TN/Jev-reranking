"""Contract-test setup diagnostics must precede the expensive check sequence."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jev_check_setup", ROOT / "scripts/check.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class CheckSetupTests(unittest.TestCase):
    def probe(self, *, installed="2.13.5", missing=False, api_error=None):
        output, error = io.StringIO(), io.StringIO()
        version = CHECK.metadata.PackageNotFoundError("pydantic") if missing else None
        schema_api = SimpleNamespace(BaseModel=object, Field=object, model_validator=object)
        importer = SimpleNamespace(import_module=lambda _: schema_api)
        if api_error:
            def fail_import(_):
                raise api_error
            importer.import_module = fail_import
        with patch.object(CHECK.metadata, "version", return_value=installed, side_effect=version), \
                patch.object(CHECK, "importlib", importer), redirect_stdout(output), redirect_stderr(error):
            result = CHECK.check_test_dependencies()
        return result, output.getvalue(), error.getvalue()

    def test_missing_pin_reports_same_interpreter_install_command(self):
        result, output, error = self.probe(missing=True)
        self.assertEqual(result, 2)
        self.assertEqual(output, "")
        self.assertIn("missing pydantic==2.13.5", error)
        self.assertIn(CHECK.sys.executable, error)
        self.assertIn('pip install -e ".[test]"', error)

    def test_wrong_version_is_not_accepted_as_the_pinned_test_environment(self):
        result, _, error = self.probe(installed="1.10.0")
        self.assertEqual(result, 2)
        self.assertIn("expected pydantic==2.13.5", error)

    def test_installed_metadata_without_importable_api_fails(self):
        result, _, error = self.probe(api_error=ImportError("missing transitive dependency"))
        self.assertEqual(result, 2)
        self.assertIn("Pydantic 2 schema API cannot be imported", error)

    def test_declared_pin_with_schema_api_passes(self):
        result, output, error = self.probe()
        self.assertEqual(result, 0)
        self.assertIn("pydantic==2.13.5", output)
        self.assertEqual(error, "")

    def test_failed_preflight_preserves_native_exit_and_stops_before_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checks"

            def fail_preflight(command, **kwargs):
                kwargs["stderr"].write("missing pinned test dependency\n")
                return SimpleNamespace(returncode=2)

            with patch.object(CHECK.sys, "argv", ["check.py", "--output", str(output)]), \
                    patch.object(CHECK.subprocess, "run", side_effect=fail_preflight) as native, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = CHECK.main()
            self.assertEqual(result, 1)
            self.assertEqual(native.call_count, 1)
            report = json.loads((output / "REPORT.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["checks"][0]["name"], "test_dependencies")
            self.assertEqual(report["checks"][0]["native_exit"], 2)
            self.assertIn("missing pinned", (output / "test_dependencies.stderr.log").read_text(encoding="utf-8"))
            self.assertFalse((output / "contracts.stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
