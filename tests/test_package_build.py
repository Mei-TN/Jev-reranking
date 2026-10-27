"""Standard and direct wheels preserve public resources and omit local files."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

from jev.resources import _long_path


REPOSITORY = Path(__file__).resolve().parents[1]
BUILD_TOOLS = importlib.util.find_spec("build") is not None and importlib.util.find_spec("hatchling") is not None


@unittest.skipUnless(BUILD_TOOLS, "optional packaging tools build/hatchling are not installed")
class SourceDistributionBuildTests(unittest.TestCase):
    def test_standard_build_from_sdist_reuses_exact_deep_vendor_archive(self):
        with tempfile.TemporaryDirectory(dir=_long_path(Path(tempfile.gettempdir()))) as temporary:
            task = Path(temporary)
            source = _long_path(task / "project")
            source.mkdir()
            for name in ("pyproject.toml", ".gitignore", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "scripts/hatch_build.py",
                         "src/jev/__init__.py", "src/jev/config.py", "src/jev/resources.py"):
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPOSITORY / name, target)
            public_metadata, ignored_metadata = {}, set()
            for folder in ("configs", "provenance", "docs", "examples", "requirements", "third_party"):
                (source / folder).mkdir()
                (source / folder / "fixture.json").write_text('{"packaging_fixture":true}\n')
                (source / folder / "public").mkdir()
                (source / folder / "public/nested.json").write_text('{"nested_public_fixture":true}\n')
                for relative in (folder + "/fixture.json", folder + "/public/nested.json"):
                    public_metadata[relative] = (source / relative).read_bytes()
                for filename in ("private-fixture.json", "excluded-fixture.json"):
                    relative = folder + "/" + filename
                    (source / relative).write_text('{"local_nonsecret_sentinel":true}\n')
                    ignored_metadata.add(relative)
                with (source / ".gitignore").open("a") as ignore:
                    ignore.write("\n/" + folder + "/private-fixture.json\n")
            (source / "configs/site.local.json").write_text('{"local_nonsecret_site_sentinel":true}\n')
            ignored_metadata.add("configs/site.local.json")
            with (source / "pyproject.toml").open("a") as project:
                project.write('\n[tool.hatch.build]\nexclude = ["**/excluded-fixture.json"]\n')
            self.assertFalse((source / ".git").exists())
            member = "vendor/" + "/".join(["preserved_historical_source"] * 12) + "/tests/native.py"
            payload = b"# unchanged archived source beyond MAX_PATH\n"
            original = source.joinpath(*member.split("/"))
            original.parent.mkdir(parents=True)
            original.write_bytes(payload)
            (source / "vendor/Zeta.py").write_bytes(b"# uppercase path-order fixture\n")
            (source / "vendor/alpha.py").write_bytes(b"# lowercase path-order fixture\n")
            self.assertGreater(len(str(original)), 260)
            output = task / "artifacts"
            result = subprocess.run([sys.executable, "-I", "-B", "-m", "build", "--no-isolation", "--outdir", str(output)],
                                    cwd=source, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sdist, = output.glob("*.tar.gz")
            wheel, = output.glob("*.whl")
            with tarfile.open(sdist) as archive:
                names = archive.getnames()
                prefix = names[0].split("/")[0] + "/"
                self.assertFalse(any(name.startswith(prefix + "vendor/") for name in names))
                self.assertIn(prefix + "vendor.zip", names)
                self.assertIn(prefix + "vendor_index.json", names)
                for name in ignored_metadata:
                    self.assertNotIn(prefix + name, names)
                for name, content in public_metadata.items():
                    self.assertEqual(archive.extractfile(prefix + name).read(), content)
                archived = archive.extractfile(prefix + "vendor.zip").read()
                index = archive.extractfile(prefix + "vendor_index.json").read()
                independent = task / "unpacked"
                independent.mkdir()
                archive.extractall(independent, filter="data")
            with zipfile.ZipFile(io.BytesIO(archived)) as preserved:
                self.assertEqual(preserved.read(member), payload)
                self.assertEqual(preserved.namelist(), sorted(preserved.namelist()))
            with zipfile.ZipFile(wheel) as package:
                self.assertEqual(package.read("jev/_resources/vendor.zip"), archived)
                self.assertEqual(package.read("jev/_resources/vendor_index.json"), index)
                self.assertFalse(any(name.startswith("jev/_resources/vendor/") for name in package.namelist()))
                standard_resources = {name: package.read(name) for name in package.namelist()
                                      if name.startswith("jev/_resources/")}
            direct_output = task / "direct-artifacts"
            direct_result = subprocess.run([sys.executable, "-I", "-B", "-m", "build", "--no-isolation", "--wheel",
                                            "--outdir", str(direct_output)],
                                           cwd=source, capture_output=True, text=True, timeout=180)
            self.assertEqual(direct_result.returncode, 0, direct_result.stdout + direct_result.stderr)
            direct_wheel, = direct_output.glob("*.whl")
            with zipfile.ZipFile(direct_wheel) as package:
                direct_resources = {name: package.read(name) for name in package.namelist()
                                    if name.startswith("jev/_resources/")}
            self.assertEqual(direct_resources, standard_resources)
            self.assertEqual(direct_resources["jev/_resources/vendor.zip"], archived)
            self.assertEqual(direct_resources["jev/_resources/vendor_index.json"], index)
            for name, content in public_metadata.items():
                self.assertEqual(direct_resources["jev/_resources/" + name], content)
            for name in ignored_metadata:
                self.assertNotIn("jev/_resources/" + name, direct_resources)
            for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
                self.assertEqual(direct_resources["jev/_resources/" + name], (source / name).read_bytes())
            self.assertEqual(json.loads(index)["files"][member]["sha256"], hashlib.sha256(payload).hexdigest())
            unpacked = independent / prefix.rstrip("/")
            self.assertFalse((unpacked / ".git").exists())
            code = ("import hashlib,json,sys;sys.path.insert(0,sys.argv[1]);"
                    "from jev.config import root;p=root();"
                    "print(json.dumps({'sha256':hashlib.sha256((p/sys.argv[2]).read_bytes()).hexdigest(),"
                    "'unrelated_source_copied':(p/'src').exists()}))")
            environment = dict(os.environ, JEV_RESOURCE_CACHE=str(task / "resource-cache"))
            materialized = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(unpacked / "src"), member],
                                          cwd=task, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(materialized.returncode, 0, materialized.stderr)
            observed = json.loads(materialized.stdout)
            self.assertEqual(observed["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertFalse(observed["unrelated_source_copied"])


if __name__ == "__main__":
    unittest.main()
