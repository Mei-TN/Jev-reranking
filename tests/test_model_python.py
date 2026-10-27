"""The launcher restores owned libraries even after a controller replaces LD paths."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class ModelPythonLauncherTests(unittest.TestCase):
    def test_owned_loader_paths_and_literal_arguments(self):
        if os.name == "nt":
            shell = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if not shell.is_file():
                self.skipTest("Git Bash is unavailable")
        else:
            shell = shutil.which("sh")
            if not shell:
                self.skipTest("POSIX shell is unavailable")
        source = Path(__file__).resolve().parents[1] / "requirements/environment/model-python"
        with tempfile.TemporaryDirectory(prefix="jev-wrapper-") as temporary:
            prefix = Path(temporary) / "mock prefix"
            (prefix / "venv/bin").mkdir(parents=True)
            libraries = ["syslibs/usr/lib/x86_64-linux-gnu", "syslibs/lib/x86_64-linux-gnu",
                         "venv/lib/python3.12/site-packages/torch/lib",
                         "venv/lib/python3.12/site-packages/nvidia/cublas/lib"]
            for relative in libraries:
                (prefix / relative).mkdir(parents=True, exist_ok=True)
            wrapper = prefix / "model-python"
            wrapper.write_bytes(source.read_bytes())
            mock = prefix / "venv/bin/python"
            mock.write_text('#!/bin/sh\nprintf "%s\\n" "$LD_LIBRARY_PATH"\nprintf "%s\\0" "$@"\n',
                            encoding="utf-8", newline="\n")
            mock.chmod(0o755)
            arguments = ["-c", "literal with spaces", "$(not-executed)", "", '--data={"a":"b c"}']
            environment = dict(os.environ, LD_LIBRARY_PATH="/deliberately/replaced/by/controller",
                               CUDA_VISIBLE_DEVICES="", MSYS_NO_PATHCONV="1")
            result = subprocess.run([str(shell), str(wrapper), *arguments], env=environment,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            loader, raw_arguments = result.stdout.split(b"\n", 1)
            self.assertEqual(raw_arguments.decode().split("\0")[:-1], arguments)
            paths = loader.decode().split(":")
            self.assertTrue(all(paths[index].endswith("/" + name)
                                for index, name in enumerate(libraries)))
            self.assertNotIn("/deliberately/", loader.decode())


if __name__ == "__main__":
    unittest.main()
