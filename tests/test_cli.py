import subprocess
import sys


def test_real_provider_requires_confirmation(tmp_path):
    cp = subprocess.run([sys.executable,"-m","translator.cli","translate","--input","x","--output","o","--checkpoint","c","--failed-output","f","--provider","gemini"], capture_output=True, text=True)
    assert cp.returncode != 0 and "confirm-real-api" in (cp.stdout + cp.stderr)

