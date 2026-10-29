"""Test stand-in for the real CLI: run the fake run, then stay alive (like the dashboard host)."""
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(root / "tools" / "fake_run.py"), *sys.argv[1:]], check=True)
time.sleep(600)
