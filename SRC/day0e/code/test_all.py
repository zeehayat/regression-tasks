from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).parent
files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "exercises").glob("*.py")) + sorted((ROOT / "solutions").glob("*.py"))
for path in files:
    if path.name == "test_all.py": continue
    subprocess.run([sys.executable, str(path)], check=True)
print(f"passed {len(files)-1} demo/exercise/solution files")
