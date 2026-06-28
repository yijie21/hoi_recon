"""Smoke test for egoaero-collect CLI (SP4 Task 6)."""
import json, os, subprocess, sys


def test_collect_cli(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = str(tmp_path / "egodexr")
    r = subprocess.run([sys.executable, "-m", "egoaero.dataset.cli",
                        "--out", out, "--n", "1", "--max-attempts", "4"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(out, "summary.json"))
    s = json.load(open(os.path.join(out, "summary.json")))
    assert "decisions" in s and s["n_attempts"] >= 1
