"""
Master Test Runner for Part 1 Shooting Script Code & Solutions
"""
import sys
import subprocess
from pathlib import Path

CODE_DIR = Path(__file__).parent.resolve()

P1_DEMOS = [
    "p1_01_secant_table.py",
    "p1_02_h_too_small.py",
    "p1_03_rule_check.py",
    "p1_04_chain_squared_error.py",
    "p1_05_dropped_factor.py",
    "p1_06_second_derivative.py",
    "solutions/p1_ex1_numerical_slope.py",
    "solutions/p1_ex2_five_polynomials.py",
    "solutions/p1_ex3_chain_proof.py"
]

def run_test_suite():
    print("=======================================================")
    print("🚀 PART 1 SHOOTING SCRIPT CODE TEST SUITE")
    print("=======================================================\n")
    
    passed = 0
    failed = 0
    
    for script in P1_DEMOS:
        script_path = CODE_DIR / script
        print(f"Running {script}...")
        try:
            res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, check=True)
            print(res.stdout.strip())
            print(f"✅ PASS: {script}\n")
            passed += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ FAIL: {script}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            failed += 1
            
    print("=======================================================")
    print(f"PART 1 TEST SUMMARY: {passed} PASSED | {failed} FAILED")
    print("=======================================================")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_test_suite()
