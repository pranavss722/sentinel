"""Run Locust load test headlessly and save results to reports/."""
import os
import subprocess
import sys


def main():
    os.makedirs("reports", exist_ok=True)

    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--headless",
        "--users", "50",
        "--spawn-rate", "5",
        "--run-time", "60s",
        "--host", "http://localhost:8000",
        "--csv", "reports/load_test",
        "--exit-code-on-error", "1",
    ]

    result = subprocess.run(cmd)

    print("Load test complete. Results in reports/load_test_*.csv")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
