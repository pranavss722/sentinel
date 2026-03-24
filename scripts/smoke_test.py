"""Smoke test for the full docker-compose + serving stack."""
import sys

import httpx

BASE_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5000"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, "OK"))
    except Exception as e:
        results.append((name, False, str(e)))


def test_health():
    resp = httpx.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200, f"status {resp.status_code}"
    body = resp.json()
    assert body["status"] == "ok", f"body: {body}"


def test_predict():
    resp = httpx.post(
        f"{BASE_URL}/predict",
        json={"features": [0.5] * 10},
        timeout=5,
    )
    assert resp.status_code == 200, f"status {resp.status_code}"
    body = resp.json()
    assert "score" in body, f"missing 'score': {body}"
    assert "model" in body, f"missing 'model': {body}"
    score = body["score"]
    assert isinstance(score, float) and 0 <= score <= 1, f"bad score: {score}"
    assert body["model"] in ("champion", "challenger"), f"bad model: {body['model']}"


def test_metrics():
    resp = httpx.get(f"{BASE_URL}/metrics", timeout=5)
    assert resp.status_code == 200, f"status {resp.status_code}"
    assert "prediction_requests_total" in resp.text, "missing prediction_requests_total"


def test_mlflow_health():
    resp = httpx.get(f"{MLFLOW_URL}/health", timeout=5)
    assert resp.status_code == 200, f"status {resp.status_code}"


def main():
    check("GET /health", test_health)
    check("POST /predict", test_predict)
    check("GET /metrics", test_metrics)
    check("MLflow /health", test_mlflow_health)

    print()
    all_pass = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All smoke tests passed.")
        sys.exit(0)
    else:
        print("Some smoke tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
