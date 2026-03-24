"""Locust load test for Sentinel."""
import logging
import random

from locust import HttpUser, between, task

logger = logging.getLogger(__name__)


class MLServingUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.5, 2.0)

    def on_start(self):
        logger.info(f"Load test user started against {self.host}")

    @task(10)
    def predict(self):
        payload = {"features": [random.random() for _ in range(10)]}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.json()  # validate JSON is parseable
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def health(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def metrics(self):
        with self.client.get("/metrics", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")
