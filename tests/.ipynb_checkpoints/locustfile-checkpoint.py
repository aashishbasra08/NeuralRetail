# tests/locustfile.py
# This file runs a load test against the deployed FastAPI service.
# It simulates 200 concurrent users sending churn and demand prediction requests.
# Install: pip install locust
# Run: locust -f tests/locustfile.py --host=https://neuralretail-api.railway.app
# Then open browser: http://localhost:8089
# Set users to 200, spawn rate to 10, then start.

from locust import HttpUser, task, between

class NeuralRetailUser(HttpUser):
    wait_time = between(1, 3)
    headers   = {"Authorization": "Bearer neuralretail-key-2026"}

    @task(3)
    def predict_churn(self):
        # This task runs 3x more than others — churn is the main endpoint
        payload = {
            "customer_id": "C1234",
            "recency":     45,
            "frequency":   8,
            "monetary":    320.5
        }
        with self.client.post(
            "/predict/churn",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "churn_probability" not in data:
                    response.failure("Missing churn_probability in response")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)
    def predict_demand(self):
        # Demand forecast endpoint — runs 2x
        payload = {
            "sku_id":       "SKU-0001",
            "horizon_days": 30
        }
        self.client.post(
            "/predict/demand",
            json=payload,
            headers=self.headers
        )

    @task(1)
    def health_check(self):
        # Basic health check — runs 1x
        self.client.get("/health")