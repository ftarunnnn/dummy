import unittest
from fastapi.testclient import TestClient
from src.api.main import app

class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_metrics_endpoint(self):
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tabular_accuracy", data)

    def test_predict_tabular_endpoint(self):
        payload = {
            "Age": 55,
            "Systolic_BP": 140,
            "Cholesterol": 220,
            "Max_Heart_Rate": 150,
            "Glucose_Level": 110,
            "BMI": 28.5,
            "Exercise_Angina": 0,
            "ST_Depression": 1.2
        }
        response = self.client.post("/api/v1/predict/tabular", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_level", data)
        self.assertIn("feature_importances", data)

    def test_predict_nlp_endpoint(self):
        payload = {"text": "Fast diagnostic model accuracy"}
        response = self.client.post("/api/v1/predict/nlp", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)

if __name__ == "__main__":
    unittest.main()
