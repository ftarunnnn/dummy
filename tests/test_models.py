import unittest
import torch
import numpy as np
from src.models.vision_net import VisionResNet, get_transforms
from src.models.tabular_net import TabularRiskClassifier
from src.models.nlp_net import TextSentimentNet
from src.training.data_generator import generate_synthetic_tabular_data, create_sample_synthetic_image

class TestModels(unittest.TestCase):

    def test_vision_resnet_forward(self):
        model = VisionResNet(num_classes=4)
        dummy_input = torch.randn(2, 3, 64, 64)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 4))

    def test_tabular_classifier_fit_predict(self):
        df, X, y = generate_synthetic_tabular_data(n_samples=100)
        model = TabularRiskClassifier()
        model.fit(X, y)
        preds = model.predict(X[:5])
        probs = model.predict_proba(X[:5])
        self.assertEqual(len(preds), 5)
        self.assertEqual(probs.shape, (5, 3))
        self.assertGreater(len(model.get_feature_importances()), 0)

    def test_nlp_sentiment_net(self):
        model = TextSentimentNet()
        sample_text = "The AI model performs great."
        model.build_vocab([sample_text])
        res = model.predict_sentiment(sample_text)
        self.assertIn(res["sentiment"], ["Negative", "Neutral", "Positive"])
        self.assertIn("confidence", res)

if __name__ == "__main__":
    unittest.main()
