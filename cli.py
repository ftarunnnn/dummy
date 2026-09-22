#!/usr/bin/env python3
"""
NeuroVision AI Command Line Interface (CLI)
Usage:
  python cli.py train
  python cli.py metrics
  python cli.py predict-nlp --text "Great diagnostic model!"
  python cli.py predict-tabular --age 55 --bp 140 --chol 230 --hr 150 --glucose 110 --bmi 28.5 --angina 0 --stdep 1.2
"""

import argparse
import sys
import os
import json

def main():
    parser = argparse.ArgumentParser(description="NeuroVision AI CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: train
    subparsers.add_parser("train", help="Train all Deep Learning & ML models")

    # Command: metrics
    subparsers.add_parser("metrics", help="Display model benchmark metrics")

    # Command: predict-nlp
    nlp_parser = subparsers.add_parser("predict-nlp", help="Run NLP sentiment inference")
    nlp_parser.add_argument("--text", type=str, required=True, help="Input text string for sentiment analysis")

    # Command: predict-tabular
    tab_parser = subparsers.add_parser("predict-tabular", help="Run tabular risk model inference")
    tab_parser.add_argument("--age", type=int, default=50)
    tab_parser.add_argument("--bp", type=int, default=130)
    tab_parser.add_argument("--chol", type=int, default=210)
    tab_parser.add_argument("--hr", type=int, default=155)
    tab_parser.add_argument("--glucose", type=int, default=100)
    tab_parser.add_argument("--bmi", type=float, default=26.0)
    tab_parser.add_argument("--angina", type=int, default=0)
    tab_parser.add_argument("--stdep", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "train":
        print("Starting model training pipeline...")
        from src.training.train_all import main as run_train
        run_train()

    elif args.command == "metrics":
        summary_path = os.path.join("models", "checkpoints", "training_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        else:
            print("No trained checkpoint found. Run 'python cli.py train' first.")

    elif args.command == "predict-nlp":
        from src.models.nlp_net import TextSentimentNet
        nlp_path = os.path.join("models", "checkpoints", "nlp_model.pt")
        vocab_path = os.path.join("models", "checkpoints", "vocab.json")
        
        model = TextSentimentNet()
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                model.vocab = json.load(f)
        if os.path.exists(nlp_path):
            import torch
            model.load_state_dict(torch.load(nlp_path, map_location=torch.device('cpu')))
        
        res = model.predict_sentiment(args.text)
        print("\n--- NLP Prediction Result ---")
        print(json.dumps(res, indent=2))

    elif args.command == "predict-tabular":
        import numpy as np
        from src.models.tabular_net import TabularRiskClassifier
        tab_path = os.path.join("models", "checkpoints", "tabular_model.joblib")
        
        if os.path.exists(tab_path):
            model = TabularRiskClassifier.load(tab_path)
        else:
            print("Warning: Using untrained default tabular model.")
            model = TabularRiskClassifier()
            dummy_X = np.random.randn(50, 8)
            dummy_y = np.random.choice([0, 1, 2], 50)
            model.fit(dummy_X, dummy_y)

        features = np.array([[
            args.age, args.bp, args.chol, args.hr,
            args.glucose, args.bmi, args.angina, args.stdep
        ]])

        probs = model.predict_proba(features)[0]
        pred_class = int(np.argmax(probs))
        risk_labels = TabularRiskClassifier.RISK_LEVELS

        print("\n--- Tabular Risk Prediction Result ---")
        print(f"Risk Level: {risk_labels[pred_class]}")
        print(f"Score Confidence: {round(probs[pred_class] * 100, 2)}%")
        print("Feature Importances:")
        for k, v in model.get_feature_importances().items():
            print(f"  - {k}: {v}%")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
