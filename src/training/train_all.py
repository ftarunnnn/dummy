import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

from src.models.vision_net import VisionResNet, get_transforms
from src.models.tabular_net import TabularRiskClassifier
from src.models.nlp_net import TextSentimentNet
from src.training.data_generator import (
    generate_synthetic_tabular_data,
    generate_synthetic_nlp_data,
    create_sample_synthetic_image
)

def train_tabular(checkpoint_dir):
    """Train Tabular Risk Classifier Model."""
    print("--- Training Tabular Risk Classifier ---")
    df, X, y = generate_synthetic_tabular_data(n_samples=1200)
    
    # Train / Test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = TabularRiskClassifier()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    save_path = os.path.join(checkpoint_dir, "tabular_model.joblib")
    model.save(save_path)
    print(f"Tabular Model trained. Metrics: {metrics}")
    print(f"Saved to: {save_path}\n")
    return metrics, model.get_feature_importances()


def train_vision(checkpoint_dir, epochs=5):
    """Train Vision Convolutional Neural Network."""
    print("--- Training Vision ResNet Model ---")
    train_tf, val_tf = get_transforms(img_size=64)
    
    images, labels = [], []
    for c in range(4):
        for _ in range(150):
            img = create_sample_synthetic_image(class_id=c, img_size=64)
            tensor = val_tf(img)
            images.append(tensor)
            labels.append(c)

    X_tensor = torch.stack(images)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VisionResNet(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_b.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_b.size(0)
            correct += (predicted == y_b).sum().item()

        epoch_acc = round(correct / total * 100, 2)
        epoch_loss = round(running_loss / total, 4)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc}%")

    save_path = os.path.join(checkpoint_dir, "vision_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Vision Model saved to: {save_path}\n")
    return {"accuracy": epoch_acc, "loss": epoch_loss}


def train_nlp(checkpoint_dir, epochs=8):
    """Train Text Sentiment Neural Network."""
    print("--- Training NLP Sentiment Net ---")
    corpus = generate_synthetic_nlp_data()
    texts = [item[0] for item in corpus]
    labels_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    y_labels = [labels_map[item[1]] for item in corpus]

    model = TextSentimentNet()
    model.build_vocab(texts)

    tensors = [model.text_to_tensor(t).squeeze(0) for t in texts]
    X_tensor = torch.stack(tensors)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x_b, y_b in loader:
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_b.size(0)
            _, preds = torch.max(outputs, 1)
            total += y_b.size(0)
            correct += (preds == y_b).sum().item()

    save_path = os.path.join(checkpoint_dir, "nlp_model.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    
    torch.save(model.state_dict(), save_path)
    with open(vocab_path, "w") as f:
        json.dump(model.vocab, f, indent=2)

    final_acc = round(correct / total * 100, 2)
    print(f"NLP Model trained. Accuracy: {final_acc}%")
    print(f"Saved model to: {save_path} and vocab to: {vocab_path}\n")
    return {"accuracy": final_acc}


def main():
    """Execute end-to-end model training suite."""
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tabular_metrics, feature_imp = train_tabular(checkpoint_dir)
    vision_metrics = train_vision(checkpoint_dir)
    nlp_metrics = train_nlp(checkpoint_dir)

    summary = {
        "tabular_metrics": tabular_metrics,
        "feature_importances": feature_imp,
        "vision_metrics": vision_metrics,
        "nlp_metrics": nlp_metrics
    }

    summary_path = os.path.join(checkpoint_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("==========================================")
    print("      NeuroVision AI Training Complete    ")
    print("==========================================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
