import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

def generate_synthetic_tabular_data(n_samples=1000, random_state=42):
    """Generate realistic synthetic tabular patient risk dataset."""
    np.random.seed(random_state)
    
    age = np.random.randint(20, 80, n_samples)
    bp = np.random.normal(120, 15, n_samples).astype(int)
    chol = np.random.normal(200, 30, n_samples).astype(int)
    max_hr = np.random.normal(150, 20, n_samples).astype(int)
    glucose = np.random.normal(100, 25, n_samples).astype(int)
    bmi = np.random.normal(26, 4, n_samples).round(1)
    angina = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    st_dep = np.random.exponential(1.0, n_samples).round(1)

    # Risk score calculation formula
    risk_score = (
        (age > 55) * 2.0 +
        (bp > 140) * 1.5 +
        (chol > 240) * 1.5 +
        (glucose > 125) * 2.0 +
        (bmi > 30) * 1.0 +
        angina * 2.5 +
        st_dep * 1.5 +
        np.random.normal(0, 1.0, n_samples)
    )

    y = np.array(pd.qcut(risk_score, q=3, labels=[0, 1, 2])).astype(int)

    X = np.column_stack([age, bp, chol, max_hr, glucose, bmi, angina, st_dep])
    feature_names = [
        "Age", "Systolic_BP", "Cholesterol", "Max_Heart_Rate",
        "Glucose_Level", "BMI", "Exercise_Angina", "ST_Depression"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["Target_Risk"] = y
    return df, X, y


def generate_synthetic_nlp_data():
    """Generate synthetic text sentiment corpus for NLP model training."""
    corpus = [
        ("The AI model diagnostic accuracy is outstanding and accurate.", "Positive"),
        ("Extremely fast inference and excellent model results.", "Positive"),
        ("Great user interface, powerful features, and easy workflow.", "Positive"),
        ("Loved the real-time visualization and detailed analytics.", "Positive"),
        ("High confidence predictions and super useful dashboard.", "Positive"),
        ("Solid overall performance and good documentation.", "Positive"),
        ("Acceptable accuracy, standard prediction response time.", "Neutral"),
        ("Model returned average confidence score with basic output.", "Neutral"),
        ("System running as expected with moderate latency.", "Neutral"),
        ("Results are okay but need further validation.", "Neutral"),
        ("Poor performance, inaccurate predictions, and slow latency.", "Negative"),
        ("Model failed to classify the target correctly, bad output.", "Negative"),
        ("High error rate and unreliable confidence metrics.", "Negative"),
        ("Unsatisfactory predictions and frustrating interface.", "Negative"),
        ("Terrible latency and constant inference timeout errors.", "Negative")
    ]
    
    # Expand corpus by adding slight variations
    expanded = []
    for text, label in corpus * 20:
        expanded.append((text, label))
    
    return expanded


def create_sample_synthetic_image(class_id=0, img_size=64):
    """Generate synthetic image patch representing 4 diagnostic classes."""
    img = Image.new("RGB", (img_size, img_size), color=(20, 25, 35))
    draw = ImageDraw.Draw(img)

    if class_id == 0:  # Normal
        draw.ellipse([15, 15, 49, 49], outline=(75, 185, 240), width=2)
    elif class_id == 1:  # Bacterial Condition
        draw.rectangle([10, 10, 54, 54], outline=(240, 140, 60), width=3)
        draw.line([10, 10, 54, 54], fill=(240, 140, 60), width=2)
    elif class_id == 2:  # Viral Infection
        draw.polygon([(32, 10), (10, 54), (54, 54)], outline=(240, 75, 95), width=2)
    else:  # Pathology Detected
        draw.ellipse([20, 20, 44, 44], fill=(220, 50, 80))
        draw.line([0, 32, 64, 32], fill=(255, 255, 255), width=1)

    return img
