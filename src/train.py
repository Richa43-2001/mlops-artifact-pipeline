import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def load_config(path='config/config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def train_model(X, y, config):
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter']
    )
    model.fit(X, y)
    return model

def save_model(model, path='model_train.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def main():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load config
    config = load_config()

    # Train model
    model = train_model(X, y, config)

    # Save model
    save_model(model)
    print("✅ Model training complete and saved as model_train.pkl")

if __name__ == "__main__":
    main()
