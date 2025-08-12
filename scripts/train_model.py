import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['target'] = y
df.to_csv('data/synthetic_data.csv', index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('models/logreg_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")