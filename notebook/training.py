# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🫀 Heart Disease Detection – Training Notebook
# 
# This notebook walks through the complete ML pipeline:
# 1. Load & explore the dataset
# 2. Preprocess features
# 3. Train classification models
# 4. Evaluate & compare
# 5. Save the best model

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 2. Load Dataset

# %%
# 📌 Change this path to your dataset
DATA_PATH = "../data/heart.csv"

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
df.info()

# %%
df.describe()

# %%
# Target distribution
target_col = "target"  # Change if your target column has a different name
print(df[target_col].value_counts())
df[target_col].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"])
plt.title("Target Distribution")
plt.show()

# %%
# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(14, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# %%
# Missing values
print("Missing values:")
print(df.isnull().sum())

# %% [markdown]
# ## 4. Preprocessing

# %%
# Separate features and target
y = df[target_col]
X = df.drop(columns=[target_col])

# Identify feature types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# %%
# Handle missing values
for col in num_cols:
    X[col].fillna(X[col].median(), inplace=True)
for col in cat_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded '{col}': {list(le.classes_)}")

# Encode target if needed
target_encoder = None
if y.dtype == "object":
    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(y), name=target_col)
    print(f"Encoded target: {list(target_encoder.classes_)}")

# %%
# Scale features
feature_names = list(X.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# %% [markdown]
# ## 5. Model Training & Evaluation

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = {}
best_model = None
best_f1 = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} (F1={best_f1:.4f})")

# %% [markdown]
# ## 6. Model Comparison

# %%
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, accs, width, label="Accuracy", color="#3498db")
ax.bar(x + width/2, f1s, width, label="F1 Score", color="#e74c3c")
ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Save Artifacts

# %%
os.makedirs("../model", exist_ok=True)

joblib.dump(best_model, "../model/model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")
joblib.dump(label_encoders, "../model/encoder.pkl")
if target_encoder:
    joblib.dump(target_encoder, "../model/target_encoder.pkl")

metadata = {
    "feature_names": feature_names,
    "numerical_features": num_cols,
    "categorical_features": cat_cols,
    "target_column": target_col,
    "target_classes": list(map(str, target_encoder.classes_)) if target_encoder else sorted(y.unique().tolist()),
    "cat_encodings": {col: list(le.classes_) for col, le in label_encoders.items()},
    "best_model_name": best_name,
    "results": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in results.items()},
}

with open("../model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ All artifacts saved to ../model/")
