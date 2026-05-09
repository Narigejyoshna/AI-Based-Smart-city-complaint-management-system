"""
IOMP — Train ML Model
Multinomial Naive Bayes + TF-IDF Vectorizer
Run this once before launching app.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Create model directory ──────────────────
os.makedirs("model", exist_ok=True)

# ── Training data ───────────────────────────
data = {
    "complaint": [
        # Water
        "Water leakage near school causing road damage",
        "Pipeline burst in residential area wasting water",
        "No water supply in sector 5 since morning",
        "Dirty water coming from tap in colony",
        "Water logging on main road after rain",
        "Broken water pipe flooding the street",
        "Underground water pipeline damaged by construction",
        "Water supply disrupted for 3 days in locality",

        # Sanitation
        "Garbage overflow in ward 5 causing bad smell",
        "Drain blockage near hospital spreading diseases",
        "Open waste dumping in residential area",
        "Sewage water leaking on street",
        "Mosquito breeding due to stagnant dirty water",
        "Overflowing dustbins causing unhygienic conditions",
        "Drainage blocked causing sewage overflow",
        "Garbage not collected for over a week",

        # Electricity
        "Street light not working near market",
        "Electric wire broken and hanging dangerously",
        "Power outage in entire neighborhood",
        "Sparking electrical pole near school",
        "Voltage fluctuation damaging appliances",
        "Transformer damaged causing power failure",
        "High tension wire broken near playground",
        "Electric pole fallen on road blocking traffic",

        # Roads
        "Road potholes causing accidents on highway",
        "Broken road with deep cracks",
        "Missing manhole cover on busy street",
        "Road construction incomplete causing traffic",
        "Damaged speed breaker causing vehicle damage",
        "Footpath broken and dangerous for pedestrians",
        "Road cave-in near market area",
        "No street signs on dangerous intersection",
    ],
    "category": [
        "Water", "Water", "Water", "Water", "Water", "Water", "Water", "Water",
        "Sanitation", "Sanitation", "Sanitation", "Sanitation", "Sanitation",
        "Sanitation", "Sanitation", "Sanitation",
        "Electricity", "Electricity", "Electricity", "Electricity", "Electricity",
        "Electricity", "Electricity", "Electricity",
        "Roads", "Roads", "Roads", "Roads", "Roads", "Roads", "Roads", "Roads",
    ]
}

df = pd.DataFrame(data)

# ── Vectorizer ──────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["complaint"])
y = df["category"]

# ── Train/Test split ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Naive Bayes ───────────────────────
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ── Evaluate ────────────────────────────────
y_pred = model.predict(X_test)
print(f"\n✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save ────────────────────────────────────
joblib.dump(model, "model/category_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\n✅ Models saved to model/")

# ── Test predictions ─────────────────────────
tests = [
    "Water pipeline burst on main road",
    "Garbage not collected for a week",
    "Street light broken in dark area",
    "Deep potholes on highway",
    "Sewage overflow near market",
]
print("\n🔮 Sample Predictions:")
for complaint in tests:
    vec = vectorizer.transform([complaint])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec).max()
    print(f"  '{complaint}'")
    print(f"   → {pred} (confidence: {conf:.0%})\n")
