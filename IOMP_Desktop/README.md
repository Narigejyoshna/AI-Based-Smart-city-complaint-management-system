# 🏙️ IOMP — Smart City AI Complaint Management System
## Desktop Application (Python + Tkinter + ML)

---

## 📁 Project Structure

```
IOMP_Desktop/
├── app.py              ← Main desktop app (run this)
├── train_model.py      ← Retrain the ML model
├── requirements.txt    ← Python dependencies
├── complaints.db       ← SQLite database (auto-created)
└── model/
    ├── category_model.pkl   ← Trained Naive Bayes model
    └── vectorizer.pkl       ← TF-IDF vectorizer
```

---

## ⚙️ Setup (VS Code)

### Step 1 — Install dependencies
Open a terminal in VS Code (`Ctrl+` `` ` ``) and run:
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Retrain the ML model
The `model/` folder already has trained files. Skip this if you want to use them.
To retrain from scratch:
```bash
python train_model.py
```

### Step 3 — Launch the desktop app
```bash
python app.py
```

---

## 🤖 ML Algorithm Details

| Component         | Details                                      |
|-------------------|----------------------------------------------|
| Algorithm         | Multinomial Naive Bayes                      |
| Vectorization     | TF-IDF (max 1000 features, n-gram 1-2)       |
| Duplicate Check   | Cosine Similarity (threshold 0.85)           |
| Categories        | Water, Sanitation, Electricity, Roads        |
| Priority Levels   | CRITICAL / HIGH / LOW (keyword-based)        |

---

## 🖥️ App Features

- **Submit Complaint** — Text input, AI auto-classifies category + priority
- **Admin Dashboard** — View all complaints, filter by priority/status, update status
- **Track Complaint** — Enter tracking ID to view status and progress
- **About Page** — Tech stack info and usage guide

---

## 🔧 Requirements

- Python 3.8 or higher
- tkinter (bundled with Python — no install needed)
- scikit-learn, joblib, numpy, pandas
