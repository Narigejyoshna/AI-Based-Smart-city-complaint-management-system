# AI-Based-Smart-city-complaint-management-system
# 🏙️ — Smart City AI Complaint Management System
is an intelligent desktop solution designed to bridge the gap between citizens and urban administration. Using Machine Learning, the system automatically categorizes public grievances and assigns priority levels to ensure urgent issues are addressed first.

## 🚀 Key Features
* **AI Auto-Classification:** Automatically detects if a complaint belongs to *Water, Sanitation, Electricity, or Roads* using NLP.
* **Smart Prioritization:** Analyzes text to label complaints as *CRITICAL, HIGH, or LOW* based on urgency keywords.
* **Duplicate Detection:** Uses **Cosine Similarity** to identify and prevent multiple reports of the same issue.
* **Admin Dashboard:** A dedicated interface for officials to track, filter, and update the status of city complaints.
* **Tracking System:** Allows users to monitor the real-time progress of their submitted issues via a unique ID.

## 🤖 Machine Learning Specs
* **Algorithm:** Multinomial Naive Bayes
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **Feature Engineering:** Supports n-grams (1-2) for better context understanding.
* **Similarity Threshold:** 0.85 (for duplicate identification).

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **GUI Framework:** Tkinter
* **ML Libraries:** Scikit-learn, Joblib, NumPy, Pandas
* **Database:** SQLite3

## ⚙️ Installation & Usage
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Narigejyoshna/AI-Based-Smart-city-complaint-management-system.git](https://github.com/Narigejyoshna/AI-Based-Smart-city-complaint-management-system.git)
   cd IOMP_Desktop
