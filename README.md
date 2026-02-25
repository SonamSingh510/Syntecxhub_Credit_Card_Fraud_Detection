# 💳 Credit Card Fraud Detection

Fraudulent transactions represent a tiny fraction of total data, making it difficult for standard models to learn. This project uses advanced sampling techniques and ensemble learning to prioritize **Recall**, ensuring that fraudulent activities are identified even at the risk of some false positives.

## 🛠️ Key Features

* **Exploratory Data Analysis (EDA):** Visualized the extreme class imbalance (99.8% normal vs. 0.2% fraud) to guide the modeling strategy.
* **Class Imbalance Handling:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training set without losing critical data.
* **Feature Scaling:** Normalized transaction amounts using `StandardScaler` to ensure the model remains unbiased toward large values.
* **Ensemble Learning:** Trained a **Random Forest Classifier** to capture complex, non-linear patterns in transaction behavior.
* **Metric-Driven Evaluation:** Focused on **ROC-AUC** and **Recall** to evaluate success beyond simple accuracy.

## 📊 Business Logic: Precision vs. Recall

In fraud detection, **Recall is King**. It is better to flag a suspicious legitimate transaction for verification (False Positive) than to let a real fraudster pass through undetected (False Negative).

## 💻 Tech Stack

* **Language:** Python 3.10
* **Libraries:** `scikit-learn`, `imbalanced-learn`, `pandas`, `seaborn`, `matplotlib`
