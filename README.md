# 📊 Vendor Invoice Intelligence Portal

An end-to-end Machine Learning application that predicts **freight costs** and flags **risky vendor invoices** for manual approval — built with Python, Scikit-learn, and Streamlit.

---

## 🧠 Business Problem

In large-scale procurement operations, two recurring challenges cost companies significantly:

1. **Unpredictable freight costs** — making budgeting and vendor negotiations difficult
2. **Anomalous vendor invoices** — where billed amounts don't match purchase orders, leading to financial leakage

This portal automates both using machine learning, reducing manual workload and improving financial accuracy.

---

## 🚀 Live Demo

> 🔗 [Coming Soon]

---

## 🏗️ Project Architecture

```
vendor-invoice-intelligence/
│
├── app.py                          # Streamlit frontend
│
├── inference/
│   ├── __init__.py
│   ├── predict_freight.py          # Freight cost inference
│   └── predict_invoice_flag.py     # Invoice risk inference
│
├── models/
│   ├── predict_freight.pkl         # Trained regression model
│   └── predict_flag_invoice.pkl    # Trained classification model
│
├── freight/
│   ├── data_preprocess.py          # Feature engineering for freight
│   ├── model_evaluation.py         # Regression model training
│   └── train.py                    # Freight model training pipeline
│
├── invoice_flag/
│   ├── data_preprocess.py          # Feature engineering + labeling
│   ├── model_evolution.py          # Classification model + GridSearch
│   └── train.py                    # Invoice flag training pipeline
│
├── requirements.txt
└── README.md
```

---

## 🤖 Models

### Model 1 — Freight Cost Prediction (Regression)

Predicts the freight cost for a vendor invoice based on invoice value and quantity.

| Feature | Description |
|---|---|
| `Dollars` | Total invoice dollar amount |
| `Quantity` | Number of items in the invoice |

**Models Evaluated:**

| Model | MAE | MSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Decision Tree | — | — | — |
| **Random Forest** ✅ | — | — | — |

> Best model selected automatically by lowest MAE and saved as `predict_freight.pkl`

---

### Model 2 — Invoice Risk Flagging (Classification)

Flags invoices that require manual approval based on abnormal patterns.

**Labeling Logic:**
- Invoice is flagged (`1`) if:
  - `|invoice_dollars - total_item_dollars| > $5`, OR
  - `avg_receiving_delay > 10 days`

| Feature | Description |
|---|---|
| `invoice_quantity` | Quantity billed on invoice |
| `invoice_dollars` | Dollar amount on invoice |
| `Freight` | Freight cost on invoice |
| `total_item_quantity` | Total quantity from purchase orders |
| `total_item_dollars` | Total dollars from purchase orders |

**Model:** Random Forest Classifier with GridSearchCV (F1-optimized)

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1 Score | — |

> Fill in your actual scores after training

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- SQLite database (`inventory.db`) with `vendor_invoice` and `purchases` tables

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Database Path

Set your database path as an environment variable:

```bash
# Windows
set DB_PATH=C:/path/to/your/inventory.db

# Mac/Linux
export DB_PATH=/path/to/your/inventory.db
```

### 5. Train the Models

```bash
# Train freight cost model
python freight/train.py

# Train invoice flag model
python invoice_flag/train.py
```

This saves trained `.pkl` files to the `models/` directory.

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🖥️ Application Features

### Freight Cost Prediction
- Input invoice quantity and dollar amount
- Get instant freight cost estimate
- Supports budgeting and vendor negotiations

### Invoice Risk Flagging
- Input 5 invoice and purchase order features
- Instantly flags invoices needing manual review
- Reduces fraud and financial anomalies

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

---

## 🗄️ Database Schema

### `vendor_invoice` table

| Column | Type | Description |
|---|---|---|
| PONumber | TEXT | Purchase order number |
| Quantity | INTEGER | Invoice quantity |
| Dollars | FLOAT | Invoice dollar amount |
| Freight | FLOAT | Freight cost |
| InvoiceDate | DATE | Date of invoice |
| PODate | DATE | Date of purchase order |
| PayDate | DATE | Date of payment |

### `purchases` table

| Column | Type | Description |
|---|---|---|
| PONumber | TEXT | Purchase order number |
| Brand | TEXT | Product brand |
| Quantity | INTEGER | Ordered quantity |
| Dollars | FLOAT | Order dollar amount |
| ReceivingDate | DATE | Date goods were received |
| PODate | DATE | Date of purchase order |

---

## 🔮 Future Improvements

- [ ] Add model confidence scores / probability display in UI
- [ ] Add data visualization dashboard (feature importance, prediction trends)
- [ ] Replace SQLite with PostgreSQL for production
- [ ] Add user authentication for the portal
- [ ] Implement model retraining pipeline
- [ ] Add SHAP explainability for invoice flag decisions
- [ ] REST API endpoint using FastAPI

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
