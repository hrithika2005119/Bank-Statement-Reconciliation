# Bank Statement Reconciliation using Machine Learning

## 📌 Overview

The **Bank Statement Reconciliation using Machine Learning** project automates the reconciliation process by matching bank statement transactions with company ledger records. It combines **Machine Learning** and **RapidFuzz-based fuzzy matching** to reduce manual effort, improve accuracy, and speed up financial reconciliation.

---

## 🚀 Features

* Upload bank statement and company ledger CSV files
* Automated transaction matching
* Fuzzy text matching using RapidFuzz
* Machine Learning-based reconciliation
* Interactive Streamlit dashboard
* Search and filter transactions
* Confidence score for matched transactions
* Reconciliation history tracking
* Export reconciliation reports as CSV

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Frontend:** Streamlit
* **Development Tools:** Google Colab, VS Code
* **Libraries:** Pandas, NumPy, Scikit-learn, RapidFuzz, Joblib, Matplotlib
* **Machine Learning Algorithm:** Random Forest Classifier

---

## 📂 Project Structure

```text
Bank-Statement-Reconciliation/

├── app.py
├── train_model.py
├── fraud_model.pkl
├── desc_encoder.pkl
├── type_encoder.pkl
├── requirements.txt
│
├── data/
│   ├── bank_transactions.csv
│   └── company_transactions.csv
│
└── README.md
```

---

## ⚙️ Workflow

1. Upload bank statement and company ledger datasets.
2. Clean and preprocess transaction data.
3. Generate features such as amount difference, date difference, and description similarity.
4. Calculate text similarity using RapidFuzz.
5. Train and evaluate the Random Forest Classifier.
6. Predict matched and unmatched transactions.
7. Display reconciliation insights on the Streamlit dashboard.
8. Export reconciliation reports.

---

## 🤖 Machine Learning

**Algorithm:** Random Forest Classifier

**Features Used:**

* Amount Difference
* Date Difference
* Transaction Description Similarity (RapidFuzz)

**Model Accuracy:** **95%**

---

## 📊 Application Modules

* Home
* User Profile
* Data Upload
* Dashboard
* Reconciliation Engine
* Insights
* History
* Search & Filter
* Reports & Export

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/hrithika2005119/Bank-Statement-Reconciliation.git
```

Navigate to the project folder:

```bash
cd Bank-Statement-Reconciliation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* Real-time bank API integration
* AI-powered transaction matching
* Fraud detection module
* Multi-user authentication
* Cloud deployment with database support

---

## 👩‍💻 Author

**Hrithika V**

B.Sc. Computer Science (Artificial Intelligence & Data Science)

Interested in **Data Science, Machine Learning, Artificial Intelligence, and Data Analytics**.

---

## 📄 License

This project was developed for **academic and educational purposes**.

