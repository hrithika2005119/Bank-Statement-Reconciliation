Bank Statement Reconciliation using Machine Learning
📌 Overview

The Bank Statement Reconciliation using Machine Learning project automates the process of matching bank statement transactions with company ledger records. It minimizes manual effort, reduces reconciliation errors, and improves efficiency by combining Machine Learning and RapidFuzz-based fuzzy matching.

🚀 Features
Upload Bank Statement and Company Ledger datasets (CSV)
Automatic transaction reconciliation
Fuzzy text matching using RapidFuzz
Machine Learning-based transaction classification
Interactive dashboard with reconciliation insights
Search and filter transactions
Reconciliation history tracking
Export reconciliation reports as CSV
Confidence score for matched transactions
🛠️ Technologies Used
Programming Language: Python
Frontend: Streamlit
Development Platform: Google Colab, VS Code
Libraries: Pandas, NumPy, Scikit-learn, RapidFuzz, Joblib, Matplotlib
Machine Learning Algorithm: Random Forest Classifier
📂 Project Structure
Bank-Statement-Reconciliation/
│
├── app.py
├── train_model.py
├── desc_encoder.pkl
├── fraud_model.pkl
├── type_encoder.pkl
├── requirements.txt
│
├── data/
│   ├── bank_transactions.csv
│   ├── company_transactions.csv
│
│
└── README.md
⚙️ Workflow
Upload bank statement and company ledger datasets.
Perform data preprocessing (cleaning, formatting, duplicate removal).
Generate features such as amount difference, date difference, and text similarity.
Calculate transaction similarity using RapidFuzz.
Train and evaluate the Random Forest Classifier.
Predict matched and unmatched transactions.
Display reconciliation results on the Streamlit dashboard.
Export reconciliation reports.
🤖 Machine Learning
Algorithm: Random Forest Classifier
Features Used:
Amount Difference
Date Difference
Transaction Description Similarity (RapidFuzz)
Model Accuracy: 95%
📊 Modules
Home
User Profile
Data Upload
Dashboard
Reconciliation Engine
Insights
History
Search & Filter
Reports & Export
▶️ Installation
git clone https://github.com/hrithika2005119/Bank-Statement-Reconciliation.git

cd Bank-Statement-Reconciliation

pip install -r requirements.txt

streamlit run app.py
📈 Future Enhancements
Real-time Bank API integration
Deep Learning-based transaction matching
Fraud detection module
Multi-user authentication
Cloud deployment with database support
👩‍💻 Author

Hrithika V
B.Sc. Computer Science (Artificial Intelligence & Data Science)

📄 License

This project is developed for academic and educational purposes.
