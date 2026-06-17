# Disease Prediction

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project that predicts diabetes risk from patient diagnostic measurements using the PIMA Indians Diabetes dataset.

## ML Pipeline

```text
EDA
  |
  v
Feature engineering and data cleaning
  |
  v
Train/test split
  |
  v
Scaling fitted on training data only
  |
  v
Model training and hyperparameter tuning
  |
  v
Evaluation on held-out test data
```

## Results

The model-building notebook trains Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM models. The final tuned model selected by cross-validated ROC-AUC was LightGBM.

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Tuned LightGBM | 0.8701 | 0.8400 | 0.7778 | 0.8077 |

These metrics were reproduced from `Disease Prediction/Data/diabetes_cleaned.csv` using the notebook split configuration: `test_size=0.2`, `random_state=42`, and stratified labels.

## Resume Bullet

- Built a diabetes disease prediction system using LightGBM with Accuracy: 87% and F1-score: 0.81, deployed through a Streamlit web app for interactive patient risk prediction.

Key metrics:

- Model: LightGBM
- Accuracy: 87%
- F1-score: 0.81

## Data Leakage Fix

Scaling is applied after the train/test split. The scaler is fitted only on `X_train` with `fit_transform`, and the test set is transformed separately with `transform`. This prevents information from the held-out test data from leaking into preprocessing.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
Disease Prediction/
|-- Data/
|   |-- diabetes.csv
|   `-- diabetes_cleaned.csv
|-- notebooks/
|   |-- 01_EDA.ipynb
|   `-- 02_Model_Building.ipynb
|-- app/
|   `-- app.py
|-- best_model.pkl
|-- scaler.pkl
|-- requirements.txt
`-- README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/bogamahesh/Disease-Prediction.git
cd Disease-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks:

```bash
jupyter notebook "Disease Prediction/notebooks/01_EDA.ipynb"
jupyter notebook "Disease Prediction/notebooks/02_Model_Building.ipynb"
```

Run the Streamlit app:

```bash
streamlit run "Disease Prediction/app/app.py"
```

The app loads `Disease Prediction/best_model.pkl` and `Disease Prediction/scaler.pkl` for interactive diabetes risk predictions.

## Disclaimer

This project is for educational purposes only and is not a medical diagnosis tool. Consult a qualified healthcare professional for medical advice.
