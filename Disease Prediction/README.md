# Disease Prediction System

This project is a **Diabetes Prediction System** that uses machine learning to predict whether a patient has diabetes based on diagnostic measurements.

## Project Structure

```
Disease Prediction/
├── Data/
│   ├── diabetes.csv              # Raw dataset
│   └── diabetes_cleaned.csv      # Cleaned dataset
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   └── 02_Model_Building.ipynb   # Model Training & Evaluation
├── app/
│   └── app.py                    # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Dataset Description

The dataset consists of several medical predictor variables and one target variable, `Outcome`. Predictor variables include:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: 0 = No Diabetes, 1 = Diabetes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bogamahesh/Disease-Prediction.git
cd "Disease Prediction"
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Exploratory Data Analysis
Open and run the `notebooks/01_EDA.ipynb` notebook to explore the dataset.

### 2. Run Model Building
Open and run the `notebooks/02_Model_Building.ipynb` notebook to:
- Train multiple machine learning models
- Evaluate and compare performance
- Save the best-performing model
- Generate feature importance analysis

### 3. Launch Web Application
Run the Streamlit app for interactive predictions:
```bash
streamlit run app/app.py
```

## Models Trained

The following models are trained and compared:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier
- LightGBM Classifier

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC
- Confusion Matrix

## Results

The best model is selected based on ROC-AUC and F1-score, then saved as `best_model.pkl` for use in the web application.

## Dependencies

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- jupyter
- streamlit
- joblib

## License

This project is open-source and available for educational and research purposes.

## Author

- **bogamahesh** - Initial work

Feel free to contribute or raise issues!
