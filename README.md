# PIMA Diabetes Prediction using K-Nearest Neighbors (KNN)

This is a machine learning project aimed at predicting the likelihood of diabetes in individuals using the PIMA Indians Diabetes dataset. The model is built using the **K-Nearest Neighbors (KNN)** algorithm and includes data preprocessing, visualization, model evaluation, and hyperparameter tuning.

## 📊 Dataset

- **Source**: [PIMA Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Description**: The datasets consists of several medical predictor variables and one target variable, `Outcome`. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Project Highlights

- Data cleaning and imputation of missing values
- Exploratory Data Analysis (EDA) with visualizations
- Feature scaling using `StandardScaler`
- Train-test split and stratified sampling
- Model training using **K-Nearest Neighbors**
- Hyperparameter tuning with `GridSearchCV`
- Evaluation using confusion matrix, ROC curve, and classification report

## Tools & Libraries

- Python, NumPy, Pandas
- Seaborn, Matplotlib, Missingno
- Scikit-learn, Mlxtend

## Model Performance

- Evaluated using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Curve
  - GridSearchCV for best `k`

## How to Run

###  Option 1: Run in Google Colab (Recommended for ease)
1. Open the notebook directly in Colab. [Click here to open in Colab](https://githubtocolab.com/vishnuxprasad/pima-diabetes-prediction-knn/blob/main/diabetes_prediction.ipynb).
2. Upload the dataset (`diabetes.csv`) when prompted.
3. Run the cells one by one.

> 💡 Tip: If the dataset isn't local, you can also load it directly from Kaggle using an [API token](https://www.kaggle.com/docs/api#interacting-with-datasets).

---

###  Option 2: Run Locally using Jupyter Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/vishnuxprasad/pima-diabetes-prediction-knn.git
   cd pima-diabetes-prediction-knn
2. Install dependencies (preferably in a virtual environment):
   
```
pip install -r requirements.txt
```
3. Launch the notebook:

```
jupyter notebook
```
Open `diabetes-prediction.ipynb` and run the cells.

### Option 3: Run as Python Script
1. Make sure dependencies are installed (preferably in a virtual environment):

```
pip install -r requirements.txt
```
2. Run the script:

```
python diabetes_prediction.py
```
