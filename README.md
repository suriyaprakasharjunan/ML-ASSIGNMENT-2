# Wine Quality Classification Using Machine Learning

## 1. Problem Statement

Wine quality assessment is traditionally performed through sensory evaluation by human experts. This project aims to develop a machine learning-based classification system that predicts wine quality using the physicochemical properties of wine.

The objective is to implement and compare five supervised machine learning classification algorithms using the same dataset and evaluate their performance using six evaluation metrics.

An interactive Streamlit application is also developed to allow users to evaluate the trained models using labelled CSV test data without retraining the models.

## 2. Dataset Description

**Dataset Name:** Wine Quality Dataset  
**Dataset Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/186/wine+quality

The project combines the red wine and white wine datasets into a single dataset.

### Dataset Information

| Characteristic | Details |
|---|---|
| Total instances | 6,497 |
| Input features | 12 |
| Target variable | quality |
| Target classes | 7 (quality scores 3–9) |
| Training instances | 5,197 |
| Testing instances | 1,300 |
| Train-test ratio | 80:20 |
| Random state | 42 |
| Problem type | Multiclass classification |

### Input Features

The dataset contains the following 12 input features:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Color (red or white)

**Target variable:** `quality`

The target variable represents the wine quality score.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Combined the red wine and white wine datasets.
- Added a categorical feature named `color` to distinguish red and white wine.
- Checked the dataset for missing values and duplicate records.
- Separated input features and target variable.
- Applied stratified train-test splitting with an 80:20 ratio.
- Used `StandardScaler` for numerical features.
- Used `OneHotEncoder` for categorical features.
- Implemented preprocessing using `ColumnTransformer` and Scikit-learn pipelines.

The same training and testing split was used for all five models to ensure a consistent comparison.

## 4. Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model was trained using the training dataset and evaluated using the same testing dataset.

The trained models, including their preprocessing pipelines, were saved using Joblib for reuse in the Streamlit application.

## 5. Evaluation Metrics

Six evaluation metrics were used to compare model performance.

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correctly classified instances. |
| AUC | Measures the area under the multiclass ROC curve using weighted one-vs-rest evaluation. |
| Precision | Measures the weighted precision across classes. |
| Recall | Measures the weighted recall across classes. |
| F1 Score | Represents the weighted harmonic mean of precision and recall. |
| MCC | Matthews Correlation Coefficient measures the overall quality of classification predictions. |

## 6. Model Performance Comparison

All five models were evaluated on the same test dataset containing 1,300 instances.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5415 | 0.7228 | 0.5424 | 0.5415 | 0.5142 | 0.2721 |
| Decision Tree | **0.5862** | 0.6955 | **0.5895** | 0.5862 | **0.5870** | **0.3893** |
| KNN | 0.5585 | 0.7389 | 0.5441 | 0.5585 | 0.5489 | 0.3259 |
| Naive Bayes | 0.2215 | 0.5525 | 0.3960 | 0.2215 | 0.2751 | 0.0697 |
| Support Vector Machine | 0.5692 | **0.7613** | 0.5360 | 0.5692 | 0.5352 | 0.3158 |

*All values are obtained from the executed model comparison results.*

## 7. Model-wise Observations

### 7.1 Logistic Regression

Logistic Regression was implemented as a baseline classification algorithm.

- Accuracy: 0.5415
- AUC: 0.7228
- Precision: 0.5424
- Recall: 0.5415
- F1 Score: 0.5142
- MCC: 0.2721

The model provides a reasonable baseline, achieving moderate classification performance across the evaluation metrics.

### 7.2 Decision Tree Classifier

The Decision Tree Classifier achieved:

- Accuracy: 0.5862
- AUC: 0.6955
- Precision: 0.5895
- Recall: 0.5862
- F1 Score: 0.5870
- MCC: 0.3893

It achieved the highest accuracy, precision, F1 score, and MCC among all five models. It is the strongest overall classifier based on these test results.

### 7.3 K-Nearest Neighbors

K-Nearest Neighbors classifies wine quality based on the distance between data points in the feature space.

- Accuracy: 0.5585
- AUC: 0.7389
- Precision: 0.5441
- Recall: 0.5585
- F1 Score: 0.5489
- MCC: 0.3259

KNN achieved the second-highest AUC among the five models and demonstrated competitive classification performance.

### 7.4 Gaussian Naive Bayes

Gaussian Naive Bayes uses a probabilistic classification approach based on Bayes' theorem and the Gaussian distribution assumption for numerical features.

- Accuracy: 0.2215
- AUC: 0.5525
- Precision: 0.3960
- Recall: 0.2215
- F1 Score: 0.2751
- MCC: 0.0697

Naive Bayes achieved the lowest performance across the six evaluation metrics in this experiment.

### 7.5 Support Vector Machine

Support Vector Machine was implemented with probability estimation to support multiclass AUC evaluation.

- Accuracy: 0.5692
- AUC: 0.7613
- Precision: 0.5360
- Recall: 0.5692
- F1 Score: 0.5352
- MCC: 0.3158

SVM achieved the highest AUC of 0.7613 among all five models, indicating the strongest performance in terms of the AUC metric.

## 8. Overall Best Model

Based on the evaluation results, the **Decision Tree Classifier is the overall best-performing model** in this experiment.

The reasons are:

- Highest accuracy: 0.5862
- Highest precision: 0.5895
- Highest F1 score: 0.5870
- Highest MCC: 0.3893

Although Support Vector Machine achieved the highest AUC of 0.7613, Decision Tree performed better across four of the six evaluation metrics.

Therefore, Decision Tree is selected as the overall best model based on its balanced classification performance on the test dataset.

## 9. Streamlit Web Application

An interactive Streamlit application was developed to evaluate the five pre-trained classification models.

### Application Features

- Select one of the five trained classification models.
- Upload a labelled CSV test dataset.
- Use the bundled `test_data.csv` when no file is uploaded.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display a confusion matrix.
- Display a classification report.
- Load saved models and preprocessing pipelines without retraining.

The application was tested locally, and all five models were verified to work successfully.

### Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/suriyaprakasharjunan/ML-ASSIGNMENT-2.git
```

Navigate to the project directory:

```bash
cd ML-ASSIGNMENT-2
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the application in a browser:

```text
http://localhost:8501
```

## 10. Project Repository

**GitHub Repository:**

https://github.com/suriyaprakasharjunan/ML-ASSIGNMENT-2

## 11. Live Application

**Streamlit Community Cloud URL:**

To be updated after successful deployment.

## 12. Project Structure

```text
ML-ASSIGNMENT-2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── ML_Assignment_2.ipynb
│
├── winequality-red.csv
├── winequality-white.csv
├── wine_quality_combined.csv
├── test_data.csv
│
├── model_comparison_results.csv
├── final_model_comparison.csv
│
└── model/
    └── Five saved classification model pipelines
```

## 13. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 14. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms for wine quality prediction.

The models were trained and evaluated using a common dataset split and compared using six performance metrics.

The Decision Tree Classifier achieved the best overall classification performance based on accuracy, precision, F1 score, and MCC, while Support Vector Machine achieved the highest AUC.

The saved machine learning pipelines were integrated into an interactive Streamlit application, allowing users to evaluate models using labelled CSV test data without retraining.

The project demonstrates the complete machine learning workflow, including data preprocessing, model training, performance evaluation, model persistence, and application development.