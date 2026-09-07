# 🩺 Breast Cancer Classification using Machine Learning

A Machine Learning project that demonstrates **supervised classification techniques** for predicting breast cancer outcomes using different Machine Learning algorithms.

The project implements and compares multiple classification models, including **Decision Tree, Logistic Regression, Bagging, Boosting, Random Forest, and Voting Classifiers**.

---

## 📌 Project Overview

Breast cancer classification is a common Machine Learning problem where the objective is to classify observations into different target categories based on their input features.

In this project, the `breast_cancer.csv` dataset is used to train and evaluate several supervised Machine Learning algorithms.

The main objective is to understand how different classification algorithms perform on the same dataset and to explore **ensemble learning techniques** for improving predictive performance.

---

## 🎯 Objectives

* Load and explore the breast cancer dataset.
* Separate input features and target labels.
* Split the dataset into training and testing sets.
* Perform feature scaling where required.
* Train multiple Machine Learning classification models.
* Generate predictions using trained models.
* Evaluate model performance.
* Understand individual classifiers and ensemble learning techniques.
* Compare different approaches for breast cancer classification.

---

## 🤖 Machine Learning Algorithms

The repository contains implementations of the following models:

### 1. Decision Tree 🌳

Decision Tree Classifier creates a tree-like structure of decisions to classify observations.

**File:**

```text
Individual_Breastcancer_DT.py
```

---

### 2. Logistic Regression 📈

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to a particular class.

**File:**

```text
Individual_Breastcancer_Logistic.py
```

---

### 3. Bagging 🌲

Bagging (Bootstrap Aggregating) trains multiple models on different samples of the training data and combines their predictions.

**File:**

```text
Individual_Breastcancer_bagging.py
```

---

### 4. Bagging with Extra Trees

This implementation demonstrates an ensemble approach using randomized tree-based estimators.

**File:**

```text
Individual_Breastcancer_baggingX.py
```

---

### 5. Boosting 🚀

Boosting combines multiple weak learners sequentially, where each new learner attempts to improve upon the errors made by previous learners.

**File:**

```text
Individual_Breastcancer_boosting.py
```

---

### 6. Random Forest 🌳🌳🌳

Random Forest combines multiple decision trees and aggregates their predictions to produce a more robust classifier.

**File:**

```text
Individual_Breastcancer_randomForest.py
```

---

### 7. Hard Voting Classifier 🗳️

Hard Voting combines predictions from multiple classifiers and selects the class receiving the majority of votes.

**File:**

```text
Individual_Breastcancer_voting_hard.py
```

---

### 8. Soft Voting Classifier 🧠

Soft Voting combines the predicted probabilities from multiple classifiers and selects the class with the highest combined probability.

**File:**

```text
Individual_Breastcancer_voting_soft.py
```

---

## 📂 Project Structure

```text
Breast-Cancer-Dataset-/
│
├── breast_cancer.csv
│
├── Individual_Breastcancer_DT.py
├── Individual_Breastcancer_Logistic.py
├── Individual_Breastcancer_bagging.py
├── Individual_Breastcancer_baggingX.py
├── Individual_Breastcancer_boosting.py
├── Individual_Breastcancer_randomForest.py
├── Individual_Breastcancer_voting_hard.py
├── Individual_Breastcancer_voting_soft.py
│
└── README.md
```

The repository currently contains the dataset and separate Python implementations for each classification approach.

---

## 📊 Dataset

The project uses:

```text
breast_cancer.csv
```

The dataset is loaded using Pandas:

```python
df = pd.read_csv("breast_cancer.csv")
```

The target variable used for classification is:

```text
target
```

The feature matrix `X` is created by removing the target column, while `Y` contains the target values.

```python
X = df.drop("target", axis=1)
Y = df["target"]
```

---

## 🔄 Machine Learning Workflow

The project follows a standard supervised Machine Learning workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Creation
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
```

The implementations use an **80% training / 20% testing** split with `random_state=42`.

---

## 📏 Model Evaluation

The models use classification evaluation techniques such as:

### Accuracy

Accuracy measures the proportion of correctly classified observations.

```python
accuracy_score(Y_test, Y_pred)
```

### Confusion Matrix

The confusion matrix provides a detailed view of correct and incorrect predictions for each class.

```python
confusion_matrix(Y_test, Y_pred)
```

Some scripts also import `classification_report` for additional classification metrics such as precision, recall, and F1-score.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Machine Learning**
* **Supervised Learning**
* **Ensemble Learning**

### Main Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ashu-k10/Breast-Cancer-Dataset-.git
```

### 2. Navigate to the project directory

```bash
cd Breast-Cancer-Dataset-
```

### 3. Install the required libraries

```bash
pip install pandas scikit-learn
```

---

## ▶️ How to Run

You can run any individual Machine Learning implementation.

For example:

```bash
python Individual_Breastcancer_DT.py
```

For Logistic Regression:

```bash
python Individual_Breastcancer_Logistic.py
```

For Random Forest:

```bash
python Individual_Breastcancer_randomForest.py
```

For Voting Classifiers:

```bash
python Individual_Breastcancer_voting_hard.py
```

or

```bash
python Individual_Breastcancer_voting_soft.py
```

Make sure that `breast_cancer.csv` is present in the same directory as the Python files.

---

## 📈 Expected Output

The programs display information such as:

```text
Shape of dataset
First few records
X shape
Y shape
Accuracy
Confusion Matrix
```

Example:

```text
Accuracy : 0.XX

Confusion Matrix :
[[XX XX]
 [XX XX]]
```

The exact results may vary depending on the algorithm and implementation.

---

## 🧩 Concepts Demonstrated

This project provides practical implementation of several important Machine Learning concepts:

* Supervised Learning
* Binary Classification
* Train-Test Split
* Feature Scaling
* Decision Trees
* Logistic Regression
* Ensemble Learning
* Bagging
* Boosting
* Random Forest
* Voting Classifiers
* Hard Voting
* Soft Voting
* Model Evaluation
* Accuracy
* Confusion Matrix
* Classification Report

---

## 🚀 Future Improvements

The project can be further improved by adding:

* Exploratory Data Analysis (EDA)
* Data visualization
* Precision, Recall and F1-score comparison
* ROC-AUC analysis
* Cross-validation
* Hyperparameter tuning
* GridSearchCV / RandomizedSearchCV
* Feature importance analysis
* Model performance comparison table
* ROC curves
* Confusion matrix visualization
* A complete ensemble model comparison
* Prediction interface for new observations

---

## ⚠️ Disclaimer

This project is intended for **educational and Machine Learning experimentation purposes only**.

The predictions generated by these models should **not be used as a substitute for professional medical diagnosis or clinical decision-making**.

---

## 👨‍💻 Author

**Ashutosh Kadu**

GitHub: [Ashu-k10](https://github.com/Ashu-k10)

---

## ⭐ Acknowledgement

This project was created as a practical implementation of **Supervised Machine Learning and Ensemble Learning techniques** using Python and Scikit-learn.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
