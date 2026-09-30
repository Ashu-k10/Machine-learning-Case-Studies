# Customer Loan Approval Using Voting Classification

A Machine Learning project that predicts whether a customer's loan application should be **Approved** or **Rejected** using multiple classification algorithms and ensemble voting techniques.

## 📌 Project Overview

Banks handle large amounts of customer information while making loan approval decisions. This project demonstrates how Machine Learning can be used to automate the loan approval process.

The project uses three individual classification algorithms:

- **Logistic Regression**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**

These models are then combined using:

- **Hard Voting Classifier**
- **Soft Voting Classifier**

The performance of all models is compared using **Accuracy**.

---

## 🎯 Objective

The main objective is to predict the `LoanApproved` target variable.

| Value | Meaning |
|---|---|
| `0` | Loan Rejected |
| `1` | Loan Approved |

The models use customer information such as:

- Age
- Income
- Credit Score
- Existing Loan
- Employment Experience
- Loan Amount

---

## 🧠 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used for binary classification problems. It predicts the probability of a customer belonging to either the approved or rejected class.

### 2. Decision Tree

Decision Tree makes predictions by creating a series of decision rules based on the input features.

### 3. K-Nearest Neighbors

KNN classifies a customer based on the classes of its nearest data points.

### 4. Hard Voting

Hard Voting combines the predictions of all three models and selects the class receiving the majority of votes.

```text
Logistic Regression → 1
Decision Tree       → 0
KNN                 → 1

Majority Vote       → 1
```

### 5. Soft Voting

Soft Voting combines the predicted probabilities from the three models and selects the class with the highest average probability.

---

## 🔄 Project Workflow

```text
             Loan Dataset
                  |
                  ↓
          Check Missing Values
                  |
                  ↓
       Separate X and Y Variables
                  |
                  ↓
        Train / Test Split
                  |
          ┌───────┼────────┐
          ↓       ↓        ↓
      Logistic   Decision  KNN
     Regression   Tree
          |        |        |
          └────────┼────────┘
                   ↓
          Individual Accuracy
                   |
          ┌────────┴────────┐
          ↓                 ↓
     Hard Voting       Soft Voting
          |                 |
          └────────┬────────┘
                   ↓
          Accuracy Comparison
```

---

## 📂 Project Structure

```text
Customer-Loan-Approval-Voting/
│
├── loan_approval.csv
├── loan_approval.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Machine Learning**
- **Ensemble Learning**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-loan-approval-voting.git
```

Navigate to the project directory:

```bash
cd customer-loan-approval-voting
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

Create a `requirements.txt` file containing:

```text
pandas
scikit-learn
```

---

## ▶️ How to Run

Make sure the dataset is present in the project directory:

```text
loan_approval.csv
```

Then run:

```bash
python loan_approval.py
```

The program will:

1. Load the dataset.
2. Display the first five records.
3. Check for missing values.
4. Separate input and output variables.
5. Split the dataset into training and testing sets.
6. Train Logistic Regression.
7. Train Decision Tree.
8. Train KNN.
9. Calculate individual model accuracy.
10. Create a Hard Voting Classifier.
11. Create a Soft Voting Classifier.
12. Calculate voting classifier accuracy.
13. Compare all models.

---

## 📊 Model Comparison

The program generates an accuracy comparison similar to:

| Model | Accuracy |
|---|---:|
| Logistic Regression | XX.XX% |
| Decision Tree | XX.XX% |
| KNN | XX.XX% |
| Hard Voting | XX.XX% |
| Soft Voting | XX.XX% |

> **Note:** Accuracy values depend on the dataset and train/test split. The project calculates the actual values when the program is executed.

---

## ⚙️ Train-Test Split

The dataset is divided into:

```python
test_size=0.2
```

This means:

- **80%** → Training data
- **20%** → Testing data

A fixed random state is used:

```python
random_state=42
```

This makes the experiment reproducible.

---

## 📏 Feature Scaling

Feature scaling is applied to **Logistic Regression** and **KNN** using `StandardScaler`.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

and:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])
```

Scaling is particularly important for KNN because it relies on distances between data points.

---

## 🗳️ Why Voting Classification?

Different Machine Learning algorithms can make different predictions on the same data.

Voting Classification combines their predictions to produce a final prediction.

This project demonstrates two approaches:

### Hard Voting

Uses the **majority class prediction**.

### Soft Voting

Uses the **average predicted probabilities**.

Using multiple models provides an example of **ensemble learning**, where multiple classifiers are combined into a single classification system.

---

## 📈 Evaluation Metric

The project uses **Accuracy**:

```python
accuracy_score(Y_test, Y_pred)
```

Accuracy is calculated as:

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

---

## 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Confusion matrix
- Precision, Recall and F1-score
- ROC-AUC evaluation
- Feature importance analysis
- Handling class imbalance
- GridSearchCV / RandomizedSearchCV
- Adding Random Forest and SVM
- Deploying the model using Flask or FastAPI
- Creating a web interface for loan prediction

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**. Real-world loan approval systems require careful validation, fairness analysis, regulatory compliance, explainability, privacy protection, and appropriate human oversight.

---

## 👨‍💻 Author

**Ashutosh Kadu**

Machine Learning / Python Project

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
