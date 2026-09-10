# Autism-Prediction-ML
# 🧠 Autism Prediction Using Machine Learning

A Machine Learning project that aims to predict whether an individual is likely to have **Autism Spectrum Disorder (ASD)** based on a set of behavioral and demographic features.

The project explores different supervised Machine Learning algorithms, compares their performance, and evaluates the obtained predictions using different classification metrics.

> **Project Status:** Machine Learning model development and evaluation completed.
> **Deployment:** Not deployed yet.

---

## 📌 Project Overview

Autism Spectrum Disorder (ASD) is a developmental condition that can affect communication, behavior, and social interaction.

The goal of this project is to build and evaluate Machine Learning classification models that can assist in predicting ASD based on the available dataset features.

The project follows a typical Machine Learning workflow:

1. Data loading and exploration
2. Data preprocessing
3. Feature preparation
4. Training different Machine Learning models
5. Model prediction
6. Model evaluation
7. Comparing model performance

The complete implementation and experiments are available in the Jupyter Notebook included in this repository.

---

## ✨ Features

* Data exploration and preprocessing
* Handling and preparing dataset features
* Binary classification for ASD prediction
* Training multiple Machine Learning models
* Model performance comparison
* Evaluation using classification metrics
* Confusion Matrix analysis
* Accuracy evaluation

---

## 📊 Dataset

The project uses a dataset containing behavioral and demographic information related to Autism Spectrum Disorder.

The dataset includes features that can be used by Machine Learning models to identify patterns associated with ASD.

The target variable represents the classification outcome:

* **0** → No ASD indication
* **1** → ASD indication

The dataset is processed before training the models to make the features suitable for Machine Learning algorithms.

---

## 🤖 Machine Learning Models

Several classification algorithms were explored and evaluated in this project.

### 1. Decision Tree

A Decision Tree is a supervised Machine Learning algorithm that makes predictions by splitting the dataset based on feature values.

It was used as one of the baseline classification models for the project.

### 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction performance and reduce overfitting.

It was used to obtain a stronger classification model compared with a single Decision Tree.

### 3. XGBoost

XGBoost (Extreme Gradient Boosting) is an ensemble learning algorithm based on gradient boosting.

It builds trees sequentially, where each new tree focuses on improving the errors made by the previous trees.

XGBoost was also evaluated as part of the model comparison process.

---

## 📈 Model Evaluation

The trained models were evaluated using classification performance metrics.

The main evaluation methods used include:

### Accuracy

Accuracy measures the percentage of correctly classified samples.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:

* **TP** = True Positives
* **TN** = True Negatives
* **FP** = False Positives
* **FN** = False Negatives

### Confusion Matrix

The Confusion Matrix provides a detailed view of the model's predictions by showing:

* True Positives
* True Negatives
* False Positives
* False Negatives

It helps us understand not only how accurate the model is, but also what types of prediction errors it makes.

---

## 🏆 Results

One of the evaluated models achieved the following result:

### Random Forest

**Accuracy: 81.875%**

Confusion Matrix:

```text
[[108, 16],
 [ 13, 23]]
```

This means that on the evaluated test set, the model correctly classified the majority of the samples.

The project also includes **Decision Tree and XGBoost** models for comparison, allowing us to examine how different ensemble and tree-based approaches perform on the same classification problem.

> The reported accuracy represents the result obtained on the project's test data and should not be interpreted as a clinical diagnostic accuracy.

---

## 🔍 Project Workflow

```text
Dataset
   │
   ▼
Data Exploration
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Preparation
   │
   ▼
Train/Test Data
   │
   ├───────────────┐
   ▼               ▼
Decision Tree   Random Forest
   │               │
   └───────┬───────┘
           ▼
        XGBoost
           │
           ▼
     Model Evaluation
           │
           ▼
      Results Comparison
```

---

## 📁 Repository Structure

```text
Autism_Prediction_using_Machine_Learning/
│
├── Autism_Prediction_using_Machine_Learning.ipynb
│
├── README.md
│
└── Dataset/
    └── autism_dataset.csv
```

> The exact file structure may change as the project is further developed.

---

## 🛠️ Technologies Used

* **Python**
* **Jupyter Notebook**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* More detailed model comparison
* Feature importance analysis
* Improving model performance
* Adding a user-friendly interface
* Deploying the trained models as a web application
* Exploring additional Machine Learning and Deep Learning approaches

---

## 👥 Team Collaborators

This project was developed collaboratively by:

| # | Team Member                     |
| - | ------------------------------- |
| 1 | **Nour Ibrahim Salama**         |
| 2 | **Amira Mohamed Ibrahim**       |
| 3 | **Abdelwahab Mahmoud**          |
| 4 | **Salma Hassan**                |
| 5 |**Yousra Islam Fikry El-Saeed**  |
| 6 |  **Raymonda Ayoub Dawoud**      |

---

## 🎯 Project Goal

The main goal of this project is to apply Machine Learning concepts to a real-world classification problem and gain practical experience in:

* Data preprocessing
* Exploratory data analysis
* Supervised Machine Learning
* Ensemble Learning
* Model evaluation
* Comparing different classification algorithms

---

## ⚠️ Disclaimer

This project is developed for **educational and Machine Learning purposes**.

The predictions generated by the models should **not be considered a medical diagnosis** and should not replace assessment by qualified healthcare professionals.
