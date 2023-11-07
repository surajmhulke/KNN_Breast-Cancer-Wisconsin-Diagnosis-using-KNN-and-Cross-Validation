# KNN_Breast-Cancer-Wisconsin-Diagnosis-using-KNN-and-Cross-Validation
 

 

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction
In this project, we aim to classify breast cancer tumors as either malignant (M) or benign (B) using the K-Nearest Neighbors (KNN) algorithm. KNN is a simple yet powerful classification algorithm that makes predictions based on the majority class among its k-nearest neighbors. We will go through each step of the process, from data loading to model evaluation.

## Importing Libraries
We start by importing the necessary libraries for our project. These libraries include NumPy for numerical operations, pandas for data manipulation, and Matplotlib for data visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Importing Dataset
We load the breast cancer dataset from a CSV file. The dataset contains various features describing tumor characteristics, including radius, texture, and smoothness.

```python
df = pd.read_csv("breast-cancer-wisconsin-data/data.csv")
```

## Exploratory Data Analysis (EDA)
To better understand the dataset, we can perform exploratory data analysis. This includes checking the data information, dropping irrelevant columns (e.g., 'id' and 'Unnamed: 32'), and converting the 'diagnosis' column to numerical values.

```python
# Checking data information
df.info()

# Dropping irrelevant columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Converting 'diagnosis' to numerical values (Malignant = 1, Benign = 0)
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
```

## Feature Engineering
Visualizing the data can help us gain insights into potential relationships between features and the target variable. We create scatter plots to visualize feature combinations for diagnosis (M or B).

## Model Development and Evaluation
After preprocessing the data, we proceed to model development using the KNN algorithm. We split the data into training and testing sets, initialize the KNN classifier with a specific number of neighbors (e.g., 13), and fit the model with the training data. We calculate the accuracy score of the model using the testing data.

```python
# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X = df.iloc[:, 1:]
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Creating and training the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

# Model evaluation
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)
```

## Conclusion
In this project, we utilized the K-Nearest Neighbors algorithm to classify breast cancer tumors as malignant or benign. After thorough data analysis, preprocessing, and model development, we achieved an accuracy score of approximately 96.28%. This classification model can be useful for predicting the malignancy of breast tumors based on their features.

 
