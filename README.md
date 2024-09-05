# GaIB-Classification-Clustering

## Table of Contents
- [Project Description](#project-description)
- [Requirements](#requirements)
- [Algorithms Implemented](#algorithms-implemented)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
- [Project Structure](#project-structure)
- [Author Information](#author-information)

## Project Description

This repository contains the submission for the Lab Assistant Selection for Lab Grafika dan Inteligensi Buatan ITB. The repository is divided into two sections:

1. **Design of Experiment + Supervised Learning**: 
   - The task focuses on **Heart Disease Classification Based on Patient Data**.
   - The workflow includes: Data Gathering, Exploratory Data Analysis, Data Preprocessing, Modeling, and Validation.
   - Five classification models are implemented and evaluated: KNN, Logistic Regression, Gaussian Naive Bayes, CART, and Random Forest.
   - These models are compared with their equivalent implementations using libraries, evaluated using **Hold-out validation** and **K-fold Cross-validation**, with **Accuracy** as the metric.

2. **Unsupervised Learning**: 
   - This task addresses the **Clustering of Wine Data**.
   - Three models are implemented: K-means, DBSCAN, and PCA.
   - These models are compared with their equivalent implementations from the **scikit-learn** library.

## Requirements

- Python 3.x
- Required Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn

Make sure to install the required libraries by running the following command:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Algorithms Implemented

### Supervised Learning

✅ KNN

✅ Logistic Regression

✅ Gaussian Naive Bayes

✅ CART

⬜ SVM

⬜ ANN

Bonus Implemented:
- Exponential Loss Function for Logistic Regression
- Newton's Method for Logistic Regression
- Bonus 1 (Ensemble Methods): Random Forest

---

### Unsupervised Learning

✅ K-MEANS

✅ DBSCAN

✅ PCA

Implemented Bonus:
- K-Means++ Initialization

## Project Structure
```bash
.
├── answers
│   ├── supervised-learning
│   │   ├── cart.pdf
│   │   ├── gaussian-naive-bayes.pdf
│   │   ├── knn.pdf
│   │   ├── logistic-regression.pdf
│   │   ├── modeling.pdf
│   │   ├── random-forest.pdf
│   ├── unsupervised-learning
│   │   ├── dbscan.pdf
│   │   ├── kmeans.pdf
│   │   ├── pca.pdf
├── dataset
│   ├── heart.csv
├── src
│   ├── supervised-learning
│   │   ├── cart.py
│   │   ├── gaussian_naive_bayes.py
│   │   ├── heart-disease-classification.ipynb
│   │   ├── knn.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   ├── unsupervised-learning
│   │   ├── dbscan.py
│   │   ├── kmeans.py
│   │   ├── pca.py
│   │   ├── wine-clusters.ipynb
├── .gitignore
├── README.md
```

## Author Information
- Name: Novelya Putri Ramadhani
- NIM: 13522096