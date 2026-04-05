# 🛡️ Machine Learning Privacy Analysis

## Overview

This project explores privacy-preserving techniques applied to machine learning models trained on sensitive healthcare data. The goal is to evaluate the trade-off between data privacy and model performance.

---

## Methods

### Privacy-Preserving Techniques

The following anonymization techniques were applied:

* k-Anonymity (k = 3)
* l-Diversity (l = 2)
* m-Invariance (m = 2)
* t-Closeness (t = 0.2)

These methods transform the dataset to reduce the risk of identity and attribute disclosure while preserving data utility.

---

### Neural Network Model

A standard neural network model was used to predict:

* Systolic blood pressure
* Diastolic blood pressure

Training details:

* 50 epochs
* Evaluation metric: Mean Absolute Error (MAE)

---

### Differential Privacy

Differential Privacy (DP) was applied to model predictions using different epsilon values:

| Epsilon | Privacy Level | Accuracy |
| ------- | ------------- | -------- |
| 0.05    | High          | Low      |
| 0.1     | Medium        | Medium   |
| 1       | Low           | High     |

Lower epsilon values provide stronger privacy but introduce more noise.

---

## Privacy Attacks

The robustness of the system was evaluated against:

* Reconstruction attacks
* Model inversion attacks

These attacks attempt to recover sensitive information from model outputs.

---

## Key Findings

* k-Anonymity provides strong privacy but significantly reduces accuracy
* l-Diversity improves the balance between privacy and utility
* t-Closeness achieves the best trade-off between privacy and performance
* Differential Privacy:

  * Low epsilon → strong privacy, lower accuracy
  * High epsilon → weaker privacy, higher accuracy

---

## Project Structure

```
.
├── DifferentialPrivacy.ipynb
├── test_and_train_data_generation.ipynb
├── pproject_train.csv
├── pproject_test.csv
├── smoking_health_data_final.csv
├── Report_SecuringML_Models.pdf
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/machine-learning-privacy-analysis.git
cd machine-learning-privacy-analysis
jupyter notebook
```

---

## Technologies

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-learn
* Matplotlib

---

## Report

The full report is included in the repository:

**Report_SecuringML_Models.pdf**

It contains detailed methodology, experiments, and analysis of privacy attacks.

---

## Authors

* Matin Roozbahani
* Miroslava Macejkova
* Rakhmatillokhon Khoshimov

University of Basel

---

## Notes

This project highlights the fundamental trade-off between privacy and model performance in real-world machine learning systems working with sensitive data.

