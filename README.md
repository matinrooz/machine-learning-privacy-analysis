# machine-learning-privacy-analysis

**What:** Group project: a blood pressure model trained on health data, protected with k-anonymity, l-diversity, t-closeness and differential privacy, then attacked.  
**Result:** None of the methods gave strong protection against the attacks; t-closeness kept the predictions closest to the original model.  
**Stack:** Python, TensorFlow/Keras, PyTorch, scikit-learn

Group project with Miroslava Macejkova and Rakhmatillokhon Khoshimov (June 2024). We trained
a neural network that predicts blood pressure from health data, protected the data or the
predictions with different privacy methods, and then attacked the model to see how much
the methods actually protect.

The full report is `Report_SecuringML_Models.pdf`.

## Data and model

- `smoking_health_data_final.csv`: age, sex, smoker, heart rate, cigarettes per day,
  cholesterol and blood pressure (e.g. "130/80"). `test_and_train_data_generation.ipynb`
  cleans it and splits it into `pproject_train.csv` and `pproject_test.csv`.
- Model: Keras MLP (128-64-32), 7 input features, 2 outputs (systolic and diastolic),
  trained for 50 epochs.

## What we tried

**Anonymizing the training data** (`k-Anonymity, l-Diversity, m-Invariance, and t-Closeness.ipynb`):
k-anonymity (k = 3), l-diversity (l = 2), m-invariance (m = 2), t-closeness (t = 0.2,
Wasserstein distance). Then the same network was trained on each version.

**Differential privacy on the predictions** (`DifferentialPrivacy.ipynb`): Laplace noise added
to the model output, with ε = 0.05, 0.1 and 1.

**Attacks**: a reconstruction attack (a second network tries to recover the true values from
the predictions) and a model inversion attack.

## Results

For one person with a true blood pressure of 110/72, the model trained on the original data
predicted 117/72. Trained on the anonymized versions it predicted 118/75 (t-closeness),
106/68 (m-invariance) and 102/64 (l-diversity). With k-anonymity it predicted 663/420, so
something clearly broke there.

In the inversion attack, a lower error means the attacker got closer to the real values. The
mean absolute error was 11.5 (systolic) and 6.9 (diastolic) for t-closeness, 16.9 and 10.3
for m-invariance, 17.0 and 10.4 for l-diversity, and 18.1 and 10.8 for k-anonymity.

In the reconstruction attack on the differentially private predictions, a higher error means
better privacy. Only ε = 0.1 made a clear difference: the error went from 320.3 to 515.5 for
systolic and from 103.8 to 311.8 for diastolic. With ε = 1 it barely moved (320.3 to 325.9,
104.0 to 105.1), and with ε = 0.05 the noisy predictions were even easier to reconstruct
(320.2 to 276.8, 103.8 to 84.5).

## What we found

- t-closeness kept the predictions closest to the original model. k-anonymity broke the
  model completely (predictions like 663/420), which points to a problem in how we encoded
  the generalized data, not only to a privacy cost.
- In the report we called t-closeness the best method overall. Looking at it again, its
  inversion error is the lowest, which means the attacker was most successful on it. So it
  gives the best utility, but not the best privacy.
- With DP, only ε = 0.1 made the reconstruction clearly harder. With ε = 0.05 the noisy
  predictions were even easier to reconstruct, which is a random effect of the noise with a
  small test set.
- In the inversion attack the error was almost the same with and without noise for all ε
  (about 15,100 for systolic and 5,260 for diastolic). The noise on the outputs did not stop
  the attack.

Our overall conclusion in the report: with these settings, none of the methods gave strong
protection, the data stayed vulnerable to the attacks in most cases.

## Run

Python 3 with pandas, numpy, scikit-learn, scipy, matplotlib, TensorFlow and PyTorch. Open
the notebooks in this order:

1. `test_and_train_data_generation.ipynb`
2. `k-Anonymity, l-Diversity, m-Invariance, and t-Closeness.ipynb`
3. `DifferentialPrivacy.ipynb`
