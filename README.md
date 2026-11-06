# ML Assignment 2 – Online Shoppers Purchasing Intention

**Name:** Sneha Ann Samuel | **ID:** 2026AJ05831

**Course:** M.Tech (AIML) – Machine Learning, Assignment 2

---

## a. Problem statement

An online store records how each visitor moves through the site during a session: which kinds of pages they open, how long they stay, how often they bounce or exit, and some context such as month, browser and traffic source. Only a small share of these sessions end in a purchase.

The task is to predict from this session data whether a session ends in a purchase (`Revenue = True`) or not (`Revenue = False`). This is a **binary classification** problem. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are then served through a Streamlit app that evaluates them on an uploaded test file.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (dataset id 468) – https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Original paper | Sakar, C.O., Polat, S.O., Katircioglu, M. & Kastro, Y. (2019), *Neural Computing and Applications*, 31, 6893–6908 |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicate rows) |
| Input features | 17 (10 numerical, 7 categorical) |
| Target | `Revenue` – True (purchase) / False (no purchase) |
| Class balance | 84.5% False, 15.5% True (imbalanced) |
| Missing values | None |

The sessions cover a one-year period, and each session belongs to a different user. This avoids bias towards a particular campaign, special day, user profile or period.

**Features**

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric | Number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | Total time (seconds) spent on each page type |
| BounceRates | numeric | Average bounce rate (from Google Analytics) of the pages visited |
| ExitRates | numeric | Average exit rate of the pages visited |
| PageValues | numeric | Average page value of the pages visited before a transaction |
| SpecialDay | numeric | Closeness of the visit date to a special day such as Mother's Day or Valentine's Day (0 to 1) |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer coded) | Visitor's OS, browser, region and traffic source |
| VisitorType | categorical | Returning_Visitor / New_Visitor / Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Preprocessing (stored inside every saved model pipeline)**
- Exact duplicate rows removed, then an 80/20 stratified train–test split (`random_state=42`): 9,764 rows for training and 2,441 rows for testing. The test part is saved as `test_data.csv`.
- The page count, duration and PageValues columns are strongly right-skewed (mostly zeros with a long tail), so `log1p` is applied before standard scaling.
- BounceRates, ExitRates and SpecialDay are standard scaled.
- OperatingSystems, Browser, Region and TrafficType are ID codes, not quantities, so they are one-hot encoded along with Month, VisitorType and Weekend (`handle_unknown="ignore"`).
- Hyperparameters were tuned with 5-fold stratified `GridSearchCV` on the training data only, using F1 as the scoring metric because of the class imbalance.

## c. GitHub Repository Link

https://github.com/SnEhA0AnN/ML-models

```
ML-models/
│-- app.py                  Streamlit application
│-- train_models.py         preprocessing, tuning, training, evaluation, saving
│-- requirements.txt
│-- README.md
│-- test_data.csv           20% hold-out test split (17 features + Revenue)
│-- data/
│     └── online_shoppers_intention.csv   original UCI file
└── model/
      ├── logistic_regression.joblib
      ├── decision_tree.joblib
      ├── knn.joblib
      ├── naive_bayes.joblib
      ├── svm.joblib
      ├── test_metrics.csv      metrics of all five models on test_data.csv
      └── model_info.json       feature list, file names, tuned hyperparameters
```

## d. Models used

All five models are trained on the same training split, and the metrics below are on the same hold-out test set (`test_data.csv`, 2,441 rows, 382 purchases). Precision, Recall and F1 are for the positive class (purchase). AUC uses the predicted probability of a purchase.

| Model | Tuned hyperparameters |
|---|---|
| Logistic Regression | C = 0.003, class_weight = balanced |
| Decision Tree | max_depth = 4, min_samples_leaf = 10, class_weight = None |
| KNN | n_neighbors = 31, weights = distance |
| Naive Bayes (Gaussian) | var_smoothing = 1.0 |
| Support Vector Machine (RBF) | C = 0.1, class_weight = balanced |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8816 | 0.9205 | 0.5903 | 0.7958 | **0.6778** | **0.6174** |
| Decision Tree | **0.9021** | **0.9237** | **0.7581** | 0.5497 | 0.6373 | 0.5924 |
| KNN | 0.8955 | 0.9143 | 0.7153 | 0.5524 | 0.6233 | 0.5702 |
| Naive Bayes | 0.8566 | 0.8720 | 0.5315 | 0.7068 | 0.6067 | 0.5291 |
| Support Vector Machine | 0.8746 | 0.9196 | 0.5699 | **0.8115** | 0.6695 | 0.6093 |

Confusion matrices on the test set (TN / FP / FN / TP): LR 1848 / 211 / 78 / 304, DT 1992 / 67 / 172 / 210, KNN 1975 / 84 / 171 / 211, NB 1821 / 238 / 112 / 270, SVM 1825 / 234 / 72 / 310.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best F1 (0.678) and best MCC (0.617). With `class_weight="balanced"` it catches about 80% of buyers (304 of 382) and keeps precision near 0.59. Strong regularisation (C = 0.003) worked best, which suggests the signal is mostly linear once the skewed columns are log-transformed. PageValues has by far the largest positive coefficient, followed by Month = Nov. Months like May and Mar and a high ExitRate push the prediction towards no purchase. |
| Decision Tree | Highest accuracy (0.902), AUC (0.924) and precision (0.758), but recall is only 0.55, so it misses 172 of the 382 buyers. It is a shallow tree (depth 4), and PageValues alone accounts for about 80% of the feature importance, with BounceRates and Month = Nov next. The high accuracy mostly comes from predicting the majority class well. Deeper trees overfit: going from depth 4 to an unlimited depth raises training F1 from 0.63 to 0.75 but lowers CV F1 from 0.644 to 0.601. |
| KNN | Similar behaviour to the Decision Tree: accuracy 0.896 and precision 0.715, but recall stays at 0.55, and it has the second-lowest F1 and MCC. Even with k = 31 and distance weighting, the one-hot columns add many dimensions that weaken the distance measure, and the minority class has fewer neighbours to vote for it. It also has to store all training rows, so it has the largest model file (~5.9 MB). |
| Naive Bayes | Weakest model on every metric except recall (0.707): F1 0.607, MCC 0.529, AUC 0.872. The independence assumption does not hold here. The page counts and their durations are strongly correlated, and the one-hot columns are binary rather than Gaussian. Heavy smoothing (var_smoothing = 1.0) was needed to stop it flagging too many sessions as purchases. With var_smoothing = 0.1, accuracy was only 0.751. |
| Support Vector Machine | Very close to Logistic Regression (F1 0.670, MCC 0.609, AUC 0.920) and gives the highest recall (0.812, 310 of 382 buyers). The best setting was a small C with balanced class weights, so the RBF kernel adds little over a linear boundary on this data. It is by far the most expensive model: about 27 s to fit, against under 0.1 s for the others, and about 1.2 s to score the test set. It also needs `probability=True` to output scores for AUC. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** With only 15.5% purchases, accuracy is misleading: a model that always predicts "no purchase" already scores 0.844. F1 and MCC are more reliable here, and Logistic Regression is best on both, with AUC (0.921) only about 0.003 below the Decision Tree. It is also the simplest and fastest model, and its coefficients are easy to interpret. SVM is a close second. If the store cared more about precision (for example, sending offers only to very likely buyers), the Decision Tree would be the better choice. |

## Streamlit app

**Live app:** https://ml-models-2026aj05831.streamlit.app

Features:
- Upload a test CSV (the 17 features + `Revenue`). `test_data.csv` can also be downloaded from the sidebar.
- Dropdown to choose any of the five saved models.
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- Confusion matrix and classification report.
- "Compare all models" tab with the metric table, a bar chart and ROC curves for all five models.
- Row-level predictions with the purchase probability, downloadable as CSV.

The app only loads the saved `.joblib` pipelines from `model/` and calls `predict` / `predict_proba`. Nothing is retrained on the uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py --data data/online_shoppers_intention.csv   # optional: rebuilds model/ and test_data.csv
streamlit run app.py
```
