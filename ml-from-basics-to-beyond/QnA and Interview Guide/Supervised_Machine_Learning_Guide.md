# The Complete Supervised Machine Learning Guide
### Linear Regression · Logistic Regression · KNN · SVM · Naive Bayes · Decision Tree · Random Forest · XGBoost
#### Algorithms → Intuition → Formulae → Hyperparameters → Cross-Validation → Evaluation Metrics → Interview Q&A

---

> 🏥 **One real-world anchor used throughout this entire guide:**
> A hospital has data on **10,000 patients** — Age, BMI, Blood Pressure, Glucose, Insulin, Smoking_Status, Family_History, Diet_Quality.
>
> | Question | Task | Algorithm(s) |
> |---|---|---|
> | *"What will this patient's exact Blood Sugar level be?"* | Regression | Linear Regression, Decision Tree, RF, XGBoost |
> | *"Will this patient develop Diabetes — Yes or No?"* | Classification | All other algorithms |
>
> Every concept in this guide — loss functions, kernels, entropy, margins, residuals, confusion matrices — connects back to one of these two questions on the same dataset.

---

## Table of Contents

- [The Complete Supervised Machine Learning Guide](#the-complete-supervised-machine-learning-guide)
    - [Linear Regression · Logistic Regression · KNN · SVM · Naive Bayes · Decision Tree · Random Forest · XGBoost](#linear-regression--logistic-regression--knn--svm--naive-bayes--decision-tree--random-forest--xgboost)
      - [Algorithms → Intuition → Formulae → Hyperparameters → Cross-Validation → Evaluation Metrics → Interview Q\&A](#algorithms--intuition--formulae--hyperparameters--cross-validation--evaluation-metrics--interview-qa)
  - [Table of Contents](#table-of-contents)
- [PART 0 — The Golden Rule \& Foundation](#part-0--the-golden-rule--foundation)
  - [0.1 Split Your Data First](#01-split-your-data-first)
  - [0.2 Feature Scaling — Which Algorithms Need It](#02-feature-scaling--which-algorithms-need-it)
  - [0.3 One-Line Mental Models for All Algorithms](#03-one-line-mental-models-for-all-algorithms)
- [PART 1 — Linear Regression](#part-1--linear-regression)
  - [1.1 What the Model Learns](#11-what-the-model-learns)
  - [1.2 Error → Loss → Cost Function: The Hierarchy](#12-error--loss--cost-function-the-hierarchy)
  - [1.3 Gradient Descent — The Learning Engine](#13-gradient-descent--the-learning-engine)
  - [1.4 Regularisation — Preventing Overfitting](#14-regularisation--preventing-overfitting)
  - [1.5 Evaluation Metrics for Regression](#15-evaluation-metrics-for-regression)
- [PART 2 — Logistic Regression](#part-2--logistic-regression)
  - [2.1 The Sigmoid Function](#21-the-sigmoid-function)
  - [2.2 Log Loss — The Cost Function](#22-log-loss--the-cost-function)
  - [2.3 Hyperparameter C (Regularisation)](#23-hyperparameter-c-regularisation)
- [PART 3 — K-Nearest Neighbors (KNN)](#part-3--k-nearest-neighbors-knn)
  - [3.1 Core Mechanism — No Training, Pure Memory](#31-core-mechanism--no-training-pure-memory)
  - [3.2 Distance Formulae](#32-distance-formulae)
  - [3.3 Hyperparameter K](#33-hyperparameter-k)
- [PART 4 — Support Vector Machine (SVM)](#part-4--support-vector-machine-svm)
  - [4.1 The Margin — Core Geometry](#41-the-margin--core-geometry)
  - [4.2 Hard Margin vs. Soft Margin](#42-hard-margin-vs-soft-margin)
  - [4.3 The Kernel Trick — SVM's Superpower](#43-the-kernel-trick--svms-superpower)
  - [4.4 Hyperparameters C and Gamma — Must Be Tuned Together](#44-hyperparameters-c-and-gamma--must-be-tuned-together)
  - [4.5 LR vs. SVM — They Both Find a Linear Boundary. What's Different?](#45-lr-vs-svm--they-both-find-a-linear-boundary-whats-different)
- [PART 5 — Naive Bayes](#part-5--naive-bayes)
  - [5.1 Bayes' Theorem — The Foundation](#51-bayes-theorem--the-foundation)
  - [5.2 The Naive Assumption](#52-the-naive-assumption)
  - [5.3 Three Variants](#53-three-variants)
    - [Variant 1 — Gaussian Naive Bayes (for continuous features)](#variant-1--gaussian-naive-bayes-for-continuous-features)
    - [Variant 2 — Multinomial Naive Bayes (for count/frequency features)](#variant-2--multinomial-naive-bayes-for-countfrequency-features)
    - [Variant 3 — Bernoulli Naive Bayes (for binary features)](#variant-3--bernoulli-naive-bayes-for-binary-features)
  - [5.4 Laplace Smoothing — Preventing Zero Probabilities](#54-laplace-smoothing--preventing-zero-probabilities)
- [PART 6 — Decision Tree](#part-6--decision-tree)
  - [6.1 How the Tree Learns — Splitting Criteria](#61-how-the-tree-learns--splitting-criteria)
  - [6.2 Gini, Entropy, and Variance Reduction](#62-gini-entropy-and-variance-reduction)
    - [Gini Index (sklearn default)](#gini-index-sklearn-default)
    - [Entropy and Information Gain](#entropy-and-information-gain)
    - [Variance Reduction (for Regression Trees)](#variance-reduction-for-regression-trees)
  - [6.3 Tree Building — Full Algorithm (CART)](#63-tree-building--full-algorithm-cart)
  - [6.4 Hyperparameters — The Pruning Dials](#64-hyperparameters--the-pruning-dials)
- [PART 7 — Random Forest](#part-7--random-forest)
  - [7.1 Two Randomisation Tricks — The Heart of Random Forest](#71-two-randomisation-tricks--the-heart-of-random-forest)
    - [Trick 1 — Bootstrap Sampling (Bagging)](#trick-1--bootstrap-sampling-bagging)
    - [Trick 2 — Feature Subsampling (Random Subspace Method)](#trick-2--feature-subsampling-random-subspace-method)
  - [7.2 Why Averaging Reduces Variance — The Math](#72-why-averaging-reduces-variance--the-math)
  - [7.3 Hyperparameters](#73-hyperparameters)
  - [7.4 Feature Importance](#74-feature-importance)
- [PART 8 — XGBoost (Extreme Gradient Boosting)](#part-8--xgboost-extreme-gradient-boosting)
  - [8.1 Gradient Boosting — Core Mechanism](#81-gradient-boosting--core-mechanism)
    - [Step-by-Step Walkthrough](#step-by-step-walkthrough)
  - [8.2 XGBoost Objective Function](#82-xgboost-objective-function)
  - [8.3 Key Hyperparameters](#83-key-hyperparameters)
  - [8.4 XGBoost vs. Random Forest — The Key Differences](#84-xgboost-vs-random-forest--the-key-differences)
  - [8.5 LightGBM and CatBoost — Brief Mentions](#85-lightgbm-and-catboost--brief-mentions)
- [PART 9 — Hyperparameter Tuning \& Cross-Validation](#part-9--hyperparameter-tuning--cross-validation)
  - [9.1 Why Default Models Fail](#91-why-default-models-fail)
  - [9.2 K-Fold Cross-Validation Mechanics](#92-k-fold-cross-validation-mechanics)
  - [9.3 Grid Search, Random Search, Bayesian Optimisation](#93-grid-search-random-search-bayesian-optimisation)
    - [Grid Search](#grid-search)
    - [Random Search](#random-search)
    - [Bayesian Optimisation (optuna)](#bayesian-optimisation-optuna)
  - [9.4 Complete Tuning Pipeline](#94-complete-tuning-pipeline)
- [PART 10 — Bias, Variance \& The Tradeoff](#part-10--bias-variance--the-tradeoff)
  - [10.1 Definitions](#101-definitions)
  - [10.2 The 2-Number Diagnostic Test](#102-the-2-number-diagnostic-test)
  - [10.3 Where Every Algorithm Sits on the Spectrum](#103-where-every-algorithm-sits-on-the-spectrum)
- [PART 11 — Evaluation Metrics: Complete Guide](#part-11--evaluation-metrics-complete-guide)
  - [11.1 The Confusion Matrix — Universal Convergence Point](#111-the-confusion-matrix--universal-convergence-point)
  - [11.2 All Classification Metrics — Formulae and Meaning](#112-all-classification-metrics--formulae-and-meaning)
    - [Accuracy](#accuracy)
    - [Precision](#precision)
    - [Recall (Sensitivity / True Positive Rate)](#recall-sensitivity--true-positive-rate)
    - [The Precision-Recall Tradeoff](#the-precision-recall-tradeoff)
    - [F1 Score](#f1-score)
    - [Specificity (True Negative Rate)](#specificity-true-negative-rate)
    - [AUC-ROC](#auc-roc)
    - [Log Loss](#log-loss)
  - [11.3 Regression Metrics — Side-by-Side](#113-regression-metrics--side-by-side)
  - [11.4 Choosing the Right Metric — Decision Guide](#114-choosing-the-right-metric--decision-guide)
  - [11.5 Domain Cheat Sheet](#115-domain-cheat-sheet)
- [PART 12 — Ensemble Methods — The Big Picture](#part-12--ensemble-methods--the-big-picture)
  - [The Three Ensemble Strategies](#the-three-ensemble-strategies)
- [PART 13 — Master Comparison Tables](#part-13--master-comparison-tables)
  - [Algorithm Properties at a Glance](#algorithm-properties-at-a-glance)
  - [Performance on Our Diabetes Dataset](#performance-on-our-diabetes-dataset)
- [PART 14 — Algorithm Selection Decision Guide](#part-14--algorithm-selection-decision-guide)
  - [Step-by-Step Framework](#step-by-step-framework)
  - [When Each Algorithm Wins](#when-each-algorithm-wins)
  - [The Practical Hierarchy (What Practitioners Actually Do)](#the-practical-hierarchy-what-practitioners-actually-do)
- [PART 15 — Quick Revision: All Formulae in One Place](#part-15--quick-revision-all-formulae-in-one-place)
  - [Model Formulae](#model-formulae)
  - [Evaluation Metrics](#evaluation-metrics)
- [PART 16 — Complete Interview Q\&A Bank](#part-16--complete-interview-qa-bank)
    - [Q1. What is the difference between Loss and Cost Function?](#q1-what-is-the-difference-between-loss-and-cost-function)
    - [Q2. Why can't we use MSE as the cost function for Logistic Regression?](#q2-why-cant-we-use-mse-as-the-cost-function-for-logistic-regression)
    - [Q3. Why is Feature Scaling important for KNN and SVM but not for Decision Trees?](#q3-why-is-feature-scaling-important-for-knn-and-svm-but-not-for-decision-trees)
    - [Q4. What are Support Vectors?](#q4-what-are-support-vectors)
    - [Q5. What does the Kernel Trick do in SVM?](#q5-what-does-the-kernel-trick-do-in-svm)
    - [Q6. Hard Margin vs. Soft Margin SVM — what is the difference?](#q6-hard-margin-vs-soft-margin-svm--what-is-the-difference)
    - [Q7. Why does Cross-Validation beat a single train/validation split?](#q7-why-does-cross-validation-beat-a-single-trainvalidation-split)
    - [Q8. In medical screening, should you optimise Precision or Recall? Why?](#q8-in-medical-screening-should-you-optimise-precision-or-recall-why)
    - [Q9. What is the Bias-Variance Tradeoff?](#q9-what-is-the-bias-variance-tradeoff)
    - [Q10. LR and SVM both find a linear boundary. What is the fundamental difference?](#q10-lr-and-svm-both-find-a-linear-boundary-what-is-the-fundamental-difference)
    - [Q11. What is the "Naive" assumption in Naive Bayes, and why does it still work?](#q11-what-is-the-naive-assumption-in-naive-bayes-and-why-does-it-still-work)
    - [Q12. What is Laplace Smoothing and why is it needed?](#q12-what-is-laplace-smoothing-and-why-is-it-needed)
    - [Q13. What is Information Gain and why is it used in Decision Trees?](#q13-what-is-information-gain-and-why-is-it-used-in-decision-trees)
    - [Q14. Why does a single Decision Tree have high variance, and how does Random Forest fix it?](#q14-why-does-a-single-decision-tree-have-high-variance-and-how-does-random-forest-fix-it)
    - [Q15. What is Out-Of-Bag error?](#q15-what-is-out-of-bag-error)
    - [Q16. Explain Gradient Boosting in simple terms.](#q16-explain-gradient-boosting-in-simple-terms)
    - [Q17. What is the difference between Random Forest and XGBoost?](#q17-what-is-the-difference-between-random-forest-and-xgboost)
    - [Q18. What is the role of the learning rate in XGBoost, and how does it interact with n\_estimators?](#q18-what-is-the-role-of-the-learning-rate-in-xgboost-and-how-does-it-interact-with-n_estimators)
    - [Q19. What is the difference between bagging and boosting?](#q19-what-is-the-difference-between-bagging-and-boosting)
    - [Q20. When would you choose Naive Bayes over Random Forest?](#q20-when-would-you-choose-naive-bayes-over-random-forest)
    - [Q21. Why is AUC-ROC better than Accuracy for model comparison?](#q21-why-is-auc-roc-better-than-accuracy-for-model-comparison)
    - [Q22. Quick-fire: Why these design choices?](#q22-quick-fire-why-these-design-choices)
    - [Q23. The One Paragraph That Ties Everything Together](#q23-the-one-paragraph-that-ties-everything-together)

---

<a name="part-0"></a>

# PART 0 — The Golden Rule & Foundation

<a name="01-split-your-data-first"></a>

## 0.1 Split Your Data First

Before any algorithm sees any data, before any hyperparameter is chosen — you do one thing:

```
10,000 Patients
│
├── 8,000 patients  →  Training Pool  (all learning happens here)
│
└── 2,000 patients  →  Test Set 🔒   (locked vault — opened exactly once at the very end)
```

**Why non-negotiable?** The test set simulates patients the hospital has *never seen before*. The moment you use it to make any decision — even once — it is no longer unseen data. Every tuning decision, every model comparison, every hyperparameter choice must be made using only the 8,000 training patients. The test set is opened **once**, at the very end, to report the final honest performance.

> ⚠️ **The Rule:** Fit everything (scalers, imputers, encoders, models) on training data only. Apply to test data. Never the reverse. Violating this is called **data leakage** — it gives you artificially inflated test scores that won't hold in production.

---

<a name="02-feature-scaling"></a>

## 0.2 Feature Scaling — Which Algorithms Need It

**The problem:** Age ranges 20–80. Insulin ranges 0–800. Algorithms that use distance or gradient steps are dominated by the large-magnitude feature.

**Standardisation (Z-score):** Transforms every feature to mean=0, std=1.

```
x_scaled  =  (x  -  mean(x))  /  std(x)
```

**Min-Max Normalisation:** Transforms every feature to range [0, 1].

```
x_scaled  =  (x  -  x_min)  /  (x_max  -  x_min)
```

| Algorithm | Needs Scaling? | Reason |
|---|---|---|
| Linear Regression | ✅ Yes | Gradient steps proportional to feature magnitude |
| Logistic Regression | ✅ Yes | Same — gradient descent driven |
| KNN | ✅ Yes | Distance calculations dominated by large-scale features |
| SVM | ✅ Yes | Margin and kernel computations affected |
| Naive Bayes | ❌ No | Works with probabilities — scale doesn't affect conditional distributions |
| Decision Tree | ❌ No | Threshold-based splits — only relative ordering matters |
| Random Forest | ❌ No | Ensemble of trees — same reasoning |
| XGBoost | ❌ No | Tree-based — split thresholds are scale-invariant |

> 💡 **Always fit the scaler on training data only.** Apply the same fitted scaler to test data. Never fit on test data.

---

<a name="03-one-line-mental-models"></a>

## 0.3 One-Line Mental Models for All Algorithms

> Memorise these first. They anchor everything else.

| Algorithm | One-Line Mental Model |
|---|---|
| **Linear Regression** | *"Draw the best-fit straight line through the data."* |
| **Logistic Regression** | *"Draw the best straight boundary, then convert distance to probability."* |
| **KNN** | *"Ask your K nearest neighbours to vote."* |
| **SVM** | *"Draw the boundary with the maximum breathing room from both sides."* |
| **Naive Bayes** | *"Use Bayes' theorem — assume all features are independent, multiply probabilities."* |
| **Decision Tree** | *"Play 20 questions — ask yes/no questions about features until you reach an answer."* |
| **Random Forest** | *"Ask 100 different decision trees and take a majority vote."* |
| **XGBoost** | *"Build trees one by one, each one fixing the mistakes of the previous."* |

---

<a name="part-1"></a>

# PART 1 — Linear Regression

> **Task:** Predict the patient's exact Blood Sugar level — a continuous number like 142 mg/dL.

<a name="11-what-the-model-learns"></a>

## 1.1 What the Model Learns

Linear Regression fits a weighted sum through the data:

```
Blood_Sugar  =  w1*Age  +  w2*BMI  +  w3*BP  +  w4*Insulin  +  b
```

| Symbol | Meaning | Learned how? |
|---|---|---|
| `w1, w2, ...` | Weights — how much each feature contributes | Gradient Descent |
| `b` | Bias / intercept | Gradient Descent |
| `Blood_Sugar` | Predicted output (y_hat) | Output of formula |

---

<a name="12-error--loss--cost-function"></a>

## 1.2 Error → Loss → Cost Function: The Hierarchy

**Error (one patient):**

```
Error_i  =  y_hat_i  -  y_i
```

Example: Predict 167 mg/dL, actual is 163 mg/dL → Error = +4.

> If you average raw errors, positives and negatives cancel — misleadingly near-zero even when predictions are way off. This is why we need proper loss functions.

**Loss functions (how one prediction is penalised):**

| Loss | Formula | Intuition | Outlier Sensitive? |
|---|---|---|---|
| **MAE** | `Mean( |y_hat - y| )` | Average absolute gap | No — all errors treated equally |
| **MSE** | `Mean( (y_hat - y)^2 )` | Squares large errors — punishes harder | Yes — one huge error dominates |
| **RMSE** | `sqrt( MSE )` | Same units as target — interpretable | Yes |

**Cost Function J(w, b) — what the model minimises:**

```
J(w, b)  =  (1/n)  x  Sum[ (y_hat_i  -  y_i)^2 ]      ← MSE Cost
```

> 💡 **The hierarchy:** Error = mistake on 1 patient. Loss = how that mistake is penalised. Cost = average penalty across all 8,000 patients. Gradient Descent minimises the Cost.

---

<a name="13-gradient-descent"></a>

## 1.3 Gradient Descent — The Learning Engine

Gradient Descent finds weights `w` that make `J(w, b)` as small as possible.

**Intuition:** Blindfolded on a hilly landscape. `J` is your altitude. You feel the slope under your feet and take one small step downhill. Repeat until you reach the valley.

**Update rule:**

```
w  :=  w  -  alpha  x  (dJ/dw)
```

**The gradient:**

```
dJ/dw  =  (2/n)  x  Sum[ (y_hat_i - y_i)  x  x_i ]
```

**`alpha` is the learning rate:**

| alpha | Effect | Problem |
|---|---|---|
| Too large (0.9) | Giant steps | Overshoots minimum — cost oscillates or explodes |
| Too small (0.000001) | Tiny steps | Converges painfully slowly |
| Just right (0.01) | Smooth descent | Converges efficiently ✅ |

---

<a name="14-regularisation"></a>

## 1.4 Regularisation — Preventing Overfitting

Without regularisation, weights grow huge to memorise training patients, then fail on new ones.

| Type | Formula | Effect | Use When |
|---|---|---|---|
| **Ridge (L2)** | `J = MSE + lambda x Sum(w_i^2)` | Shrinks all weights toward zero | All features matter a little |
| **Lasso (L1)** | `J = MSE + lambda x Sum(|w_i|)` | Drives some weights to exactly zero | Many irrelevant features — built-in feature selection |
| **ElasticNet** | `J = MSE + lambda1 x Sum(|w_i|) + lambda2 x Sum(w_i^2)` | Mix of L1 and L2 | High-dimensional sparse data |

| lambda | Behaviour |
|---|---|
| Too large | Weights forced to ~0 → **Underfitting (High Bias)** |
| Too small | Weights grow huge → **Overfitting (High Variance)** |
| Just right | Balanced → Generalises ✅ |

---

<a name="15-evaluation-metrics-for-regression"></a>

## 1.5 Evaluation Metrics for Regression

After opening the test vault with the final model:

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | `Mean( |y_hat - y| )` | On average, off by X mg/dL |
| RMSE | `sqrt( Mean( (y_hat-y)^2 ) )` | Large errors penalised more — dangerous predictions surfaced |
| R² | `1  -  SS_res / SS_tot` | Fraction of variance the model explains (1.0 = perfect, 0.0 = useless) |

**Interpreting R²:**

| R² | Meaning |
|---|---|
| 1.0 | Perfect |
| 0.87 | Excellent — explains 87% of variance |
| 0.5 | Moderate |
| 0.0 | No better than predicting the mean for everyone |
| < 0 | Actively worse than the mean — something is very wrong |

> ❌ No confusion matrix for regression — the output is a continuous number, not a class label.

---

<a name="part-2"></a>

# PART 2 — Logistic Regression

> **Task:** Classify patient as Diabetic (1) or Not Diabetic (0).
> **Philosophy:** *"What is the probability of this patient belonging to the Diabetic class?"*

<a name="21-the-sigmoid-function"></a>

## 2.1 The Sigmoid Function

Linear Regression can predict −0.3 or 1.7 — meaningless for a probability. The sigmoid squashes any real number into (0, 1):

```
P(Diabetic=1 | x)  =  1  /  (1  +  e^(-z))       where  z = w^T * x + b
```

- If `P >= 0.5` → predict **Diabetic**
- If `P < 0.5` → predict **Not Diabetic**

**Decision Boundary:** Where `P = 0.5` → `z = 0` → `w^T*x + b = 0` (a linear hyperplane separating the two classes).

---

<a name="22-log-loss--the-cost-function"></a>

## 2.2 Log Loss — The Cost Function

With MSE + sigmoid, the cost surface becomes **non-convex** — full of local minima. Gradient Descent can get stuck. Log Loss gives a **convex surface**, guaranteeing convergence to the global minimum.

**Loss for one patient:**

```
Loss  =  -[ y  x  log(p)  +  (1-y)  x  log(1-p) ]
```

**Cost over all n patients:**

```
J  =  -(1/n)  x  Sum[ y_i  x  log(p_i)  +  (1-y_i)  x  log(1-p_i) ]
```

| Situation | Loss |
|---|---|
| Patient IS diabetic (y=1), model says p=0.95 | Very small ✅ |
| Patient IS diabetic (y=1), model says p=0.50 | Medium |
| Patient IS diabetic (y=1), model says p=0.05 | Very large ❌ |

> 💡 Gradient Descent still minimises this cost — **same engine as Linear Regression, different loss formula**. The update rule is identical; only the gradient changes.

---

<a name="23-hyperparameter-c"></a>

## 2.3 Hyperparameter C (Regularisation)

In scikit-learn, `C = 1/lambda` (inverted convention):

```
Cost  =  Log_Loss  +  (1/C)  x  ||w||^2
```

| C | Regularisation | Boundary | Risk |
|---|---|---|---|
| High C (e.g. 100) | Weak | Complex | Overfitting |
| Low C (e.g. 0.001) | Strong | Simple | Underfitting |
| C = 1.0 (default) | Moderate | Balanced | Often sweet spot ✅ |

**Nature: Probabilistic · Global · Parametric**
- **Global** — every training patient influences the weights
- **Probabilistic** — gives calibrated probabilities, not just labels
- Foundation of the output layer in neural networks (softmax = multinomial logistic regression)

---

<a name="part-3"></a>

# PART 3 — K-Nearest Neighbors (KNN)

> **Philosophy:** *"What do this patient's nearest neighbours look like — what class are they?"*

<a name="31-core-mechanism"></a>

## 3.1 Core Mechanism — No Training, Pure Memory

KNN is a **lazy learner**. It stores all 8,000 training patients and does nothing during training. At inference time:

1. Compute distance to all 8,000 training patients
2. Find the K closest ones
3. Take a majority vote

```
y_hat  =  mode { y_i : x_i in N_K(x) }
```

There is **no explicit decision boundary** learned — it emerges as a Voronoi-like partition of feature space.

> 💡 **KNN has NO cost function and NO gradient descent** — a frequent exam question. There are no weights to optimise. Training cost: O(1) — just store data. Inference cost: O(n × d) per query.

---

<a name="32-distance-formulae"></a>

## 3.2 Distance Formulae

| Distance | Formula | When to use |
|---|---|---|
| Euclidean (L2) | `sqrt( Sum( (x_i - x_j)^2 ) )` | Default. Continuous features, similar scales. |
| Manhattan (L1) | `Sum( |x_i - x_j| )` | When outliers are present. Less sensitive to extremes. |
| Minkowski | `( Sum( |x_i - x_j|^p ) )^(1/p)` | Generalisation: p=1 → Manhattan, p=2 → Euclidean. |

> ⚠️ **WHY SCALING IS CRITICAL:** If Age is 0–80 and Insulin is 0–800, Insulin completely dominates the distance calculation. Standardise ALL features before KNN.

**KNN fails in high dimensions:** As dimensions increase, all points become approximately equidistant from each other — the concept of "nearest neighbour" loses meaning. This is called the **curse of dimensionality**.

---

<a name="33-hyperparameter-k"></a>

## 3.3 Hyperparameter K

| K value | Boundary | Bias-Variance |
|---|---|---|
| K = 1 | Jagged — every training point is its own island | Low Bias, Very High Variance → **Overfit** |
| K = 5 to 15 | Smooth — captures the true underlying pattern | Balanced → **Sweet Spot** ✅ |
| K = n (all) | Flat — predicts majority class for everyone | Very High Bias, Low Variance → **Underfit** |

> **Rule:** K should be **odd** (avoids ties). Starting point: `K = sqrt(n_training)`.

**Nature: Non-Parametric · Local · Instance-Based**

---

<a name="part-4"></a>

# PART 4 — Support Vector Machine (SVM)

> **Philosophy:** *"Where is the safest, widest possible boundary between the two classes?"*

<a name="41-the-margin--core-geometry"></a>

## 4.1 The Margin — Core Geometry

SVM finds the boundary with the **maximum margin**: the widest possible gap between the two classes.

```
Margin width  =  2  /  ||w||
```

The training points on the margin edges are **Support Vectors** — the *only* points that determine the boundary. Remove any non-support-vector patient and the boundary stays identical. This is why SVM is called **sparse**.

**The optimisation problem:**

```
Minimise:    (1/2)  x  ||w||^2
Subject to:  y_i  x  (w^T * x_i  +  b)  >=  1     for all i
```

Maximising margin = minimising `||w||`. This is a convex quadratic programming problem with a **unique global solution**.

---

<a name="42-hard-margin-vs-soft-margin"></a>

## 4.2 Hard Margin vs. Soft Margin

| Type | What it does | When it breaks | Real-world usage |
|---|---|---|---|
| **Hard Margin** | Zero tolerance — all points outside the margin | Fails on noisy/non-separable data | Almost never |
| **Soft Margin** | Allows controlled violations with penalty C | Needs C tuned | **Almost always** ✅ |

**Soft Margin objective:**

```
Minimise:  (1/2) x ||w||^2  +  C  x  Sum(slack_i)
```

| C | Margin | Risk |
|---|---|---|
| High C | Narrow, strict → penalises violations heavily | Overfitting |
| Low C | Wide, tolerant → allows more violations | Underfitting |

---

<a name="43-the-kernel-trick"></a>

## 4.3 The Kernel Trick — SVM's Superpower

Real patient data is rarely linearly separable. The kernel trick implicitly maps data into a higher-dimensional space where it *becomes* linearly separable — without ever explicitly computing that space.

```
K(x_i, x_j)  =  phi(x_i)^T  x  phi(x_j)
```

| Kernel | Formula | Intuition |
|---|---|---|
| Linear | `x_i^T  x  x_j` | No transformation. Use when data is already separable. |
| RBF / Gaussian | `exp( -gamma  x  ||x_i - x_j||^2 )` | Smooth non-linear boundaries. Most widely used. |
| Polynomial | `( x_i^T  x  x_j  +  c )^d` | Polynomial surface. Good for image data. |

---

<a name="44-hyperparameters-c-and-gamma"></a>

## 4.4 Hyperparameters C and Gamma — Must Be Tuned Together

| Parameter | High value | Low value |
|---|---|---|
| **C** | Hard margin → overfitting | Soft margin → underfitting |
| **gamma (RBF)** | Each point only influences nearby area → complex, wiggly boundary → overfit | Each point influences wide area → smooth broad boundary → underfit |

**Nature: Geometric · Margin-Based · Sparse**

---

<a name="45-lr-vs-svm--key-difference"></a>

## 4.5 LR vs. SVM — They Both Find a Linear Boundary. What's Different?

Both find `w^T*x + b = 0`. The difference is *what they optimise* and *which patients they care about*:

| Dimension | Logistic Regression | SVM |
|---|---|---|
| **Objective** | Maximise likelihood (probabilistic) | Maximise geometric margin |
| **Loss Function** | Log-loss / cross-entropy | Hinge loss |
| **Which patients matter** | ALL — every patient contributes | ONLY support vectors — rest are irrelevant |
| **Outlier sensitivity** | Higher — outliers pull the weights | Lower — outliers far from margin are ignored |
| **Output** | Calibrated probability [0, 1] | Raw decision score (not a probability) |
| **Non-linearity** | Needs manual feature engineering | Kernel trick handles it natively |
| **Solution method** | Gradient Descent | Convex Quadratic Programming |
| **Theoretical basis** | MLE / PAC Learning | Structural Risk Minimization (Vapnik, 1995) |

> **Key geometric insight:** When data is perfectly linearly separable, LR finds infinitely many valid hyperplanes. SVM finds the **unique one** with the maximum margin — theoretically the best-generalising solution by VC dimension theory.

---

<a name="part-5"></a>

# PART 5 — Naive Bayes

> **Philosophy:** *"Which class would most likely have generated these features — assuming they're independent?"*

<a name="51-bayes-theorem--the-foundation"></a>

## 5.1 Bayes' Theorem — The Foundation

```
P(y | x)  =  P(x | y)  x  P(y)
             ──────────────────
                   P(x)
```

| Symbol | Name | Meaning |
|---|---|---|
| `P(y|x)` | **Posterior** | What we want: probability of class given features |
| `P(x|y)` | **Likelihood** | Probability of observing these features given the class |
| `P(y)` | **Prior** | Baseline probability of this class (without seeing features) |
| `P(x)` | **Evidence** | Same for all classes — cancels when comparing |

**For classification:**

```
Predict Diabetic  if  P(x|Diabetic)  x  P(Diabetic)  >  P(x|Not Diabetic)  x  P(Not Diabetic)
```

---

<a name="52-the-naive-assumption"></a>

## 5.2 The Naive Assumption

> **Assumption: All features are conditionally independent given the class label.**

```
P(x | y)  =  P(x1|y)  x  P(x2|y)  x  P(x3|y)  x  ...  x  P(xn|y)
```

**Is this realistic?** Almost never. BMI and Glucose are correlated. Age and BP are correlated.

**Then why does it work?** Because even with incorrect probability estimates, the *ranking* of classes (which posterior is larger) is often correct. The model is wrong about the magnitude of probabilities but right about which class is more likely. This is enough for classification.

> Additionally: low variance from fewer parameters compensates for the bias from the wrong assumption — especially with small data.

---

<a name="53-three-variants"></a>

## 5.3 Three Variants

### Variant 1 — Gaussian Naive Bayes (for continuous features)

```
P(xi | y)  =  (1 / sqrt(2 x pi x sigma^2_iy))  x  exp( -(xi - mu_iy)^2  /  (2 x sigma^2_iy) )
```

Training: compute `mu_iy` (mean) and `sigma_iy` (std dev) of each feature within each class.

**In practice — use log-probabilities to avoid numerical underflow:**

```
log P(y | x)  ∝  log P(y)  +  Sum_i[ log P(xi | y) ]
```

### Variant 2 — Multinomial Naive Bayes (for count/frequency features)

Used primarily for **text classification** (word frequency counts):

```
P(word_i | class_y)  =  (count(word_i in class_y)  +  alpha)
                         ──────────────────────────────────────────────
                         (total words in class_y  +  alpha x vocab_size)
```

### Variant 3 — Bernoulli Naive Bayes (for binary features)

Used when features are binary (present = 1, absent = 0):

```
P(xi | y)  =  p_iy^xi  x  (1 - p_iy)^(1-xi)
```

---

<a name="54-laplace-smoothing"></a>

## 5.4 Laplace Smoothing — Preventing Zero Probabilities

**The problem:** If a feature value never appeared in training for a class, its count = 0, making P(xi|y) = 0. A single zero makes the entire posterior zero — the model can never assign that class.

**Laplace Smoothing** adds a pseudocount alpha (usually 1):

```
P(xi | y)  =  (count(xi, y)  +  alpha)  /  (count(y)  +  alpha  x  K)

where K = number of possible values for feature xi
```

**Properties, Strengths, Weaknesses:**

| Strengths | Weaknesses |
|---|---|
| Extremely fast — O(n x d) training | Independence assumption is almost always wrong |
| Works well with small data | Distributional assumptions (Gaussian) may not hold |
| Excellent for text (Multinomial NB) | Probability estimates can be poorly calibrated |
| Natural multi-class | Cannot capture feature interactions |
| Online/streaming learning (incremental updates) | |

**When to use:**
- Text classification (spam, sentiment, news)
- Very small training datasets
- Real-time systems where speed is critical
- Streaming/online learning

---

<a name="part-6"></a>

# PART 6 — Decision Tree

> **Philosophy:** *"Split the patients by asking yes/no questions about features, until each group is as pure as possible."*

<a name="61-how-the-tree-learns--splitting-criteria"></a>

## 6.1 How the Tree Learns — Splitting Criteria

The tree is built **greedily** — at each node, find the best (feature, threshold) pair that maximally reduces impurity:

```
For each feature f:
  For each possible threshold t:
    Compute: Information Gain = Impurity(Parent) - weighted_avg Impurity(children)

Choose (f*, t*) with maximum Information Gain.
```

---

<a name="62-gini-entropy-variance-reduction"></a>

## 6.2 Gini, Entropy, and Variance Reduction

### Gini Index (sklearn default)

```
Gini(node)  =  1  -  Sum_k( p_k^2 )

Example: node with 70% Diabetic, 30% Not:
Gini  =  1  -  (0.70^2 + 0.30^2)  =  1  -  0.58  =  0.42

Gini = 0.0  →  Pure node (all one class)
Gini = 0.5  →  Maximum impurity (50/50 split)
```

### Entropy and Information Gain

```
Entropy(node)  =  - Sum_k( p_k  x  log2(p_k) )

For same node (70% D, 30% ND):
Entropy  =  -(0.70 x log2(0.70)  +  0.30 x log2(0.30))  =  0.881 bits

Information Gain:
IG  =  Entropy(parent)  -  Sum[ (|child| / |parent|)  x  Entropy(child) ]
```

### Variance Reduction (for Regression Trees)

```
Var(node)  =  (1/n)  x  Sum( (yi - y_mean)^2 )

Variance Reduction  =  Var(parent)  -  Sum[ (|child| / |parent|)  x  Var(child) ]
```

Leaf prediction = **mean of target values in that leaf**.

**Gini vs. Entropy:**

| Property | Gini | Entropy |
|---|---|---|
| Computation | Faster (no logarithm) | Slightly slower |
| Practical result | Nearly identical trees (~2% differ) | Nearly identical |
| Recommendation | Use sklearn default | Use if specific reason |

---

<a name="63-tree-building-algorithm"></a>

## 6.3 Tree Building — Full Algorithm (CART)

```
BuildTree(node, data, depth):

  Step 1 — Check stopping criteria:
    - All samples are the same class      → make leaf
    - max_depth reached                   → make leaf
    - Fewer samples than min_samples_split → make leaf
    - No split improves impurity          → make leaf

  Step 2 — Find best split:
    Compute IG for every (feature, threshold) pair
    Select (f*, t*) with maximum gain

  Step 3 — Split data into Left and Right subsets
  Step 4 — Recursively call BuildTree(Left, depth+1)
  Step 5 — Recursively call BuildTree(Right, depth+1)
```

---

<a name="64-hyperparameters--the-pruning-dials"></a>

## 6.4 Hyperparameters — The Pruning Dials

Without constraints, a tree splits until every leaf has one patient — 100% training accuracy, terrible test accuracy.

| Hyperparameter | Low value → | High value → |
|---|---|---|
| `max_depth` | Underfit (shallow, simple) | Overfit (deep, complex) |
| `min_samples_split` | Overfit (splits on tiny groups) | Underfit (few splits) |
| `min_samples_leaf` | Overfit (single-sample leaves) | Underfit (large leaves) |
| `max_features` | Less variance, more bias | More variance, more complexity |

```
max_depth = 2          max_depth = 5             max_depth = None

   Glucose?               Glucose?                  Glucose?
   /      \               /        \                ....deep....
Diab.  BMI?          BMI?       Age+BP?             thousands of
        /   \          /\           /    \           tiny leaves
      ND  Diab.    ND  Diab.    Diab.  Family?       (memorises noise)

Train: 71%             Train: 88%                  Train: 100%
Test:  69%             Test:  84% ✅               Test:   67%
UNDERFIT               SWEET SPOT                  OVERFIT
```

> ⚠️ **The Variance Problem:** Change just 5 training patients and you may get a completely different tree. **This is exactly what Random Forest was designed to fix.**

---

<a name="part-7"></a>

# PART 7 — Random Forest

> **Philosophy:** *"The wisdom of the crowd beats the opinion of any individual expert."*

<a name="71-two-randomisation-tricks"></a>

## 7.1 Two Randomisation Tricks — The Heart of Random Forest

Random Forest introduces two sources of randomness to ensure trees are **diverse (uncorrelated)**. Correlated trees give you nothing — averaging ten trees that make the same mistakes doesn't help.

### Trick 1 — Bootstrap Sampling (Bagging)

Each tree is trained on a bootstrap sample — random sampling **with replacement**, same size as original.

```
Original: [P1, P2, P3, ..., P8000]

Tree 1:   [P1, P1, P3, P7, P12, P12, ...]   ← different selection
Tree 2:   [P4, P4, P9, P10, P1, P22, ...]   ← different selection
Tree 3:   [P2, P8, P8, P11, P6,  ...]       ← different again

~37% of patients NOT selected per tree  →  Out-of-Bag (OOB) samples
```

**OOB samples** = free internal validation, no separate validation set needed:

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)   # free, unbiased test accuracy estimate
```

### Trick 2 — Feature Subsampling (Random Subspace Method)

At **each split of each tree**, only a random subset of `max_features` is considered.

```
Typical: max_features = sqrt(n_features)  for classification
         max_features = n_features / 3    for regression

With 8 features {Glucose, BMI, Age, BP, Insulin, Smoking, Family, Diet}:
Tree 1, root: consider only {Glucose, Age, Smoking}  → best = Glucose < 140
Tree 2, root: consider only {BP, Family, Smoking}    → best = Family = Yes
```

Without this, the dominant feature (Glucose) would be the root of every tree → all trees correlated → averaging helps little. Feature subsampling forces **tree diversity**.

> This is the key difference between **Bagging** (just resampling data) and **Random Forest** (resampling data + randomising features at each split).

---

<a name="72-why-averaging-reduces-variance"></a>

## 7.2 Why Averaging Reduces Variance — The Math

```
Variance of the Forest (average of n trees):

  Var(Forest)  =  rho  x  sigma^2  +  (1 - rho)  x  sigma^2 / n

where:
  sigma^2 = variance of each individual tree
  rho     = pairwise correlation between trees

When rho = 1 (all trees identical):
  Var(Forest) = sigma^2          ← no improvement

When rho = 0 (completely independent):
  Var(Forest) = sigma^2 / n      ← shrinks by factor n

With bootstrap + feature subsampling: rho ≈ 0.05–0.15

Example (100 trees, sigma^2=1, rho=0.1):
  Var(Forest) = 0.1 + 0.9*(1/100) = 0.109   ← 9x reduction!
```

**Prediction:**

```
Classification: Final = majority vote across all B trees
Regression:     Final = mean prediction across all B trees
```

---

<a name="73-hyperparameters"></a>

## 7.3 Hyperparameters

| Hyperparameter | Effect | Typical range |
|---|---|---|
| `n_estimators` | More trees = lower variance (diminishing returns after ~200) | 100–500 |
| `max_depth` | Deeper = more complex = more overfit risk | 5–20 or None |
| `max_features` | Lower = more diverse trees = lower variance, higher bias | 'sqrt', 'log2', 0.3–0.7 |
| `min_samples_leaf` | Higher = simpler trees = lower variance | 1–20 |
| `bootstrap` | True = bagging; False = all data per tree | True |

> **n_estimators does not cause overfitting** — adding more trees only hits diminishing returns. Unlike max_depth, you cannot overfit by adding more trees.

---

<a name="74-feature-importance"></a>

## 7.4 Feature Importance

Random Forest ranks features by how much each one improves the splits across all trees.

```
Feature Importances (sample):
Glucose          0.341  ████████████████████████████████████
BMI              0.198  ████████████████████
Age              0.151  ███████████████
Insulin          0.112  ███████████
Blood_Pressure   0.089  █████████
Family_History   0.062  ██████
Smoking          0.031  ███
Diet_Quality     0.016  ██
```

> ⚠️ Impurity-based importance is biased toward high-cardinality and continuous features. Use **permutation importance** or **SHAP** for more reliable rankings.

**Strengths:** Low variance, OOB error, parallel training, robust to outliers, built-in feature importance.
**Weaknesses:** Black-box (needs SHAP), memory intensive, slower inference than single tree.

---

<a name="part-8"></a>

# PART 8 — XGBoost (Extreme Gradient Boosting)

> **Philosophy:** *"Don't train all trees simultaneously — train them sequentially. Each new tree corrects the mistakes of all previous trees."*

<a name="81-gradient-boosting--core-mechanism"></a>

## 8.1 Gradient Boosting — Core Mechanism

```
BAGGING (Random Forest):
  Tree1  Tree2  Tree3  ...  TreeN
    ↓      ↓      ↓           ↓
    ←── average / majority vote ──→
  (all built simultaneously)

BOOSTING (XGBoost):
  Tree1 → has errors
  Tree2 → built to fix Tree1's errors    → fewer errors
  Tree3 → built to fix Trees1+2 errors   → fewer errors
  ...
  Final = weighted sum of all trees
```

### Step-by-Step Walkthrough

```
ROUND 0 — Initial Prediction:
  F0(x) = log(0.30/0.70) = -0.847  (log-odds, base rate = 30%)
  → Everyone gets 30% predicted diabetes probability.

ROUND 1 — Compute Residuals:
  r_i  =  y_i  -  p_hat_i

  Priya (y=1, Diabetic):       r = 1 - 0.30 = +0.70   (under-predicted)
  Arjun (y=0, Not Diabetic):   r = 0 - 0.30 = -0.30   (over-predicted)

ROUND 1 — Train Tree1 to predict residuals:
  Tree1 learns:  Glucose > 155 AND BMI > 30  →  residual ≈ +0.65
                 Glucose < 120               →  residual ≈ -0.25

ROUND 1 — Update Ensemble:
  F1(x)  =  F0(x)  +  eta  x  Tree1(x)      (eta = learning rate, e.g. 0.1)

  Priya: -0.847 + 0.1 x 0.65 = -0.782  →  sigmoid = 0.314
  (Now 31.4% — slightly better. Priya is actually diabetic.)

ROUND 2 — New residuals from F1, train Tree2 ...

ROUND T — Final:
  F_T(x)  =  F0(x)  +  eta  x  Sum_t[ Tree_t(x) ]
  p_hat   =  sigmoid( F_T(x) )
```

> Each tree is a correction to the previous ensemble's mistakes. The ensemble iteratively focuses on the hardest patients — those where the current prediction is most wrong.

---

<a name="82-xgboost-objective-function"></a>

## 8.2 XGBoost Objective Function

```
Obj  =  Sum_i[ L(yi, y_hat_i) ]   +   Sum_t[ Omega(ft) ]
        ─────────────────────────       ───────────────────
              Training Loss                Regularisation

Omega(ft)  =  gamma  x  T  +  (1/2)  x  lambda  x  Sum_j( w_j^2 )
               ─────────        ───────────────────────────────────
               penalty on       L2 penalty on
               number of        leaf weights
               leaves T
```

**Second-order Taylor expansion — what makes XGBoost "eXtreme":**

Standard GBM uses only the first derivative (gradient). XGBoost uses **both gradient and Hessian**:

```
Gradients per patient:
  g_i  =  first derivative of loss  =  y_hat_i  -  y_i    (residual)
  h_i  =  second derivative of loss  =  y_hat_i  x  (1 - y_hat_i)   (Hessian)
```

**Optimal leaf weight (closed-form):**

```
w*_j  =  -(Sum of g_i in leaf j)  /  (Sum of h_i in leaf j  +  lambda)
```

**Split gain:**

```
Gain  =  (1/2) x [ G_L^2/(H_L+lambda)  +  G_R^2/(H_R+lambda)  -  G_P^2/(H_P+lambda) ]  -  gamma
```

The `-gamma` means a split is only made if gain exceeds gamma — **automatic pruning** built into the gain calculation.

---

<a name="83-key-hyperparameters"></a>

## 8.3 Key Hyperparameters

| Parameter | What It Controls | High → | Low → |
|---|---|---|---|
| `n_estimators` | Number of trees | More complex | Underfit |
| `learning_rate (eta)` | Step size per tree | Overfit | Underfit |
| `max_depth` | Depth of each tree | Overfit | Underfit |
| `subsample` | Fraction of rows per tree | — | More regularisation |
| `colsample_bytree` | Fraction of features per tree | — | More regularisation |
| `gamma` | Min gain to split | More pruning | Less pruning |
| `reg_lambda` | L2 on leaf weights | More regularisation | Less |

**The learning rate + n_estimators tradeoff:**

```
Low eta (0.01):   tiny corrections → need MORE trees → smoother, less overfit
High eta (0.3):   big corrections  → need FEWER trees → can overshoot, more overfit

RULE: Lower eta + more trees almost always outperforms high eta + few trees.

eta=0.3, 50 trees:    Test AUC = 0.87
eta=0.1, 150 trees:   Test AUC = 0.90
eta=0.05, 300 trees:  Test AUC = 0.92  ✅
```

**Early Stopping — The Most Practical Tool:**

```python
model = xgb.XGBClassifier(
    n_estimators          = 1000,   # set high — early stopping decides actual number
    learning_rate         = 0.05,
    early_stopping_rounds = 50,     # stop if no improvement for 50 rounds
    eval_metric           = 'logloss'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
print(f"Best iteration: {model.best_iteration}")
```

---

<a name="84-xgboost-vs-random-forest"></a>

## 8.4 XGBoost vs. Random Forest — The Key Differences

| Property | Random Forest | XGBoost |
|---|---|---|
| **Tree building** | Parallel — all built independently | Sequential — each corrects previous |
| **Error focus** | Each tree sees different data (bootstrap) | Each tree explicitly targets current errors |
| **Bias-Variance** | Primarily reduces Variance | Reduces both Bias AND Variance |
| **Fixing underfitting** | Hard — cannot decrease bias easily | Easy — add more rounds |
| **Fixing overfitting** | max_depth, max_features | learning_rate, regularisation, gamma |
| **Training speed** | Fast (parallel) | Slower (sequential) but highly optimised |
| **Regularisation** | Implicit (bootstrap, feature sampling) | Explicit — L1/L2 on leaf weights + gamma |
| **Out-of-the-box** | Great — robust to default settings | Good — learning_rate is sensitive |
| **Tuned performance** | Very good | Usually better than RF |

---

<a name="85-lightgbm-and-catboost"></a>

## 8.5 LightGBM and CatBoost — Brief Mentions

**LightGBM (Microsoft, 2017):** Leaf-wise growth (vs XGBoost's level-wise) + GOSS sampling. Result: significantly faster training with similar accuracy.

**CatBoost (Yandex, 2017):** Handles categorical features natively without encoding, using ordered boosting to reduce prediction shift bias.

---

<a name="part-9"></a>

# PART 9 — Hyperparameter Tuning & Cross-Validation

<a name="91-why-default-models-fail"></a>

## 9.1 Why Default Models Fail

Running any model with default settings gives you a random guess at the complexity level:

```
UNDERFITTING                         OVERFITTING
(Model too simple)                   (Model too complex)

Training Accuracy:  62%              Training Accuracy:  99%
Test Accuracy:      60%              Test Accuracy:      61%

Didn't learn enough.                 Memorised the 8,000 patients.
Failed the exam without studying.    Crammed textbook including typos.
                                     Failed the real exam.
```

**Hyperparameter tuning finds the sweet spot between these two extremes.**

Every hyperparameter is a complexity dial:

| Algorithm | Hyperparameter | Low value → | High value → |
|---|---|---|---|
| Linear Regression | lambda | Overfit (no weight penalty) | Underfit (all weights ~0) |
| Logistic Regression | C (= 1/lambda) | Underfit | Overfit |
| KNN | K | Overfit (K=1 chaotic) | Underfit (K=n flat) |
| SVM | C | Underfit (soft margin) | Overfit (hard margin) |
| SVM | gamma (RBF) | Underfit (broad influence) | Overfit (narrow influence) |
| Decision Tree | max_depth | Underfit (shallow) | Overfit (deep) |
| Random Forest | n_estimators | High variance | More stable |
| XGBoost | learning_rate | Underfit | Overfit |

---

<a name="92-k-fold-cross-validation-mechanics"></a>

## 9.2 K-Fold Cross-Validation Mechanics

**The forbidden mistake:**

```
Step 1: Train model on training data
Step 2: Tune hyperparameters until test accuracy looks good   ← WRONG
Step 3: Report that test accuracy

❌ The test set is now contaminated. Your reported accuracy is fake.
```

**Cross-validation creates a validation layer inside the training pool:**

```
Full Dataset (10,000 patients)
│
├── Test Set (2,000 patients) 🔒    ← NEVER touched during tuning
│
└── Training Pool (8,000 patients)
    ├── Fold 1: [Val 1600 | Train 6400]
    ├── Fold 2: [Train 1600 | Val 1600 | Train 4800]
    ├── Fold 3: [Train 3200 | Val 1600 | Train 3200]
    ├── Fold 4: [Train 4800 | Val 1600 | Train 1600]
    └── Fold 5: [Train 6400 | Val 1600]
```

For each hyperparameter value: train 5 times on different 6400-patient subsets, validate on the remaining 1600, **average the 5 scores**. This is a stable, unbiased estimate of generalisation.

**Why K-Fold is better than a single split:**

A single split is noisy — performance depends heavily on which patients happened to fall in validation by chance. K-Fold averages K different sets. By the Law of Large Numbers:

```
std_error_of_mean  =  std_of_individual_folds  /  sqrt(K)
```

**Stratified K-Fold** ensures each fold maintains the same class proportion as the full dataset — critical for imbalanced data.

**5-Fold vs. 10-Fold vs. LOOCV:**

| Method | Folds | Train size/fold | Variance of estimate | Compute cost | Best for |
|---|---|---|---|---|---|
| 5-Fold | 5 | 80% | Moderate | Low | Standard choice for most datasets |
| 10-Fold | 10 | 90% | Lower | Higher | Larger datasets, more reliable estimate |
| LOOCV | n | ~100% | Lowest | Very High | Tiny datasets (< 100 samples) |

---

<a name="93-search-strategies"></a>

## 9.3 Grid Search, Random Search, Bayesian Optimisation

### Grid Search

Evaluates every combination in a predefined grid.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'model__C':     [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

**For SVM (two hyperparameters — Grid Search):**

```
              gamma=0.001   gamma=0.01   gamma=0.1
C=0.1    [       78%           81%          76%   ]
C=1.0    [       82%           87%          83%   ]  ← Best
C=10     [       83%           85%          79%   ]
C=100    [       84%           82%          71%   ]
```

### Random Search

Samples `n_iter` random combinations from the hyperparameter space. More efficient than grid search for high-dimensional spaces because it covers more ground per evaluation.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_distributions = {
    'C':     loguniform(0.001, 1000),
    'gamma': loguniform(0.0001, 10)
}
random_search = RandomizedSearchCV(svc, param_distributions, n_iter=50, cv=cv, scoring='f1')
```

### Bayesian Optimisation (optuna)

Learns from previous trials to intelligently decide where to search next. Best when each training run is expensive.

```python
import optuna

def objective(trial):
    C = trial.suggest_float('C', 0.001, 1000, log=True)
    gamma = trial.suggest_float('gamma', 0.0001, 10, log=True)
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

| Strategy | Best for | Trade-off |
|---|---|---|
| Grid Search | ≤3 hyperparameters, small grid | Exhaustive but expensive |
| Random Search | ≥3 hyperparameters, continuous ranges | Efficient exploration |
| Bayesian | Expensive models, many hyperparameters | Smartest, most complex |

---

<a name="94-complete-tuning-pipeline"></a>

## 9.4 Complete Tuning Pipeline

```
FULL DATASET (10,000 patients)
│
├── TEST SET 🔒 (2,000 patients) ────────────────────────────────────────┐
│   Locked. Never opened until Step 5.                                   │
│                                                                         │
└── TRAINING POOL (8,000 patients)                                       │
    │                                                                     │
    ├── Preprocess inside Pipeline                                        │
    │   (impute, scale, encode — fit on train fold only each CV round)   │
    │                                                                     │
    ├── HYPERPARAMETER SEARCH                                             │
    │   Try C=0.001 → 5-fold CV → avg F1 = 0.74                         │
    │   Try C=0.01  → 5-fold CV → avg F1 = 0.79                         │
    │   Try C=0.1   → 5-fold CV → avg F1 = 0.83                         │
    │   Try C=1.0   → 5-fold CV → avg F1 = 0.87  ✅ Best                │
    │   Try C=10    → 5-fold CV → avg F1 = 0.84                         │
    │                                                                     │
    ├── RETRAIN final model (C=1.0) on ALL 8,000 patients                │
    │                                                                     │
    └────────────────────────────────────────────────────────────────►  │
                                                                          │
    STEP 5: Evaluate ONCE on 2,000 test patients ◄────────────────────┘
            Report final honest: Accuracy, Precision, Recall, F1, AUC
```

**Data leakage in cross-validation — the most dangerous mistake:**

```python
# WRONG — leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # uses ALL data including validation!
X_train, X_test = train_test_split(X_scaled)

# CORRECT — use Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression())
])
scores = cross_val_score(pipe, X_train, y_train, cv=5)
# Pipeline fits scaler on each fold's training portion only ✅
```

---

<a name="part-10"></a>

# PART 10 — Bias, Variance & The Tradeoff

<a name="101-definitions"></a>

## 10.1 Definitions

**Bias** — Error from wrong assumptions. A straight line trying to fit a curve will always be wrong, no matter how much data you give it.

**Variance** — Error from sensitivity to specific training data. A K=1 KNN draws a completely different boundary if you swap just 5 of the 8,000 training patients.

**Irreducible Error** — Randomness in the real world no model can eliminate. A perfectly healthy patient randomly develops diabetes — no features could predict this.

**The fundamental equation:**

```
Total Error  =  Bias^2  +  Variance  +  Irreducible Noise
```

**The tradeoff:** Increasing complexity reduces Bias but increases Variance. The sweet spot — minimum total error — is the bottom of a U-shaped error curve.

---

<a name="102-the-2-number-diagnostic-test"></a>

## 10.2 The 2-Number Diagnostic Test

Just look at two numbers:

| Training Accuracy | Test Accuracy | Diagnosis | Fix |
|---|---|---|---|
| Low (55%) | Low (54%) | **HIGH BIAS — Underfitting** | More complex model, more features, lower regularisation |
| High (99%) | Low (62%) | **HIGH VARIANCE — Overfitting** | More data, regularise, increase K, lower C |
| High (87%) | High (85%) | **SWEET SPOT** ✅ | Deploy |

---

<a name="103-where-every-algorithm-sits"></a>

## 10.3 Where Every Algorithm Sits on the Spectrum

```
← HIGH BIAS (Underfit)                                  HIGH VARIANCE (Overfit) →

Naive Bayes    Lin/Log Reg    SVM (tuned)   RF (tuned)   XGBoost    KNN (K=1)
(independence  (high lambda)  (C=1,g=0.01)  (n=200)      (high lr)  Decision Tree
 violated)                                                            (no pruning)
```

**Diagnosing by algorithm:**

| Algorithm | Typical issue | Quick diagnosis | Fix |
|---|---|---|---|
| Naive Bayes | High Bias (independence violated) | Train ≈ Test, both moderate | Switch to tree model |
| Decision Tree | High Variance | Train >> Test | max_depth, min_samples |
| Random Forest | Slight overfit | Train slightly > Test | Increase min_samples_leaf |
| XGBoost | Severe overfit (high lr) | Train >> Test | Lower lr, add subsample, gamma |

---

<a name="part-11"></a>

# PART 11 — Evaluation Metrics: Complete Guide

<a name="111-the-confusion-matrix"></a>

## 11.1 The Confusion Matrix — Universal Convergence Point

**Why all classifiers meet here:** LR, KNN, SVM, NB, DT, RF, XGBoost all produce one thing for each patient: a predicted label `y_hat ∈ {0, 1}`. The confusion matrix only ever sees `(y_true, y_hat)` pairs — it does **not** know or care whether that label came from a sigmoid, a majority vote, a margin hyperplane, or a probability product.

**Building the matrix** — open vault: 2,000 test patients (600 Diabetic, 1,400 Not Diabetic):

```
                          PREDICTED
                    Not Diabetic (0)    Diabetic (1)
         ┌─────────────────────────────────────────────┐
A   Not  │                             │               │
C  Diab. │    TN = 1,312  ✅           │   FP = 88  ❌ │
T   (0)  │                             │  Type I Error │
U        ├─────────────────────────────────────────────┤
A  Diab. │                             │               │
L   (1)  │    FN = 78   ❌             │   TP = 522 ✅ │
         │  Type II Error              │               │
         └─────────────────────────────────────────────┘
```

| Cell | Full Name | What happened | Consequence |
|---|---|---|---|
| **TP = 522** | True Positive | Said Diabetic ✅, Was Diabetic ✅ | Correct — patient gets treatment |
| **TN = 1,312** | True Negative | Said Healthy ✅, Was Healthy ✅ | Correct — no unnecessary intervention |
| **FP = 88** | False Positive (Type I) | Said Diabetic ❌, Was Healthy | Unnecessary tests — stressful, not fatal |
| **FN = 78** | False Negative (Type II) | Said Healthy ❌, Was Diabetic | **Patient goes home untreated — potentially fatal** |

> 💡 **Memory trick:** True/False = was the prediction correct? Positive/Negative = what did the model predict?

**How each algorithm produces ŷ:**

```
Logistic Regression:  y_hat = 1  if  P(y=1|x) >= tau  (default tau=0.5)
KNN:                  y_hat = mode { y_i : x_i in N_K(x) }
SVM:                  y_hat = sign( w^T*x + b )
```

---

<a name="112-all-classification-metrics"></a>

## 11.2 All Classification Metrics — Formulae and Meaning

### Accuracy

```
Accuracy  =  (TP + TN)  /  (TP + TN + FP + FN)
           =  (522 + 1312) / 2000  =  91.7%
```

*Of all 2,000 patients, what fraction did the model get right?*

> ⚠️ **The Accuracy Trap:** On an imbalanced dataset (950 healthy, 50 diabetic), a model that predicts everyone is healthy scores 95% accuracy — yet misses every single diabetic. **Never trust Accuracy alone on imbalanced data.**

---

### Precision

```
Precision  =  TP  /  (TP + FP)
           =  522 / (522 + 88)  =  85.6%
```

*Of all patients the model flagged as Diabetic, what fraction actually were?*

> Optimise Precision when **False Positives are costly** — spam filter, legal decisions.

---

### Recall (Sensitivity / True Positive Rate)

```
Recall  =  TP  /  (TP + FN)
        =  522 / (522 + 78)  =  87.0%
```

*Of all patients who actually were Diabetic, what fraction did the model catch?*

> Optimise Recall when **False Negatives are costly** — disease screening, fraud detection, fire alarms.

---

### The Precision-Recall Tradeoff

| Threshold | Strategy | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| 0.7 (strict) | Only flag when very confident | High ↑ | Low ↓ | Few | Many |
| 0.5 (default) | Balanced | 85.6% | 87.0% | 88 | 78 |
| 0.3 (lenient) | Flag aggressively | Low ↓ | High ↑ | Many | Few |

> There is no universally right threshold — it is a clinical/business decision based on the relative cost of FP vs FN in your domain.

---

### F1 Score

```
F1  =  2  x  (Precision  x  Recall)  /  (Precision  +  Recall)
    =  2 x 0.856 x 0.870 / (0.856 + 0.870)  =  86.3%
```

**Why harmonic mean and not simple average?**

A model with Precision=90%, Recall=10%:
- Arithmetic mean = **50%** — looks passable
- F1 = **18%** — correctly reveals this model is terrible

The harmonic mean is always closer to the *smaller* value. A model cannot hide terrible Recall behind great Precision.

> Use F1 when you want one balanced number, especially on imbalanced datasets.

---

### Specificity (True Negative Rate)

```
Specificity  =  TN  /  (TN + FP)
             =  1312 / (1312 + 88)  =  93.7%
```

*Of all actually-healthy patients, what fraction were correctly identified?*

```
Recall (Sensitivity)  →  How well the model catches DIABETIC patients
Specificity           →  How well the model catches HEALTHY patients
Together: complete picture across BOTH classes.
```

---

### AUC-ROC

The ROC curve plots **Recall (TPR)** on the Y-axis against **False Positive Rate (1 − Specificity)** on the X-axis, sweeping the decision threshold from 1.0 down to 0.0.

```
Recall (TPR)
1.0 │                  ╭──────────── Perfect model (AUC=1.0)
    │            ╭─────╯
    │       ╭────╯                ← Your model (AUC ≈ 0.93)
    │   ╭───╯
    │╭──╯
    │╱                            ← Random guessing (AUC=0.5)
    └──────────────────────────────
    0.0                        1.0
             FPR (1 - Specificity)
```

| AUC | Meaning |
|---|---|
| 1.0 | Perfect model |
| 0.9+ | Excellent |
| 0.8 | Good |
| 0.7 | Fair |
| 0.5 | Random guessing — no discriminating power |

> **Why AUC is powerful:** It is **threshold-independent** — evaluates discriminating ability across all possible thresholds at once. Gold standard for comparing classifiers.

---

### Log Loss

```
Log Loss  =  -(1/n)  x  Sum[ y  x  log(p)  +  (1-y)  x  log(1-p) ]
```

Unlike all other metrics that only look at the final label, Log Loss evaluates the **confidence** of predictions. It catastrophically penalises a model that is wrong and confident.

| Algorithm | Probability Output | Log Loss compatible? |
|---|---|---|
| Logistic Regression | Natively calibrated | ✅ Directly |
| KNN | Vote fraction as proxy | ⚠️ Less calibrated |
| SVM | Raw decision score | ❌ Needs Platt Scaling |
| Naive Bayes | Posterior probability | ✅ But naivety distorts calibration |
| Random Forest | Average vote fraction | ✅ Reasonably calibrated |
| XGBoost | Sigmoid of boosted score | ✅ Well calibrated |

---

<a name="113-regression-metrics"></a>

## 11.3 Regression Metrics — Side-by-Side

Using 5 test patients with Blood Sugar predictions:

| Patient | Actual | Predicted | Error |
|---|---|---|---|
| P1 | 145 | 142 | −3 |
| P2 | 160 | 175 | +15 |
| P3 | 120 | 118 | −2 |
| P4 | 200 | 165 | −35 |
| P5 | 95 | 97 | +2 |

```
MAE   =  (3 + 15 + 2 + 35 + 2) / 5         =  11.4 mg/dL
MSE   =  (9 + 225 + 4 + 1225 + 4) / 5      =  293.4 (mg/dL)^2
RMSE  =  sqrt(293.4)                        =  17.1 mg/dL
R^2   =  1  -  (1467 / 6370)               =  0.77
```

**MAE vs RMSE — the gap reveals outliers:**

```
MAE  = 11.4
RMSE = 17.1
Gap  = 5.7   ← Because P4 (error = -35)
             RMSE punishes 35 as 35^2 = 1225
             MAE treats it the same as a 3 mg/dL error

RMSE >= MAE always. Large gap = outlier errors hiding in the average.
```

| Metric | Formula | Unit | Outlier sensitive? | Best for |
|---|---|---|---|---|
| MAE | `Mean( |y_hat - y| )` | Same as target | No | General purpose, business-friendly |
| MSE | `Mean( (y_hat-y)^2 )` | Squared units | Yes | Mathematical optimisation (GD) |
| RMSE | `sqrt(MSE)` | Same as target | Yes | When large errors are costly |
| R² | `1 - SS_res/SS_tot` | Unitless 0–1 | Moderate | Understanding explanatory power |

---

<a name="114-choosing-the-right-metric"></a>

## 11.4 Choosing the Right Metric — Decision Guide

```
REGRESSION task?
└── Report MAE + RMSE + R²

CLASSIFICATION task?
│
├── Is dataset balanced?
│   ├── YES → Accuracy is a reasonable starting point
│   └── NO  → NEVER use Accuracy alone → use F1, AUC
│
├── What is the cost of each error?
│   ├── FN is more dangerous (disease, fraud, fire)  → optimise RECALL
│   ├── FP is more dangerous (spam, legal)           → optimise PRECISION
│   └── Both matter equally                           → optimise F1
│
├── Comparing models holistically?
│   └── Use AUC-ROC (threshold-independent, single number)
│
└── Need calibrated probability outputs?
    └── Use Log Loss
```

---

<a name="115-domain-cheat-sheet"></a>

## 11.5 Domain Cheat Sheet

| Domain | Task | Most Important Metric | Reason |
|---|---|---|---|
| Cancer / Disease Screening | Classification | **Recall** | Missing real case (FN) can be fatal |
| Spam Filter | Classification | **Precision** | Deleting legitimate emails (FP) destroys trust |
| Fraud Detection | Classification | **Recall + F1** | Missing fraud (FN) = direct financial loss |
| House Price Prediction | Regression | **RMSE + R²** | Large price errors matter a lot |
| Delivery Time Prediction | Regression | **MAE** | All errors roughly equal importance |
| Credit Risk Scoring | Classification | **AUC + Log Loss** | Calibrated probabilities needed |
| Weather Forecasting | Regression | **RMSE** | Extreme weather errors are costly |
| Resume Screening | Classification | **Precision** | False positives waste recruiter time |
| Fire / Flood Alarm | Classification | **Recall** | Missing a real event is catastrophic |
| Genomics / bioinformatics | Classification | **AUC** | High-dimensional, compare models overall |

---

<a name="part-12"></a>

# PART 12 — Ensemble Methods — The Big Picture

## The Three Ensemble Strategies

**Strategy 1 — Bagging (Bootstrap Aggregating)**

Train multiple models independently on random data subsets. Average/vote their predictions.

```
Goal: Reduce VARIANCE
Mechanism: Uncorrelated errors cancel when averaged
Example: Random Forest
```

**Strategy 2 — Boosting**

Train models sequentially. Each model focuses on the examples the previous models got wrong.

```
Goal: Reduce BIAS
Mechanism: Weak learners combine into a strong learner
Example: XGBoost, LightGBM, AdaBoost
```

**Strategy 3 — Stacking**

Train multiple different models. Feed their predictions as input to a **meta-model** that learns the best way to combine them.

```
Level 0 (base models):  LR → 0.72    RF → 0.85    SVM → 0.79
                                ↓         ↓          ↓
Level 1 (meta-model):  [0.72, 0.85, 0.79] → Final prediction

Meta-model learns: "trust RF 60%, SVM 30%, LR 10%"
```

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| **Models trained** | In parallel | Sequentially | Parallel + meta-model |
| **What it reduces** | Variance | Bias | Both |
| **Overfitting risk** | Low | Medium-High | High (meta-model) |
| **Complexity** | Simple | Moderate | Complex |
| **Example** | Random Forest | XGBoost | BlendedModel |

---

<a name="part-13"></a>

# PART 13 — Master Comparison Tables

## Algorithm Properties at a Glance

| Property | Linear Reg | Logistic Reg | KNN | SVM | Naive Bayes | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|---|---|---|---|
| **Task** | Regression | Classification | Both | Both | Classification | Both | Both | Both |
| **Output** | Number | Probability | Label | Score | Probability | Label | Label | Label/Number |
| **Boundary** | Linear | Linear | Any (Voronoi) | Linear+Kernel | Curved (Gaussian) | Rectangular | Complex | Complex |
| **Scaling needed** | ✅ | ✅ | ✅✅ MANDATORY | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Interpretable** | ✅✅ | ✅✅ | ❌ | ❌ | Partial | ✅✅ (if shallow) | ❌ (SHAP) | ❌ (SHAP) |
| **Non-linear data** | ❌ | ❌ | ✅ | ✅ (kernel) | ❌ | ✅ | ✅ | ✅ |
| **Small data** | ✅ | ✅ | Okay | ✅ | ✅✅ | Okay | ✅ | Needs tuning |
| **Large data (>1M)** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Missing values** | Struggles | Struggles | Struggles | Struggles | Natural | Struggles | Handles | **Natively** |
| **Overfit risk** | Low | Low | High (K=1) | Low | Low | **Very High** | Low | Medium |
| **Training speed** | Fast | Fast | Zero | Slow | **Very Fast** | Fast | Moderate | Moderate |
| **Inference speed** | Fast | Fast | **Slow** | Fast | Fast | Fast | Fast | Fast |
| **High dimensions** | ✅ | ✅ | ❌ | ✅ | ✅ (NLP) | ❌ | ✅ | ✅ |
| **Typical performance** | Baseline | Baseline | Moderate | Good | Fast/Good | Weak | Strong | **Best** |

## Performance on Our Diabetes Dataset

```
                  CV F1    Test F1   Test AUC   Train Time   Interpretable?
Linear Regression  —       —         —          < 1 sec      ✅ (regression task)
Logistic Regression 0.79   0.78      0.87       < 1 sec      ✅
KNN (K=7)           0.75   0.73      0.83         5 sec       ❌
SVM (RBF)           0.82   0.81      0.89        20 sec       ❌
Naive Bayes         0.74   0.72      0.81       < 1 sec      Partial
Decision Tree       0.80   0.78      0.85         2 sec       ✅ (shallow)
Random Forest       0.87   0.86      0.92        45 sec       ❌ (SHAP needed)
XGBoost             0.91   0.89      0.95        90 sec       ❌ (SHAP needed)
LightGBM            0.91   0.89      0.95        30 sec       ❌ (SHAP needed)

Performance hierarchy for tabular data:
NB < KNN < DT < Logistic Reg ≈ SVM < RF ≈ LightGBM ≈ XGBoost
```

---

<a name="part-14"></a>

# PART 14 — Algorithm Selection Decision Guide

## Step-by-Step Framework

```
WHAT IS YOUR TASK?
│
├── Predict a continuous number (price, glucose, temperature)
│       ↓
│   → Linear Regression (start here — always)
│     + Ridge/Lasso if overfitting
│     + Random Forest / XGBoost for non-linear patterns
│
└── Predict a class label (yes/no, category)
        ↓
    HOW MUCH DATA?
    │
    ├── < 1,000 samples
    │       ↓
    │   → Naive Bayes (text/symptoms, speed critical)
    │     or SVM with RBF kernel (tabular, small data)
    │     or Logistic Regression (interpretability needed)
    │
    ├── 1,000 – 100,000 samples
    │       ↓
    │   → Random Forest (robust default, minimal tuning)
    │     or XGBoost (best performance with tuning)
    │     or Logistic Regression (if interpretability required)
    │
    └── > 100,000 samples
            ↓
        WHAT TYPE OF DATA?
        ├── Tabular → XGBoost / LightGBM
        └── Images / Text / Audio → Neural Networks
```

## When Each Algorithm Wins

| Algorithm | Best Scenario | Avoid When |
|---|---|---|
| **Linear Regression** | Continuous output, linear relationship, interpretability | Non-linear data without feature engineering |
| **Logistic Regression** | Strong baseline, interpretability required, legal/medical accountability | Non-linear boundaries without feature engineering |
| **KNN** | Small dataset, low dimensions, quick prototype, recommendation systems | Large dataset, high dimensions, need fast predictions |
| **SVM** | Small-medium data, high-dimensional features, genomics | > 100K rows (too slow), need probability outputs |
| **Naive Bayes** | Text classification, spam, very small data, need speed | Feature independence completely violated and accuracy matters |
| **Decision Tree** | Full interpretability required, explain every decision, rule extraction | Production models (overfits — use RF instead) |
| **Random Forest** | General-purpose baseline, feature importance needed | Need interpretability, memory-constrained |
| **XGBoost** | Tabular data competitions, maximum performance | Raw images/audio/text, need simple/fast model |

## The Practical Hierarchy (What Practitioners Actually Do)

```
Step 1: Always start with Logistic Regression / Linear Regression
        → Establishes a baseline
        → If this works well enough, ship it (simplicity wins)

Step 2: Try Random Forest
        → Handles non-linearity automatically
        → Built-in feature importance
        → Hard to mess up (robust to hyperparameters)

Step 3: Try XGBoost
        → Usually beats Random Forest with tuning
        → Use GridSearch + Cross-Validation to tune
        → Industry standard for tabular data

Step 4: If nothing works → Neural Networks
        → Only justified for large data or unstructured input
```

> **The Golden Rule: Start simple. Upgrade only when the simpler model is provably insufficient.**

---

<a name="part-15"></a>

# PART 15 — Quick Revision: All Formulae in One Place

## Model Formulae

```
── LINEAR REGRESSION ─────────────────────────────────────────────

Prediction:  y_hat  =  w1*x1 + w2*x2 + ... + wn*xn + b
Update:      w  :=  w  -  alpha x (dJ/dw)
Gradient:    dJ/dw  =  (2/n) x Sum[ (y_hat_i - y_i) x x_i ]
Ridge Cost:  J  =  MSE  +  lambda x Sum(w_i^2)
Lasso Cost:  J  =  MSE  +  lambda x Sum(|w_i|)

── LOGISTIC REGRESSION ───────────────────────────────────────────

Sigmoid:     P  =  1 / (1 + e^(-z))     where z = w^T*x + b
Decision:    y_hat = 1 if P >= 0.5,  else 0
Log Loss:    Loss = -[ y x log(p)  +  (1-y) x log(1-p) ]
Cost:        J  =  -(1/n) x Sum[ y_i x log(p_i) + (1-y_i) x log(1-p_i) ]
Reg. Cost:   J  =  Log_Loss  +  (1/C) x ||w||^2

── KNN ───────────────────────────────────────────────────────────

Prediction:   y_hat  =  mode { y_i : x_i in N_K(x) }
Euclidean:    d(x,x') = sqrt( Sum( (x_i - x'_i)^2 ) )
Manhattan:    d(x,x') = Sum( |x_i - x'_i| )
Minkowski:    d(x,x') = ( Sum( |x_i - x'_i|^p ) )^(1/p)

── SVM ───────────────────────────────────────────────────────────

Margin width:   2 / ||w||
Hard margin:    Minimise (1/2)||w||^2
                s.t. y_i x (w^T*x_i + b) >= 1 for all i
Soft margin:    Minimise (1/2)||w||^2 + C x Sum(slack_i)
RBF kernel:     K(x_i,x_j) = exp( -gamma x ||x_i - x_j||^2 )

── NAIVE BAYES ───────────────────────────────────────────────────

Posterior:    P(y|x) = P(x|y) x P(y) / P(x)
Naive assump: P(x|y) = P(x1|y) x P(x2|y) x ... x P(xn|y)
Gaussian:     P(xi|y) = (1/sqrt(2*pi*sigma^2)) x exp(-(xi-mu)^2 / (2*sigma^2))
Laplace:      P(xi|y) = (count(xi,y) + alpha) / (count(y) + alpha x K)
Log form:     log P(y|x) ∝ log P(y) + Sum_i[ log P(xi|y) ]

── DECISION TREE ─────────────────────────────────────────────────

Gini:         Gini(node) = 1 - Sum_k( p_k^2 )
Entropy:      Entropy(node) = - Sum_k( p_k x log2(p_k) )
Info Gain:    IG = Impurity(parent) - Sum[(|child|/|parent|) x Impurity(child)]
Var. Red.:    VR = Var(parent) - Sum[(|child|/|parent|) x Var(child)]

── RANDOM FOREST ─────────────────────────────────────────────────

Var(Forest) = rho x sigma^2  +  (1-rho) x sigma^2 / n
max_features = sqrt(n_features)  for classification
             = n_features / 3    for regression
OOB fraction ≈ 0.37              (~37% not sampled per bootstrap)

── XGBOOST ───────────────────────────────────────────────────────

Update:    F_t(x) = F_{t-1}(x) + eta x h_t(x)
Objective: Obj = Sum_i[L(yi,y_hat_i)] + Sum_t[gamma*T + (1/2)*lambda*Sum_j(w_j^2)]
Leaf wt:   w*_j = -(Sum g_i in leaf j) / (Sum h_i in leaf j + lambda)
Gain:      (1/2)*[G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G_P^2/(H_P+lambda)] - gamma
Gradients: g_i = y_hat_i - y_i;  h_i = y_hat_i x (1 - y_hat_i)
```

## Evaluation Metrics

```
── CLASSIFICATION ────────────────────────────────────────────────

Accuracy    =  (TP + TN)  /  (TP + TN + FP + FN)
Precision   =  TP  /  (TP + FP)
Recall      =  TP  /  (TP + FN)
F1          =  2 x Precision x Recall  /  (Precision + Recall)
Specificity =  TN  /  (TN + FP)
FPR         =  FP  /  (FP + TN)   ← x-axis of ROC curve

Log Loss    =  -(1/n) x Sum[ y x log(p) + (1-y) x log(1-p) ]

── REGRESSION ────────────────────────────────────────────────────

MAE   =  (1/n) x Sum( |y_hat - y| )
RMSE  =  sqrt( (1/n) x Sum( (y_hat - y)^2 ) )
R^2   =  1  -  SS_res / SS_tot
      =  1  -  Sum(y_hat-y)^2  /  Sum(y_mean-y)^2

── BIAS-VARIANCE ─────────────────────────────────────────────────

Total Error  =  Bias^2  +  Variance  +  Irreducible Noise
Var(Forest)  =  rho x sigma^2  +  (1-rho) x sigma^2 / n
```

---

<a name="part-16"></a>

# PART 16 — Complete Interview Q&A Bank

> 📖 Read each answer once before a viva. These cover every conceptual question asked about supervised ML.

---

### Q1. What is the difference between Loss and Cost Function?

Loss = error on a single training sample. Cost = average loss over the entire training dataset. Gradient Descent minimises the Cost, which implicitly minimises the average Loss. The distinction matters because Gradient Descent needs one scalar to minimise — it cannot optimise individual losses simultaneously.

---

### Q2. Why can't we use MSE as the cost function for Logistic Regression?

With MSE + sigmoid, the cost surface becomes **non-convex** — full of local minima. Gradient Descent may get permanently stuck at a suboptimal solution. Log Loss produces a **convex surface**, guaranteeing that gradient descent converges to the one global minimum, regardless of starting point.

---

### Q3. Why is Feature Scaling important for KNN and SVM but not for Decision Trees?

KNN uses distance metrics and SVM computes margins — features with large scales completely dominate these calculations. Decision Trees use threshold-based splits on individual features — only the relative ordering of values within a feature matters, not their absolute magnitude. Scale does not affect relative ordering. Random Forest and XGBoost inherit this tree property.

---

### Q4. What are Support Vectors?

The training points that lie exactly on the margin boundaries of an SVM. They are the **only** points that determine the decision boundary. Remove any non-support-vector patient from training and the boundary stays completely identical. This makes SVM **sparse** and robust to outliers far from the margin.

---

### Q5. What does the Kernel Trick do in SVM?

It implicitly maps data into a higher-dimensional feature space where it becomes linearly separable — without actually computing that space. Only dot products `K(xi, xj)` between pairs of original-space points are ever computed. For the RBF kernel, this effectively maps data into infinite-dimensional space, enabling highly non-linear boundaries while remaining computationally efficient.

---

### Q6. Hard Margin vs. Soft Margin SVM — what is the difference?

Hard Margin requires all training points to be correctly classified with zero violations — fails on noisy or non-separable data because no valid hyperplane exists. Soft Margin introduces slack variables and a C parameter that penalises violations, allowing controlled misclassification. Real-world data almost always uses Soft Margin.

---

### Q7. Why does Cross-Validation beat a single train/validation split?

A single split is noisy — performance depends heavily on which patients happened to fall in the validation set by chance. If those patients are unusually easy or hard to classify, the estimate is biased. K-Fold averages K different validation sets where every patient serves as validation exactly once, giving a much more stable, reliable estimate of generalisation performance.

---

### Q8. In medical screening, should you optimise Precision or Recall? Why?

**Recall.** Missing a real diabetic (False Negative) means the patient goes home untreated — potentially fatal. A false alarm (False Positive) leads to additional confirmatory tests — inconvenient but not dangerous. The asymmetric cost of errors dictates the metric. Minimising FN = maximising Recall. The choice of metric is fundamentally a clinical/business decision, not a mathematical one.

---

### Q9. What is the Bias-Variance Tradeoff?

Total error = Bias² + Variance + irreducible noise. Increasing model complexity reduces Bias (model can represent more complex patterns) but increases Variance (model becomes more sensitive to specific training data). Decreasing complexity does the opposite. The sweet spot — minimum total error — is found via hyperparameter tuning guided by cross-validation.

---

### Q10. LR and SVM both find a linear boundary. What is the fundamental difference?

LR finds the boundary that maximises the likelihood of the training data — **all patients contribute**. SVM finds the boundary that maximises the **geometric margin** — only the support vectors matter; all other patients are completely irrelevant. LR outputs calibrated probabilities; SVM outputs a raw decision score. When data is perfectly separable, LR finds infinitely many valid boundaries; SVM finds the **unique one** with the maximum margin.

---

### Q11. What is the "Naive" assumption in Naive Bayes, and why does it still work?

The naive assumption is that all features are conditionally independent given the class label — P(x|y) = product of P(xi|y). In reality, features like Glucose and BMI are correlated. However, the model's job is classification — deciding which class has the higher posterior, not computing accurate probabilities. Even when individual probability estimates are wrong, the **ranking** of classes is often correct. The correct ordering is enough for classification.

---

### Q12. What is Laplace Smoothing and why is it needed?

If a feature value never appeared in training data for a particular class, its count = 0, making P(xi|y) = 0. Since NB multiplies all feature probabilities, a single zero makes the entire posterior zero — the model can never assign that class regardless of all other evidence. Laplace smoothing adds a pseudocount alpha (usually 1) to all counts, ensuring every probability is positive and preventing the zero-product catastrophe.

---

### Q13. What is Information Gain and why is it used in Decision Trees?

Information Gain measures how much a proposed split reduces impurity at a node: IG = Impurity(parent) − weighted average Impurity(children). The algorithm evaluates every possible (feature, threshold) pair and chooses the one with maximum IG. This greedy, top-down approach builds the tree one split at a time. Using Entropy, IG directly measures the reduction in uncertainty (Shannon entropy) achieved by knowing the feature's value.

---

### Q14. Why does a single Decision Tree have high variance, and how does Random Forest fix it?

A single tree has high variance because small changes in training data can produce a completely different tree structure. Random Forest reduces variance through: (1) Bootstrap sampling — each tree trains on a different random subset, so each makes different errors. (2) Feature subsampling — at each split, only sqrt(n_features) are considered, decorrelating the trees. Mathematically: Var(Forest) = rho × sigma² + (1−rho) × sigma²/n. Lower correlation rho leads to greater variance reduction.

---

### Q15. What is Out-Of-Bag error?

About 37% of training samples are not included in any given tree's bootstrap sample — these are Out-Of-Bag (OOB) samples for that tree. Each training sample can be evaluated by all trees for which it was OOB. Averaging these evaluations gives the OOB error — a free, unbiased estimate of the model's generalisation performance, similar to cross-validation, without requiring a separate validation set.

---

### Q16. Explain Gradient Boosting in simple terms.

Gradient Boosting trains trees sequentially, where each new tree targets the mistakes of the previous ensemble. Start with a constant prediction (e.g., the base rate). Compute residuals — how wrong the current prediction is for each sample. Train a new shallow tree to predict these residuals. Add this tree (scaled by a small learning rate) to the ensemble. Compute new residuals and repeat. The word "gradient" means gradient descent in function space — each tree is a step in the direction that most reduces the loss function.

---

### Q17. What is the difference between Random Forest and XGBoost?

Random Forest builds trees **in parallel** using bootstrap sampling and feature subsampling, then averages — primarily reduces variance. XGBoost builds trees **sequentially** where each corrects residual errors of all previous trees — reduces both bias and variance. XGBoost uses explicit L1/L2 regularisation on leaf weights and second-order Taylor expansion (gradient + Hessian) for more accurate updates. Random Forest works well out-of-the-box; XGBoost achieves higher performance with proper tuning and dominates tabular data competitions.

---

### Q18. What is the role of the learning rate in XGBoost, and how does it interact with n_estimators?

The learning rate eta controls how much each new tree contributes: F_t = F_{t-1} + eta × tree_t. A small eta means tiny corrections — you need more trees but the final model generalises better. A large eta means big corrections — fewer trees but risk overshooting. Practical rule: use a small learning rate (0.05) + many trees (200–1000) + early stopping. Early stopping automatically finds the optimal n_estimators for any given learning rate.

---

### Q19. What is the difference between bagging and boosting?

**Bagging** trains multiple models in parallel, each on a different bootstrap sample, and combines by averaging or majority vote. The goal is to reduce variance. Example: Random Forest. **Boosting** trains models sequentially where each focuses on the errors of the previous ensemble. The goal is to reduce bias iteratively. Example: XGBoost, AdaBoost. Bagging is more robust and parallelisable; boosting achieves a higher performance ceiling but requires careful tuning to avoid overfitting.

---

### Q20. When would you choose Naive Bayes over Random Forest?

Choose Naive Bayes when: (1) Speed is critical — NB trains in milliseconds, ideal for real-time systems; (2) Very small datasets — RF can overfit with few samples while NB's simple statistics are stable; (3) Text classification — Multinomial NB with bag-of-words is a strong NLP baseline; (4) Streaming/online learning — NB can update parameters incrementally with each new observation. In practice, NB is often the first model tried in new text classification problems.

---

### Q21. Why is AUC-ROC better than Accuracy for model comparison?

Accuracy depends on the chosen decision threshold (default 0.5) and is misleading on imbalanced datasets. AUC-ROC is **threshold-independent** — it evaluates the model's inherent ability to discriminate between classes across all possible thresholds simultaneously. AUC = 0.92 means: given one random diabetic and one random healthy patient, the model assigns a higher probability to the diabetic 92% of the time — regardless of where you set the threshold.

---

### Q22. Quick-fire: Why these design choices?

| Question | Answer |
|---|---|
| Why sigmoid in Logistic Regression? | Squeezes any number into (0,1) so we get a valid probability |
| Why maximise margin in SVM? | The widest margin gives the most generalisation — farthest from both classes |
| Why odd K in KNN? | Avoids tie votes in binary classification |
| Why log-probabilities in Naive Bayes? | Prevents numerical underflow when multiplying many small numbers |
| Why Laplace smoothing? | Prevents zero probabilities from annihilating the entire class score |
| Why Random Forest beats Decision Tree? | Averaging uncorrelated trees cancels out variance |
| Why Boosting beats Bagging? | Sequential error correction reduces bias, not just variance |
| Why XGBoost uses second-order gradients? | More accurate step direction → fewer trees needed for same performance |
| Why feature scaling doesn't matter for trees? | Splits are based on rank/order of values, not their magnitude |
| Why KNN fails in high dimensions? | All points become equidistant — "nearest" loses meaning (curse of dimensionality) |

---

### Q23. The One Paragraph That Ties Everything Together

You start with the simplest model — **Linear or Logistic Regression** — because a simple model that works is always better than a complex one that barely outperforms it. When the data is non-linear, you move to **KNN** if the dataset is small and low-dimensional, **SVM** if it's medium-sized and high-dimensional, or **Naive Bayes** if it's text or you have very little data and need speed. When you need raw power on tabular data, **Decision Trees** give you interpretability at the cost of overfitting — so you almost always use **Random Forest** instead, which buys robustness through diversity. When you want the best possible performance, **XGBoost** wins by learning sequentially from its own mistakes. Everything from Random Forest onward is an **Ensemble Method** — combining weak learners to produce a strong one, either in parallel (Bagging) or in sequence (Boosting). The right choice always depends on your data size, data type, interpretability requirements, and how much you're willing to trade performance for simplicity. But the discipline never changes: split your data honestly, tune with cross-validation, measure with the right metric for your domain, and always ask — *"How wrong is my model, and how do I make it less wrong, honestly?"*

---

> **The one truth that unifies all of supervised ML:**
>
> *Naive Bayes asks what the data looked like when it was generated.*
> *Linear and Logistic Regression find the best weighted straight line through it.*
> *KNN asks the nearest neighbours to vote.*
> *SVM finds the boundary with the most breathing room.*
> *Decision Trees ask yes/no questions until they reach an answer.*
> *Random Forest asks 100 differently trained experts and takes a majority vote.*
> *XGBoost asks what the current ensemble is still getting wrong — and fixes it.*
>
> *All eight ultimately answer the same question:*
> *Given these features, what is the most honest prediction I can make?*
> *They just approach that honesty from completely different directions.*
