# The Complete ML Learning Guide
### Linear Regression · Logistic Regression · KNN · SVM
#### Algorithms → Loss Functions → Hyperparameters → Cross-Validation → Evaluation Metrics

---

> 🏥 **One real-world anchor used throughout this entire guide:**
> A hospital has data on **1,000 patients** — Age, BMI, Blood Pressure, Glucose Level, Insulin.
>
> | Question | Task Type | Algorithm(s) |
> |---|---|---|
> | *"What will this patient's exact Blood Sugar be next month?"* | Regression | Linear Regression |
> | *"Will this patient be Diabetic or Not Diabetic?"* | Classification | Logistic Regression / KNN / SVM |
>
> Same patients. Same features. Two completely different tasks. Every concept in this guide connects back to one of these two questions.

---

## Table of Contents

- [Chapter 1 — The Golden Rule: Split Your Data First](#chapter-1)
- [Chapter 2 — Linear Regression](#chapter-2)
  - [2.1 What the Model Learns](#21-what-the-model-learns)
  - [2.2 Error → Loss → Cost Function](#22-error--loss--cost-function-the-hierarchy)
  - [2.3 Gradient Descent](#23-gradient-descent)
  - [2.4 Feature Scaling](#24-feature-scaling)
  - [2.5 Regularisation](#25-regularisation)
  - [2.6 Cross-Validation](#26-cross-validation)
  - [2.7 Evaluation Metrics for Regression](#27-evaluation-metrics-for-regression)
- [Chapter 3 — The Three Classification Algorithms](#chapter-3)
  - [3.1 Core Philosophy of Each](#31-the-core-philosophy-of-each)
  - [3.2 Logistic Regression](#32-logistic-regression)
  - [3.3 K-Nearest Neighbors (KNN)](#33-k-nearest-neighbors-knn)
  - [3.4 Support Vector Machine (SVM)](#34-support-vector-machine-svm)
  - [3.5 LR vs SVM — Key Difference](#35-lr-vs-svm--they-both-find-a-linear-boundary-so-whats-different)
- [Chapter 4 — Hyperparameter Tuning & Cross-Validation](#chapter-4)
  - [4.1 Why Default Models Fail](#41-why-default-models-fail)
  - [4.2 What Each Hyperparameter Controls](#42-what-each-hyperparameter-actually-controls)
  - [4.3 Why Cross-Validation](#43-why-cross-validation--the-problem-it-solves)
  - [4.4 Full Tuning Pipeline](#44-the-full-tuning-pipeline-in-practice)
  - [4.5 5-Fold vs 10-Fold vs LOOCV](#45-5-fold-vs-10-fold-vs-loocv)
- [Chapter 5 — Bias, Variance & The Tradeoff](#chapter-5)
  - [5.1 Definitions](#51-what-are-they-really)
  - [5.2 The Archery Mental Model](#52-the-archery-mental-model)
  - [5.3 Diagnosing the Problem](#53-diagnosing-the-problem--the-2-number-test)
  - [5.4 Where Every Model Sits](#54-where-every-model-setting-sits)
- [Chapter 6 — The Confusion Matrix](#chapter-6)
  - [6.1 Why All Three Classifiers Meet Here](#61-why-all-three-classifiers-meet-here)
  - [6.2 Building the Confusion Matrix](#62-building-the-confusion-matrix)
  - [6.3 How Each Algorithm Produces ŷ](#63-how-each-algorithm-internally-produces-y)
- [Chapter 7 — Evaluation Metrics: Complete Guide](#chapter-7)
  - [7.1 Classification Metrics](#71-classification-metrics--all-formulae)
  - [7.2 Regression Metrics](#72-regression-metrics--side-by-side)
  - [7.3 Choosing the Right Metric](#73-choosing-the-right-metric--decision-guide)
  - [7.4 Domain Cheat Sheet](#74-domain-cheat-sheet)
- [Chapter 8 — The Complete Mental Model](#chapter-8)
- [Chapter 9 — Quick Revision: All Formulae in One Place](#chapter-9)
- [Chapter 10 — Viva / Interview Q&A Bank](#chapter-10)

---

<a name="chapter-1"></a>

# CHAPTER 1 — The Golden Rule: Split Your Data First

Before any algorithm sees any data, before any hyperparameter is chosen, before any model is built — you do **one thing first**:

```
1,000 Patients
│
├── 800 patients  →  Training Pool  (all learning happens here)
│
└── 200 patients  →  Test Set 🔒   (locked vault — never opened until the very end)
```

**Why is this non-negotiable?**

The test set simulates patients the hospital has *never seen before*. The moment you use the test set to make any decision — even once — it is no longer unseen data. Every tuning decision, every model comparison, every hyperparameter choice must be made using only the 800 training patients. The test set is opened exactly **once**, at the very end, to report the final honest performance.

> ⚠️ **The Rule:** Fit everything (scalers, imputers, encoders, models) on training data only. Apply to test data. Never the reverse.

---

<a name="chapter-2"></a>

# CHAPTER 2 — Linear Regression (The Regression Task)

<a name="21-what-the-model-learns"></a>

## 2.1 What the Model Learns

The doctor wants to predict an exact glucose level — a continuous number like 142 mg/dL. Linear Regression fits a weighted sum through the data:

```
Glucose  =  w1*Age  +  w2*BMI  +  w3*BP  +  w4*Insulin  +  b
```

| Symbol | Meaning | Learned how? |
|---|---|---|
| `w1, w2, w3, w4` | Weights — how much each feature contributes | Gradient Descent |
| `b` | Bias / intercept — baseline prediction | Gradient Descent |
| `Glucose` | Predicted output (y_hat) | Output of the formula |

---

<a name="22-error--loss--cost-function-the-hierarchy"></a>

## 2.2 Error → Loss → Cost Function: The Hierarchy

This is the most important conceptual chain in all of supervised learning.

### Error (one patient)

The raw difference between prediction and truth:

```
Error_i  =  y_hat_i  -  y_i
```

**Example:** Model predicts 167 mg/dL, actual is 163 mg/dL → Error = +4.

> If you simply average raw errors, positives and negatives cancel out — giving a misleadingly near-zero score even when predictions are way off. This is why we need proper loss functions.

---

### Loss Function (one prediction penalised)

| Loss | Formula | Intuition | Outlier Sensitive? |
|---|---|---|---|
| **MAE** | `Mean( |y_hat - y| )` | Average absolute gap. Off by 5 mg/dL on average. | No — all errors treated equally |
| **MSE** | `Mean( (y_hat - y)^2 )` | Squares large errors — punishes bad predictions harder. | Yes — one huge error dominates |
| **RMSE** | `sqrt( MSE )` | Same units as target (mg/dL). Directly interpretable. | Yes |

---

### Cost Function J(w, b) — what the model minimises

The average MSE loss over ALL 800 training patients:

```
J(w, b)  =  (1/n) * Sum[ (y_hat_i  -  y_i)^2 ]
```

> 💡 **The hierarchy in one line:**
> **Error** = mistake on 1 patient.
> **Loss** = how that mistake is penalised.
> **Cost** = average penalty across all 800 patients.
> **Gradient Descent** minimises the Cost.

---

<a name="23-gradient-descent"></a>

## 2.3 Gradient Descent — The Engine That Learns the Weights

Gradient Descent finds the weights `w` that make `J(w, b)` as small as possible.

**Intuition:** You are blindfolded on a hilly landscape. `J` is your altitude. You feel the slope under your feet (the gradient) and take one small step downhill. Repeat until you reach the valley.

**Update rule:**

```
w  :=  w  -  alpha * (dJ/dw)
```

**The gradient:**

```
dJ/dw  =  (2/n) * Sum[ (y_hat_i - y_i) * x_i ]
```

**`alpha` is the learning rate** — the size of each step:

| alpha | Effect | Problem |
|---|---|---|
| Too large (e.g. 0.9) | Giant steps | Overshoots the minimum — cost oscillates or explodes |
| Too small (e.g. 0.000001) | Tiny steps | Converges painfully slowly |
| Just right (e.g. 0.01) | Smooth descent | Converges efficiently ✅ |

---

<a name="24-feature-scaling"></a>

## 2.4 Feature Scaling — Why It's Critical Before Gradient Descent

**The problem:** Age ranges 20–80. Insulin ranges 0–800. Gradient Descent takes steps proportional to feature magnitude — so Insulin dominates and the descent zig-zags wildly.

### Fix 1 — Standardisation (Z-score)

Transforms every feature to **mean = 0, std = 1**:

```
x_scaled  =  (x  -  mean(x))  /  std(x)
```

### Fix 2 — Min-Max Normalisation

Transforms every feature to **range [0, 1]**:

```
x_scaled  =  (x  -  x_min)  /  (x_max  -  x_min)
```

> ⚠️ **Critical rule:** Always fit the scaler on **training data only**. Apply the same scaler to test data. Never fit on test data — that would leak test information into training.

**Which algorithms need scaling?**

| Algorithm | Needs Scaling? | Reason |
|---|---|---|
| Linear Regression (GD) | ✅ Yes | Gradient steps proportional to magnitude |
| Logistic Regression (GD) | ✅ Yes | Same as above |
| KNN | ✅ Yes | Distance calculations dominated by large-scale features |
| SVM | ✅ Yes | Margin and kernel computations affected |
| Decision Tree | ❌ No | Threshold-based splits — scale doesn't change ordering |
| Random Forest / XGBoost | ❌ No | Tree-based — scale invariant |

---

<a name="25-regularisation"></a>

## 2.5 Regularisation — The Hyperparameter That Prevents Overfitting

Without regularisation, the model assigns enormous weights to memorise the 800 training patients, then fails on new ones.

**Ridge (L2) Regularisation:**

```
J_regularised  =  MSE  +  lambda * Sum( w_i^2 )
```

The `lambda` term penalises large weights:

| lambda | Weights | Model Behaviour |
|---|---|---|
| Too large | Forced to ~0 | Flat prediction — **underfitting (High Bias)** |
| Too small | Can grow huge | Memorises training data — **overfitting (High Variance)** |
| Just right | Balanced | Captures true pattern — **generalises** ✅ |

---

<a name="26-cross-validation"></a>

## 2.6 Cross-Validation — Finding the Right Lambda

You cannot tune `lambda` using the test set (locked vault). Instead, use 5-fold cross-validation on the 800 training patients:

```
Fold 1: [160 Val | 640 Train] → lambda=0.01 → Val RMSE = 18.2
Fold 2: [160 Val | 640 Train] → lambda=0.01 → Val RMSE = 17.8
Fold 3: [160 Val | 640 Train] → lambda=0.01 → Val RMSE = 18.9
Fold 4: [160 Val | 640 Train] → lambda=0.01 → Val RMSE = 17.5
Fold 5: [160 Val | 640 Train] → lambda=0.01 → Val RMSE = 18.6
                                               Average  = 18.2  ← Score for lambda=0.01

Repeat for lambda=0.1  → Average RMSE = 15.4
Repeat for lambda=1.0  → Average RMSE = 14.1  ✅  ← Best
Repeat for lambda=10   → Average RMSE = 16.8
```

**Best lambda = 1.0.** Retrain on all 800 patients with this value. Then — and only then — open the vault.

---

<a name="27-evaluation-metrics-for-regression"></a>

## 2.7 Evaluation Metrics for Regression

After opening the test vault with the final model:

| Metric | Formula | Our Result | Interpretation |
|---|---|---|---|
| MAE | `Mean( |y_hat - y| )` | 4.3 mg/dL | On average, off by 4.3 mg/dL |
| RMSE | `sqrt( Mean( (y_hat-y)^2 ) )` | 6.1 mg/dL | Large errors penalised more |
| R² | `1  -  SS_res / SS_tot` | 0.89 | Model explains 89% of variance |

**Interpreting R²:**

| R² | Meaning |
|---|---|
| 1.0 | Perfect — model explains everything |
| 0.89 | Excellent — strong model |
| 0.5 | Moderate — better than guessing, but half the story is missing |
| 0.0 | Useless — same as predicting the mean for everyone |
| < 0 | Actively worse than the mean predictor — something is very wrong |

> ✅ Always report both **RMSE** (surfaces large errors) and **R²** (explains how much variance is captured). MAE is most business-friendly — a doctor instantly understands "off by 4.3 mg/dL."

> ❌ No confusion matrix for regression — the output is a continuous number, not a class label.

---

<a name="chapter-3"></a>

# CHAPTER 3 — The Three Classification Algorithms

The same 800 training patients, but the target is now binary: **Diabetic (1)** or **Not Diabetic (0)**.

<a name="31-the-core-philosophy-of-each"></a>

## 3.1 The Core Philosophy of Each

> **Logistic Regression:** *"What is the probability of this patient belonging to the Diabetic class?"*
>
> **KNN:** *"What do this patient's nearest neighbours look like — what class are they?"*
>
> **SVM:** *"Where is the safest, widest possible boundary between the two classes?"*

Three genuinely different answers to the same problem — yet they all converge to the same evaluation framework at the end.

---

<a name="32-logistic-regression"></a>

## 3.2 Logistic Regression

### The Sigmoid Function

Linear Regression can predict values like −0.3 or 1.7 — meaningless for a probability. The sigmoid squashes any real number into (0, 1):

```
P(Diabetic=1 | x)  =  1 / (1  +  e^(-z))       where  z = w^T * x + b
```

- If `P >= 0.5` → predict **Diabetic**
- If `P < 0.5` → predict **Not Diabetic**

**Decision Boundary:** Where `P = 0.5` → `z = 0` → `w^T*x + b = 0` (a linear hyperplane).

---

### Log Loss — Why Not MSE?

With MSE + sigmoid, the cost surface becomes **non-convex** — full of local minima. Gradient Descent can get stuck. Log Loss gives a **convex surface**, guaranteeing convergence to the global minimum.

**Loss for one patient:**

```
Loss  =  -[ y * log(p)  +  (1-y) * log(1-p) ]
```

**Cost over all n patients:**

```
J  =  -(1/n) * Sum[ y_i * log(p_i)  +  (1-y_i) * log(1-p_i) ]
```

| Situation | Loss |
|---|---|
| Patient IS diabetic (y=1), model says p=0.95 | Very small ✅ |
| Patient IS diabetic (y=1), model says p=0.50 | Medium |
| Patient IS diabetic (y=1), model says p=0.05 | Very large ❌ |

> 💡 Gradient Descent still minimises this cost — **same engine as Linear Regression, different loss formula**.

---

### Hyperparameter: C (Regularisation)

In scikit-learn, `C = 1/lambda` (inverted convention):

```
Cost  =  Log_Loss  +  (1/C) * ||w||^2
```

| C | Regularisation | Boundary | Risk |
|---|---|---|---|
| High C (e.g. 100) | Weak | Complex | Overfitting |
| Low C (e.g. 0.001) | Strong | Simple | Underfitting |
| C = 1.0 (default) | Moderate | Balanced | Sweet spot ✅ |

---

### Nature: Probabilistic · Global · Parametric

- **Global** — every single training patient influences the weights
- **Probabilistic** — gives calibrated probabilities, not just labels
- Foundation of the output layer in neural networks (softmax = multinomial logistic regression)

---

<a name="33-k-nearest-neighbors-knn"></a>

## 3.3 K-Nearest Neighbors (KNN)

### The Model — No Training, Pure Memory

KNN is a **lazy learner**. It stores all 800 training patients and does nothing during training. When a new patient arrives:

1. Compute distance to all 800 training patients
2. Find the K closest ones
3. Take a majority vote

```
y_hat  =  mode { y_i : x_i in N_K(x) }
```

There is **no explicit decision boundary** — it emerges as a Voronoi-like partition of feature space.

---

### Distance Formulae

| Distance | Formula | When to use |
|---|---|---|
| Euclidean (L2) | `sqrt( Sum( (x_i - x_j)^2 ) )` | Default. Continuous features, similar scales. |
| Manhattan (L1) | `Sum( |x_i - x_j| )` | When outliers are present. Less sensitive to extremes. |
| Minkowski | `( Sum( |x_i - x_j|^p ) )^(1/p)` | Generalisation: p=1 → Manhattan, p=2 → Euclidean. |

> ⚠️ **WHY FEATURE SCALING IS CRITICAL FOR KNN:**
> If Age is 0–80 and Insulin is 0–800, Insulin completely dominates the distance calculation. Standardise ALL features before KNN.

---

### Hyperparameter: K

| K value | Boundary | Bias-Variance |
|---|---|---|
| K = 1 | Jagged — every point is its own island | Low Bias, Very High Variance → **Overfit** |
| K = 5 to 15 | Smooth — captures the true pattern | Balanced → **Sweet Spot** ✅ |
| K = n (all) | Flat — predicts majority class for everyone | Very High Bias, Low Variance → **Underfit** |

> **Rule:** K should be **odd** (avoids ties). Common starting point: `K = sqrt(n_training)`.

---

### Nature: Non-Parametric · Local · Instance-Based

- **No cost function** and **no gradient descent** (frequent exam question!)
- Training cost: O(1) — just store data
- Inference cost: O(n × d) — scan all training points per query

---

<a name="34-support-vector-machine-svm"></a>

## 3.4 Support Vector Machine (SVM)

### The Margin — The Core Geometry

SVM finds the boundary with the **maximum margin**: the widest possible gap between the two classes.

```
Margin width  =  2 / ||w||
```

The training points on the margin edges are called **Support Vectors** — the *only* points that determine the final boundary.

**The Optimisation Problem:**

```
Minimise:    (1/2) * ||w||^2
Subject to:  y_i * (w^T * x_i  +  b)  >=  1     for all i
```

Maximising margin = minimising `||w||`. This is a convex quadratic programming problem with a **unique global solution**.

---

### Hard Margin vs. Soft Margin

| Type | What it does | When it breaks | Real-world usage |
|---|---|---|---|
| **Hard Margin** | Zero tolerance — all points outside the margin | Fails on noisy/non-separable data | Almost never |
| **Soft Margin** | Allows controlled violations with penalty C | Needs C tuned | **Almost always** ✅ |

**Soft Margin objective:**

```
Minimise:  (1/2) * ||w||^2  +  C * Sum( slack_i )
```

| C | Margin | Risk |
|---|---|---|
| High C | Narrow, strict → penalises violations heavily | Overfitting |
| Low C | Wide, tolerant → allows more violations | Underfitting |

---

### The Kernel Trick — SVM's Superpower

Real patient data is rarely linearly separable. The kernel trick implicitly maps data into a higher-dimensional space where it *becomes* linearly separable — without ever explicitly computing that space.

```
K(x_i, x_j)  =  phi(x_i)^T * phi(x_j)
```

| Kernel | Formula | Intuition |
|---|---|---|
| Linear | `x_i^T * x_j` | No transformation. Use when data is already separable. |
| RBF / Gaussian | `exp( -gamma * ||x_i - x_j||^2 )` | Smooth non-linear boundaries. Most widely used. |
| Polynomial | `( x_i^T * x_j  +  c )^d` | Polynomial surface. Good for image data. |

---

### Hyperparameters: C and Gamma (RBF) — Must Be Tuned Together

| Parameter | High value | Low value |
|---|---|---|
| **C** | Hard margin → overfitting | Soft margin → underfitting |
| **gamma** | Each point influences only nearby area → wiggly boundary → overfit | Each point influences wide area → smooth boundary → underfit |

---

### Nature: Geometric · Margin-Based · Sparse

- **Geometric** — thinks in distances and margins, not probabilities
- **Sparse** — only support vectors matter; all other patients are irrelevant
- **Robust** — outliers far from the margin have zero influence on the boundary

---

<a name="35-lr-vs-svm--they-both-find-a-linear-boundary-so-whats-different"></a>

## 3.5 LR vs. SVM — They Both Find a Linear Boundary. So What's Different?

Both find `w^T*x + b = 0`. The difference is *what they optimise* and *which patients they care about*:

| Dimension | Logistic Regression | SVM |
|---|---|---|
| **Objective** | Maximise likelihood (probabilistic) | Maximise geometric margin |
| **Loss Function** | Log-loss / cross-entropy | Hinge loss |
| **Which patients matter** | ALL — every patient contributes | ONLY support vectors — rest are irrelevant |
| **Outlier sensitivity** | Higher — outliers pull the weights | Lower — outliers far from margin ignored |
| **Output** | Calibrated probability [0, 1] | Raw decision score (not a probability) |
| **Non-linearity** | Needs manual feature engineering | Kernel trick handles it natively |
| **Solution method** | Gradient Descent | Convex Quadratic Programming |
| **Theoretical basis** | MLE / PAC Learning | Structural Risk Minimization (Vapnik, 1995) |

> **Key geometric insight:** When data is perfectly linearly separable, LR finds infinitely many valid hyperplanes. SVM finds the **unique one** with the maximum margin — theoretically the best-generalising solution by VC dimension theory.

---

<a name="chapter-4"></a>

# CHAPTER 4 — Hyperparameter Tuning & Cross-Validation

<a name="41-why-default-models-fail"></a>

## 4.1 Why Default Models Fail

Running any model with default settings gives you a random guess at the complexity level:

```
UNDERFITTING                         OVERFITTING
(Model too simple)                   (Model too complex)

Training Accuracy:  62%              Training Accuracy:  99%
Test Accuracy:      60%              Test Accuracy:      61%

Failed the exam without studying.    Crammed the textbook including typos.
                                     Failed the real exam.
```

Hyperparameter tuning finds the **sweet spot** between these two extremes.

---

<a name="42-what-each-hyperparameter-actually-controls"></a>

## 4.2 What Each Hyperparameter Actually Controls

| Algorithm | Hyperparameter | Low value → | High value → |
|---|---|---|---|
| Linear Regression | lambda | Overfit (no penalty on weights) | Underfit (weights all ~0) |
| Logistic Regression | C (= 1/lambda) | Underfit (too simple) | Overfit (too complex) |
| KNN | K | Overfit (K=1 is chaotic) | Underfit (K=n is flat) |
| SVM | C | Underfit (soft margin) | Overfit (hard margin) |
| SVM | gamma (RBF) | Underfit (broad influence) | Overfit (narrow influence) |

Every hyperparameter is a **complexity dial** — turning it one way increases complexity (lower bias, higher variance), the other way decreases it (higher bias, lower variance).

---

<a name="43-why-cross-validation--the-problem-it-solves"></a>

## 4.3 Why Cross-Validation — The Problem It Solves

**The forbidden mistake:**

```
Step 1: Train model on training data
Step 2: Tune hyperparameters until test accuracy looks good  ← WRONG
Step 3: Report that test accuracy

❌ The test set is now contaminated. Your reported accuracy is fake.
```

Cross-validation creates a **validation layer** inside the training pool — the test set stays locked:

```
Full Dataset (1,000 patients)
│
├── Test Set (200 patients) 🔒  ← NEVER touched during tuning
│
└── Training Pool (800 patients)
    ├── Fold 1: [Val 160 | Train 640]
    ├── Fold 2: [Train 160 | Val 160 | Train 480]
    ├── Fold 3: [Train 320 | Val 160 | Train 320]
    ├── Fold 4: [Train 480 | Val 160 | Train 160]
    └── Fold 5: [Train 640 | Val 160]
```

For each hyperparameter value: train 5 times, validate on a different fold each time, **average the 5 scores**. This average is a stable, unbiased estimate of generalisation.

---

<a name="44-the-full-tuning-pipeline-in-practice"></a>

## 4.4 The Full Tuning Pipeline in Practice

**For Logistic Regression (single hyperparameter C):**

```
Try C=0.001 → 5-fold CV → avg accuracy = 74%
Try C=0.01  → 5-fold CV → avg accuracy = 79%
Try C=0.1   → 5-fold CV → avg accuracy = 83%
Try C=1.0   → 5-fold CV → avg accuracy = 86%  ✅  Best
Try C=10    → 5-fold CV → avg accuracy = 84%
Try C=100   → 5-fold CV → avg accuracy = 81%
```

**For SVM (two hyperparameters — Grid Search):**

```
              gamma=0.001   gamma=0.01   gamma=0.1
C=0.1    [       78%           81%          76%   ]
C=1.0    [       82%           87%          83%   ]  ← Best: C=1.0, gamma=0.01
C=10     [       83%           85%          79%   ]
C=100    [       84%           82%          71%   ]
```

**After tuning:** Retrain on all 800 patients with best hyperparameters → open the vault exactly **once** → report the honest final accuracy.

---

<a name="45-5-fold-vs-10-fold-vs-loocv"></a>

## 4.5 5-Fold vs. 10-Fold vs. LOOCV

| Method | Folds | Train size/fold | Variance of estimate | Compute cost | Best for |
|---|---|---|---|---|---|
| 5-Fold | 5 | 80% | Moderate | Low | Standard choice for most datasets |
| 10-Fold | 10 | 90% | Lower | Higher | Larger datasets, more reliable estimate |
| LOOCV | n | ~100% | Lowest | Very High | Tiny datasets (< 100 samples) |

---

<a name="chapter-5"></a>

# CHAPTER 5 — Bias, Variance & The Tradeoff

<a name="51-what-are-they-really"></a>

## 5.1 What Are They, Really?

**Bias** — Error from wrong assumptions. A straight line trying to fit a curve will always be wrong, no matter how much data you give it.

**Variance** — Error from sensitivity to specific training data. A K=1 KNN draws completely different boundaries if you swap just 5 of the 800 training patients.

**Irreducible Error** — Randomness in the real world no model can eliminate (e.g., a healthy patient randomly develops diabetes).

**The fundamental equation:**

```
Total Error  =  Bias^2  +  Variance  +  Irreducible Noise
```

---

<a name="52-the-archery-mental-model"></a>

## 5.2 The Archery Mental Model

```
HIGH BIAS + LOW VARIANCE        LOW BIAS + HIGH VARIANCE
(Consistently wrong)            (Right area, all over the place)

    ● ●                               ●
     ●                             ●     ●
    ● ●                               ●
Clustered but off-centre.         Scattered but centred.

HIGH BIAS + HIGH VARIANCE       LOW BIAS + LOW VARIANCE ✅
(Scattered AND wrong)            (Clustered AND centred = GOAL)

  ●    ●                              ●●
     ●                                ●
●       ●                            ●●
Worst of both worlds.            Bullseye every time.
```

The **model** is the archer. The **bullseye** is the true real-world pattern. The **arrows** are predictions.

---

<a name="53-diagnosing-the-problem--the-2-number-test"></a>

## 5.3 Diagnosing the Problem — The 2-Number Test

Just look at two numbers:

| Training Accuracy | Test Accuracy | Diagnosis | Fix |
|---|---|---|---|
| Low (55%) | Low (54%) | **HIGH BIAS — Underfitting** | More complex model, more features, lower regularisation |
| High (97%) | Low (62%) | **HIGH VARIANCE — Overfitting** | More data, regularise, increase K, lower C |
| High (87%) | High (85%) | **SWEET SPOT** ✅ | Deploy |

---

<a name="54-where-every-model-setting-sits"></a>

## 5.4 Where Every Model Setting Sits

```
← HIGH BIAS (Underfit)                               HIGH VARIANCE (Overfit) →

Linear Reg       Logistic Reg      SVM           KNN          KNN
high lambda      low C             C=1, g=0.01   K=7          K=1
                                   (sweet spot)
```

Every hyperparameter moves the model along this spectrum. Cross-validation tells you where you are. Tuning moves you toward the sweet spot.

---

<a name="chapter-6"></a>

# CHAPTER 6 — The Confusion Matrix: Universal Convergence Point

<a name="61-why-all-three-classifiers-meet-here"></a>

## 6.1 Why All Three Classifiers Meet Here

LR, KNN, and SVM work completely differently internally — but they all fulfil the same contract:

```
f : X  →  Y
    Features  →  Class Label
```

Every algorithm produces one thing for each patient: a predicted label `y_hat ∈ {0, 1}`. The confusion matrix only ever sees `(y_true, y_hat)` pairs. It does **not** know or care whether that label came from a sigmoid, a majority vote, or a margin hyperplane.

**Example — three models see the same new patient:**

| Algorithm | Internal mechanism | Final output |
|---|---|---|
| Logistic Regression | Computes P(Diabetic) = 0.73 → 0.73 > 0.5 | **y_hat = 1 (Diabetic)** |
| KNN | 5 neighbours: 4 Diabetic, 1 Not → majority vote | **y_hat = 1 (Diabetic)** |
| SVM | Computes w^T*x + b = +2.3 → positive side | **y_hat = 1 (Diabetic)** |

All three said y_hat = 1. The confusion matrix sees: *"You predicted 1, the truth was 1 — that's a True Positive."*

---

<a name="62-building-the-confusion-matrix"></a>

## 6.2 Building the Confusion Matrix

Open the vault: 200 test patients — 120 Not-Diabetic, 80 Diabetic.

```
                          PREDICTED
                    Not Diabetic (0)    Diabetic (1)
         ┌─────────────────────────────────────────────┐
A   Not  │                             │               │
C  Diab. │    TN = 110  ✅             │   FP = 10  ❌ │
T   (0)  │                             │  Type I Error │
U        ├─────────────────────────────────────────────┤
A  Diab. │                             │               │
L   (1)  │    FN = 11   ❌             │   TP = 69  ✅ │
         │  Type II Error              │               │
         └─────────────────────────────────────────────┘
```

| Cell | Full Name | What happened | Consequence |
|---|---|---|---|
| **TP = 69** | True Positive | Said Diabetic, Was Diabetic ✅ | Correct — patient gets treatment |
| **TN = 110** | True Negative | Said Healthy, Was Healthy ✅ | Correct — no unnecessary intervention |
| **FP = 10** | False Positive (Type I) | Said Diabetic, Was Healthy ❌ | Unnecessary tests — stressful, not fatal |
| **FN = 11** | False Negative (Type II) | Said Healthy, Was Diabetic ❌ | **Patient goes home untreated — potentially fatal** |

> 💡 **Memory trick:** The first word (True/False) = was the prediction correct? The second word (Positive/Negative) = what did the model predict?

---

<a name="63-how-each-algorithm-internally-produces-y"></a>

## 6.3 How Each Algorithm Internally Produces ŷ

**Logistic Regression** — threshold τ converts probability to label:

```
y_hat = 1   if  P(y=1|x) >= tau     (default tau = 0.5)
y_hat = 0   otherwise
```

Sweeping τ from 0 to 1 traces the **ROC curve**.

**KNN** — implicit threshold is K/2:

```
y_hat  =  mode { y_i : x_i in N_K(x) }
```

Changing K → changes neighbourhoods → changes the confusion matrix.

**SVM** — threshold at the hyperplane (score = 0):

```
y_hat  =  sign( w^T*x  +  b )
```

No natural probability output. For ROC curves, the raw decision score is used. **Platt Scaling** is needed to convert SVM scores into calibrated probabilities.

---

<a name="chapter-7"></a>

# CHAPTER 7 — Evaluation Metrics: Complete Guide

<a name="71-classification-metrics--all-formulae"></a>

## 7.1 Classification Metrics — All Formulae

All metrics flow from the same four numbers: **TP, TN, FP, FN**. Computed identically regardless of which model produced them.

**Three models, same 200 test patients:**

```
              Logistic Reg    KNN (K=7)    SVM (C=1, gamma=0.01)
TN               108             105              110
FP                12              15               10
FN                15              18               11
TP                65              62               69
```

---

### Accuracy

```
Accuracy  =  (TP + TN)  /  (TP + TN + FP + FN)
```

| Model | Calculation | Result |
|---|---|---|
| Logistic Reg | (65+108)/200 | 86.5% |
| KNN | (62+105)/200 | 83.5% |
| SVM | (69+110)/200 | **89.5%** ✅ |

> ⚠️ **The Accuracy Trap:** On an imbalanced dataset (950 healthy, 50 diabetic), a model that predicts everyone is healthy scores 95% accuracy — yet misses every single diabetic. **Never trust Accuracy alone on imbalanced data.**

---

### Precision

```
Precision  =  TP  /  (TP + FP)
```

| Model | Calculation | Result |
|---|---|---|
| Logistic Reg | 65/(65+12) | 84.4% |
| KNN | 62/(62+15) | 80.5% |
| SVM | 69/(69+10) | **87.3%** ✅ |

*Of all patients flagged as Diabetic, what fraction actually were?*

> Optimise Precision when **False Positives are costly** (spam filter, legal decisions).

---

### Recall (Sensitivity / True Positive Rate)

```
Recall  =  TP  /  (TP + FN)
```

| Model | Calculation | Result |
|---|---|---|
| Logistic Reg | 65/(65+15) | 81.3% |
| KNN | 62/(62+18) | 77.5% |
| SVM | 69/(69+11) | **86.3%** ✅ |

*Of all patients who actually were Diabetic, what fraction did we catch?*

> Optimise Recall when **False Negatives are costly** — in medical screening, this is almost always the priority.

---

### The Precision-Recall Tradeoff

| Threshold | Strategy | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| 0.7 (strict) | Only flag when very confident | High ↑ | Low ↓ | Few | Many |
| 0.5 (default) | Balanced | 87.3% | 86.3% | 10 | 11 |
| 0.3 (lenient) | Flag aggressively | Low ↓ | High ↑ | Many | Few |

> There is no universally right threshold — it is a clinical/business decision based on the relative cost of FP vs FN in your domain.

---

### F1 Score

```
F1  =  2 * (Precision * Recall)  /  (Precision + Recall)
```

| Model | Result |
|---|---|
| Logistic Reg | 82.8% |
| KNN | 79.0% |
| SVM | **86.8%** ✅ |

**Why harmonic mean and not simple average?**

A model with Precision=90% and Recall=10%:
- Arithmetic mean = **50%** — looks passable
- F1 = **18%** — correctly reveals this model is terrible

The harmonic mean is always closer to the *smaller* value. A model cannot hide terrible Recall behind great Precision.

> Use F1 when you want one balanced number, especially on imbalanced datasets.

---

### Specificity (True Negative Rate)

```
Specificity  =  TN  /  (TN + FP)
             =  110 / (110 + 10)
             =  91.7%
```

*Of all actually-healthy patients, what fraction were correctly identified as healthy?*

```
Recall (Sensitivity)  →  How well the model catches DIABETIC patients
Specificity           →  How well the model catches HEALTHY patients
Together they give the complete picture across BOTH classes.
```

---

### AUC-ROC

The ROC curve plots **Recall (TPR)** on the Y-axis against **False Positive Rate (1 − Specificity)** on the X-axis, sweeping the decision threshold from 1.0 down to 0.0.

```
Recall (TPR)
1.0 │                  ╭──────────── Perfect model (AUC=1.0)
    │            ╭─────╯
    │       ╭────╯                ← Your model (AUC ≈ 0.92)
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

> **Why AUC is powerful:** It is **threshold-independent** — it evaluates discriminating ability across all possible thresholds at once. The gold standard for comparing classifiers.

---

### Log Loss

```
Log Loss  =  -(1/n) * Sum[ y * log(p)  +  (1-y) * log(1-p) ]
```

Unlike all other metrics that only look at the final label, Log Loss evaluates the **confidence** of predictions. It catastrophically penalises a model that is wrong and confident.

| Algorithm | Probability Output | Log Loss compatible? |
|---|---|---|
| Logistic Regression | Natively calibrated probabilities | ✅ Directly |
| KNN | Vote fraction as proxy | ⚠️ Less well-calibrated |
| SVM | Raw decision score — not a probability | ❌ Needs Platt Scaling first |

---

<a name="72-regression-metrics--side-by-side"></a>

## 7.2 Regression Metrics — Side-by-Side

Using 5 test patients with Blood Sugar predictions:

| Patient | Actual | Predicted | Error |
|---|---|---|---|
| P1 | 145 | 142 | −3 |
| P2 | 160 | 175 | +15 |
| P3 | 120 | 118 | −2 |
| P4 | 200 | 165 | −35 |
| P5 | 95 | 97 | +2 |

**Computed metrics:**

```
MAE   =  (3 + 15 + 2 + 35 + 2) / 5                =  11.4 mg/dL
MSE   =  (9 + 225 + 4 + 1225 + 4) / 5             =  293.4  (mg/dL)^2
RMSE  =  sqrt(293.4)                               =  17.1 mg/dL
R^2   =  1  -  (1467 / 6370)                       =  0.77
```

**MAE vs RMSE — the gap reveals outliers:**

```
MAE  = 11.4
RMSE = 17.1
Gap  = 5.7  ← Because of P4 (error = -35)
             RMSE punishes that 35 as 35^2 = 1225
             MAE treats it the same as a 3 mg/dL error
```

> **RMSE >= MAE always.** A large gap means some really bad predictions are hiding in the average.

**Full comparison:**

| Metric | Formula | Unit | Outlier sensitive? | Best for |
|---|---|---|---|---|
| MAE | `Mean( |y_hat - y| )` | Same as target | No | General purpose, business-friendly |
| MSE | `Mean( (y_hat-y)^2 )` | Squared units | Yes | Mathematical optimisation (GD) |
| RMSE | `sqrt(MSE)` | Same as target | Yes | When large errors are costly |
| R² | `1 - SS_res/SS_tot` | Unitless 0–1 | Moderate | Understanding explanatory power |

---

<a name="73-choosing-the-right-metric--decision-guide"></a>

## 7.3 Choosing the Right Metric — Decision Guide

```
REGRESSION task?
└── Report MAE (interpretable) + RMSE (catches big errors) + R² (explanatory power)

CLASSIFICATION task?
│
├── Is dataset balanced?
│   ├── YES → Accuracy is a reasonable starting point
│   └── NO  → NEVER use Accuracy alone → use F1, AUC
│
├── What is the cost of each error?
│   ├── FN is more dangerous (disease, fraud, fire) → optimise RECALL
│   ├── FP is more dangerous (spam filter, legal)   → optimise PRECISION
│   └── Both matter equally                          → optimise F1
│
├── Comparing models holistically?
│   └── Use AUC-ROC (threshold-independent, single number)
│
└── Need calibrated probability outputs?
    └── Use Log Loss
```

---

<a name="74-domain-cheat-sheet"></a>

## 7.4 Domain Cheat Sheet

| Domain | Task | Most Important Metric | Reason |
|---|---|---|---|
| Cancer / Disease Screening | Classification | **Recall** | Missing real case (FN) can be fatal |
| Spam Filter | Classification | **Precision** | Deleting legitimate emails (FP) destroys trust |
| Fraud Detection | Classification | **Recall + F1** | Missing fraud (FN) means direct financial loss |
| House Price Prediction | Regression | **RMSE + R²** | Large price errors matter a lot |
| Delivery Time Prediction | Regression | **MAE** | All errors roughly equal importance |
| Credit Risk Scoring | Classification | **AUC + Log Loss** | Calibrated probabilities needed for risk bands |
| Weather Forecasting | Regression | **RMSE** | Extreme weather errors are costly |
| Resume Screening | Classification | **Precision** | False positives waste recruiter time |
| Fire / Flood Alarm | Classification | **Recall** | Missing a real event is catastrophic |

---

<a name="chapter-8"></a>

# CHAPTER 8 — The Complete Mental Model

## 8.1 Everything Connected in One Diagram

```
SAME 1,000 PATIENT DATASET
│
├───────────────────────────────────────────────────────────────────┐
│                       SPLIT FIRST (80/20)                         │
│           800 Training Pool               200 Test Set 🔒          │
└───────────────────────────────────────────────────────────────────┘
           │                                         │
           │   5-FOLD CROSS-VALIDATION               │
           │   (finds best hyperparameters)          │
           ▼                                         ▼
┌───────────────────────────┐       ┌────────────────────────────────┐
│   REGRESSION TASK         │       │   CLASSIFICATION TASK           │
│   Predict glucose level   │       │   Diabetic or Not?              │
│                           │       │                                 │
│   Linear Regression       │       │   Logistic Regression           │
│   Hyperparameter: lambda  │       │   └─ Cost: Log Loss             │
│   Loss: MSE               │       │   └─ Hyperparameter: C          │
│   Engine: Gradient Desc.  │       │   └─ Philosophy: Probabilistic  │
│                           │       │                                 │
│   Evaluation:             │       │   KNN                           │
│   • MAE (interpretable)   │       │   └─ No cost function           │
│   • RMSE (outliers)       │       │   └─ Hyperparameter: K          │
│   • R² (explanatory)      │       │   └─ Philosophy: Local Memory   │
│                           │       │                                 │
│   No confusion matrix.    │       │   SVM                           │
│   (continuous output)     │       │   └─ Cost: Hinge Loss           │
└───────────────────────────┘       │   └─ Hyperparameters: C, gamma  │
                                    │   └─ Philosophy: Max Margin     │
                                    │                                 │
                                    │   All three → Confusion Matrix  │
                                    │   • Accuracy                    │
                                    │   • Precision                   │
                                    │   • Recall ← critical here      │
                                    │   • F1 Score                    │
                                    │   • AUC-ROC                     │
                                    └────────────────────────────────┘
```

---

## 8.2 The One-Paragraph Synthesis

You have patient data. You split it honestly — 800 for learning, 200 locked away. For the continuous question (glucose level), Linear Regression with regularisation finds a weighted formula, cross-validation finds the right regularisation strength by honestly averaging performance across 5 different validation splits, and RMSE/R² tell you how precise the predictions are and how much of the story your features are actually telling. For the binary question (diabetic or not), three algorithms each draw a decision boundary in a fundamentally different way — Logistic Regression probabilistically through maximum likelihood, KNN geometrically by local neighbourhood consensus, SVM by finding the hyperplane with the maximum breathing room — cross-validation finds the right complexity dial for each, and the confusion matrix acts as the universal scoreboard that doesn't care how each model decided, only what it decided. Recall is your north star in medical settings because a missed diabetic is catastrophically more costly than a false alarm. The "magic" of tuning and cross-validation is simply this: you stopped guessing at the complexity level and systematically found the level at which the model captures the true pattern of diabetes without memorising the 800 patients it trained on.

---

<a name="chapter-9"></a>

# CHAPTER 9 — Quick Revision: All Formulae in One Place

## Regression Formulae

```
── MODEL ─────────────────────────────────────────────────────────

Linear Regression:
  y_hat  =  w1*x1 + w2*x2 + ... + wn*xn + b

── LOSS / COST ───────────────────────────────────────────────────

MAE   =  (1/n) * Sum( |y_hat_i - y_i| )
MSE   =  (1/n) * Sum( (y_hat_i - y_i)^2 )
RMSE  =  sqrt( MSE )
R^2   =  1  -  SS_res / SS_tot

Cost (Ridge):
  J  =  MSE  +  lambda * Sum( w_i^2 )

── GRADIENT DESCENT ──────────────────────────────────────────────

Update:    w  :=  w  -  alpha * (dJ/dw)
Gradient:  dJ/dw  =  (2/n) * Sum[ (y_hat_i - y_i) * x_i ]

── FEATURE SCALING ───────────────────────────────────────────────

Standardisation:    x_scaled  =  (x - mean) / std
Min-Max:            x_scaled  =  (x - x_min) / (x_max - x_min)
```

## Classification Formulae

```
── LOGISTIC REGRESSION ───────────────────────────────────────────

Sigmoid:   P  =  1 / (1 + e^(-z))     where z = w^T*x + b
Decision:  y_hat = 1 if P >= 0.5,  else 0

Log Loss (one sample):
  Loss  =  -[ y * log(p)  +  (1-y) * log(1-p) ]

Log Loss (cost over n samples):
  J  =  -(1/n) * Sum[ y_i * log(p_i)  +  (1-y_i) * log(1-p_i) ]

Regularised cost:
  J  =  Log_Loss  +  (1/C) * ||w||^2

── KNN ───────────────────────────────────────────────────────────

Prediction:   y_hat  =  mode { y_i : x_i in N_K(x) }

Euclidean:    d(x, x')  =  sqrt( Sum( (x_i - x'_i)^2 ) )
Manhattan:    d(x, x')  =  Sum( |x_i - x'_i| )
Minkowski:    d(x, x')  =  ( Sum( |x_i - x'_i|^p ) )^(1/p)

── SVM ───────────────────────────────────────────────────────────

Margin width:    2 / ||w||
Hard margin:     Minimise (1/2)||w||^2
                 Subject to: y_i*(w^T*x_i + b) >= 1  for all i
Soft margin:     Minimise (1/2)||w||^2 + C*Sum(slack_i)
RBF kernel:      K(x_i, x_j)  =  exp( -gamma * ||x_i - x_j||^2 )
```

## Evaluation Metrics Formulae

```
── CLASSIFICATION ────────────────────────────────────────────────

Accuracy   =  (TP + TN)  /  (TP + TN + FP + FN)
Precision  =  TP  /  (TP + FP)
Recall     =  TP  /  (TP + FN)
F1         =  2 * Precision * Recall  /  (Precision + Recall)
Specificity =  TN  /  (TN + FP)
FPR        =  FP  /  (FP + TN)   ← x-axis of ROC curve

Log Loss   =  -(1/n) * Sum[ y*log(p) + (1-y)*log(1-p) ]

── REGRESSION ────────────────────────────────────────────────────

MAE        =  (1/n) * Sum( |y_hat - y| )
MSE        =  (1/n) * Sum( (y_hat - y)^2 )
RMSE       =  sqrt( MSE )
R^2        =  1  -  SS_res/SS_tot
           =  1  -  Sum(y_hat - y)^2  /  Sum(y_mean - y)^2
```

## Bias-Variance Equation

```
Total Error  =  Bias^2  +  Variance  +  Irreducible Noise
```

---

<a name="chapter-10"></a>

# CHAPTER 10 — Viva / Interview Q&A Bank

> 📖 Read each answer once before a viva. These cover every conceptual question examiners ask about these four algorithms.

---

### Q1. What is the difference between Loss and Cost Function?

**Loss** = error on a single training sample. **Cost** = average loss over the entire training dataset. Gradient Descent minimises the Cost, which implicitly minimises average Loss. The distinction matters because Gradient Descent needs a single scalar to minimise — it cannot optimise individual losses simultaneously.

---

### Q2. Why can't we use MSE as the cost function for Logistic Regression?

With MSE + sigmoid, the cost surface becomes **non-convex** — full of local minima. Gradient Descent may get permanently stuck at a suboptimal solution. Log Loss produces a **convex surface**, guaranteeing that gradient descent converges to the one global minimum, regardless of the starting point.

---

### Q3. Why is Feature Scaling important for KNN but not for Decision Trees?

KNN uses **distance metrics** — features with large scales dominate the distance calculation completely. A difference of 500 in Insulin outweighs a difference of 10 in Age, even if Age is more clinically relevant. Decision Trees use **threshold-based splits** on individual features — only the relative ordering of values within that feature matters, not their absolute magnitude. Scale does not affect relative ordering.

---

### Q4. What are Support Vectors?

The training points that lie exactly on the margin boundaries of an SVM. They are the **only** points that determine the decision boundary. Remove any non-support-vector patient from the training set and the boundary stays completely identical. This makes SVM **sparse** — the final model depends on a small subset of training data, making it robust to outliers far from the margin.

---

### Q5. What does the Kernel Trick do?

It implicitly maps data into a higher-dimensional feature space where it becomes linearly separable — without actually computing that high-dimensional space. Only dot products `K(xi, xj)` between pairs of original-space points are ever computed. For the RBF kernel, this effectively maps data into infinite-dimensional space, enabling highly non-linear boundaries, while remaining computationally efficient.

---

### Q6. Hard Margin vs. Soft Margin SVM — what is the difference?

**Hard Margin** requires all training points to be correctly classified and outside the margin — zero tolerance for violations. It fails completely on noisy or non-linearly separable data because no valid hyperplane exists. **Soft Margin** introduces slack variables and a C parameter that penalises violations, allowing controlled misclassification. The C parameter tunes how tolerant the model is. Real-world data always uses Soft Margin.

---

### Q7. Why Cross-Validation instead of a single train/validation split?

A single split is noisy — performance depends heavily on which patients happened to fall in the validation set by chance. If those 160 patients are unusually easy or hard to classify, the estimate is biased. K-Fold averages K different validation sets, where every patient serves as validation exactly once. This gives a much more stable, reliable estimate of generalisation performance.

---

### Q8. In medical screening, should you optimise Precision or Recall? Why?

**Recall.** Missing a real diabetic (False Negative) means the patient goes home untreated — potentially fatal. A false alarm (False Positive) leads to additional confirmatory tests — inconvenient but not dangerous. The asymmetric cost of errors dictates the metric. Minimising FN = maximising Recall. The choice of which metric to optimise is fundamentally a clinical/business decision, not a mathematical one.

---

### Q9. What is the Bias-Variance Tradeoff?

Total error = Bias² + Variance + irreducible noise. Increasing model complexity reduces Bias (model can represent more complex patterns) but increases Variance (model becomes more sensitive to specific training data). Decreasing complexity does the opposite. The sweet spot — minimum total error — is the bottom of a U-shaped error curve, found via hyperparameter tuning guided by cross-validation.

---

### Q10. How does Gradient Descent connect Linear Regression and Logistic Regression?

Gradient Descent is the common optimisation engine for both. In Linear Regression it minimises MSE cost; in Logistic Regression it minimises Log Loss cost. The update rule `w := w - alpha * dJ/dw` is **identical** — only the gradient formula `dJ/dw` changes because the loss functions are different. Same engine, different fuel.

---

### Q11. LR and SVM both find a linear boundary. What is the fundamental difference?

LR finds the boundary that maximises the likelihood of the training data — **all patients contribute** to shaping it. SVM finds the boundary that maximises the **geometric margin** — only the support vectors on the margin edges matter; all other patients are completely irrelevant. LR outputs calibrated probabilities; SVM outputs a raw decision score. When data is perfectly separable, LR finds infinitely many valid boundaries; SVM finds the **unique one** with the maximum margin.

---

### Q12. KNN has no training phase — so what does cross-validation tune in KNN?

**K itself.** K is not learned from data — it is a hyperparameter you set before running KNN. There are no weights to update and no cost function to minimise during "training" (KNN just stores the data). Cross-validation evaluates different values of K (K=1, 3, 5, 7, ...) and picks the value that gives the best average validation accuracy across the folds. The only "learning" is choosing the right neighbourhood size.

---

### Q13. Why is AUC-ROC a better comparison metric than Accuracy?

Accuracy depends on the chosen decision threshold (default 0.5) and is misleading on imbalanced datasets. AUC-ROC is **threshold-independent** — it evaluates the model's inherent ability to discriminate between classes across all possible thresholds simultaneously. An AUC of 0.92 means: given one random diabetic and one random healthy patient, the model assigns a higher probability to the diabetic 92% of the time — regardless of where you set the threshold.

---

### Q14. What is the difference between Precision and Recall? When does each matter?

**Precision** = of all patients the model *flagged as positive*, what fraction actually were positive. Focus: quality of positive predictions. **Recall** = of all patients who *actually are positive*, what fraction did the model catch. Focus: coverage of true positives. Precision matters when False Positives are costly (spam filter — you don't want to delete real emails). Recall matters when False Negatives are costly (disease screening — you don't want to miss sick patients).

---

### Q15. What is the R² metric and when does it fail?

R² = 1 − (SS_res / SS_tot). It measures the fraction of variance in the target that the model explains. R²=1 is perfect; R²=0 means the model is no better than predicting the mean; R²<0 means the model is actively worse than predicting the mean. It fails as a standalone metric when: (1) the dataset has extreme outliers that inflate SS_tot, making a mediocre model look good; (2) comparing models across different datasets or targets; (3) when large individual errors matter — use RMSE alongside R² to surface dangerous mispredictions.

---

> **The one question that unifies every concept in this guide:**
>
> *"How wrong is my model, and how do I make it less wrong — honestly?"*
>
> - **Loss functions** measure "how wrong"
> - **Gradient Descent** reduces "how wrong" during training
> - **Regularisation** prevents overfitting so "less wrong on training" means "less wrong in real life"
> - **Cross-validation** ensures "less wrong" is measured honestly — not on data the model has already seen
> - **Evaluation metrics** are the precise, domain-specific language for describing "how wrong" after deployment