# Classical ML Algorithms — Comparison, Intuition & Interview Notes
> Simple language. Real examples. What to use and why. Built for interview and viva revision.

---

## Table of Contents
1. [One-Line Mental Models — All Algorithms](#1-one-line-mental-models--all-algorithms)
2. [Linear Regression](#2-linear-regression)
3. [Logistic Regression](#3-logistic-regression)
4. [K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
5. [Support Vector Machines (SVM)](#5-support-vector-machines-svm)
6. [Naive Bayes](#6-naive-bayes)
7. [Decision Trees](#7-decision-trees)
8. [Random Forests](#8-random-forests)
9. [XGBoost / Gradient Boosting](#9-xgboost--gradient-boosting)
10. [Ensemble Methods — The Big Picture](#10-ensemble-methods--the-big-picture)
11. [Head-to-Head Algorithm Comparisons](#11-head-to-head-algorithm-comparisons)
12. [What to Use and Why — The Decision Guide](#12-what-to-use-and-why--the-decision-guide)
13. [Interview Cheat Sheet](#13-interview-cheat-sheet)

---

## 1. One-Line Mental Models — All Algorithms

> Before anything else, memorize these. They anchor everything.

| Algorithm | One-Line Mental Model |
|---|---|
| **Linear Regression** | *"Draw the best-fit straight line through the data."* |
| **Logistic Regression** | *"Draw the best straight boundary, then convert distance to probability."* |
| **KNN** | *"Ask your K nearest neighbors to vote."* |
| **SVM** | *"Draw the boundary with the maximum breathing room from both sides."* |
| **Naive Bayes** | *"Use Bayes' theorem — assume all features are independent, multiply probabilities."* |
| **Decision Tree** | *"Play 20 questions — ask yes/no questions about features until you reach an answer."* |
| **Random Forest** | *"Ask 100 different decision trees and take a majority vote."* |
| **Gradient Boosting / XGBoost** | *"Build trees one by one, each one fixing the mistakes of the previous."* |

---

## 2. Linear Regression

### What It Does
Predicts a **continuous number** by fitting a straight line (or hyperplane) through the data.

### The Simple Example
Predict a house's **price** from its **size**:

```
Price ($)
|              ● (actual)
|          ●  /
|       ●   /  ← best fit line
|     ●   /
|   ●   /
+------------------→ Size (sq ft)

Equation: Price = w × Size + b
```

For every 1 sq ft increase in size, price increases by $w$. That's it.

### The Math (One Line)
$$\hat{y} = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b = \vec{w}\cdot\vec{x} + b$$

Learned by minimizing **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$

### Key Properties
- **Output:** Continuous number (price, temperature, glucose level)
- **Boundary:** Linear — a straight line or flat hyperplane
- **Training:** Gradient descent or closed-form solution
- **Assumptions:** Linear relationship between features and output; normally distributed errors

### Evaluation Metrics
| Metric | Formula | Plain Meaning |
|---|---|---|
| MAE | $\frac{1}{m}\sum\|\hat{y}-y\|$ | Average absolute error |
| RMSE | $\sqrt{\frac{1}{m}\sum(\hat{y}-y)^2}$ | Penalizes large errors more |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | % of variance explained (1.0 = perfect) |

### When It Fails
- When the relationship is not linear (use polynomial features or tree methods)
- When outliers are extreme (RMSE gets dominated by them)
- When features are highly correlated (multicollinearity distorts weights)

### Regularization — Preventing Overfitting
| Type | What It Does | When to Use |
|---|---|---|
| **Ridge (L2)** | Shrinks all weights toward zero | When all features matter a little |
| **Lasso (L1)** | Drives some weights to exactly zero | Feature selection — many irrelevant features |
| **ElasticNet** | Combination of Ridge and Lasso | High-dimensional sparse data |

---

## 3. Logistic Regression

### What It Does
Predicts a **class label** (0 or 1) by drawing a linear decision boundary and converting distance to a **probability** via the sigmoid function.

### The Simple Example
Predict if a student **passes or fails** based on hours studied:

```
Pass/Fail
|
1|  × × ×  × × ×         × = Pass
  |        |              ○ = Fail
0|○ ○ ○   |
|         ↑
|    Decision Boundary
+------------------→ Hours Studied

Sigmoid squeezes any number into (0, 1):
P(Pass) = σ(w × hours + b)
```

### The Math
$$\hat{y} = \sigma(\vec{w}\cdot\vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}\cdot\vec{x}+b)}}$$

Decision: predict class 1 if $\hat{y} \geq 0.5$, else class 0.

Learned by minimizing **Binary Cross-Entropy Loss**:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\hat{y}^{(i)} + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

### Key Properties
- **Output:** Probability [0, 1] → class label
- **Boundary:** Always linear (straight line in 2D, hyperplane in nD)
- **Interpretable:** Weights directly tell you feature importance
- **Calibrated:** Outputs genuine probabilities

### Linear Regression vs. Logistic Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Task** | Regression (continuous output) | Classification (class label) |
| **Output** | Any real number | Probability between 0 and 1 |
| **Loss function** | MSE | Binary Cross-Entropy |
| **Activation** | None (identity) | Sigmoid |
| **Decision boundary** | No boundary — predicts a value | Linear hyperplane |
| **Example** | Predict house price | Predict pass/fail |

### When It Fails
- Non-linear decision boundaries (data not linearly separable)
- Very high-dimensional non-linear data (images, audio) — use Neural Networks

---

## 4. K-Nearest Neighbors (KNN)

### What It Does
Makes predictions by finding the **K most similar training examples** and letting them vote (classification) or averaging them (regression).

### The Simple Example
Classify a new patient as Diabetic or Healthy:

```
Inflammation
|
|  ○ ○         × = Diabetic
|    ○   ★     ○ = Healthy
|       × ×    ★ = New patient
|         ×
+------------------→ Blood Sugar

K=3: nearest 3 points = ○, ○, × → vote: Healthy wins (2 vs 1)
```

### Key Properties
- **No training phase** — stores all data (lazy learner)
- **Decision boundary** — adapts to any shape (non-linear)
- **Slow at prediction** — computes distance to every training point: $O(m \times n)$
- **Feature scaling mandatory** — Euclidean distance is scale-sensitive

### The K Tradeoff

```
K = 1 → fits every training point exactly → OVERFITTING
K = m → predicts majority class always  → UNDERFITTING
K = optimal (cross-validation) → generalizes well ✓
```

### Logistic Regression vs. KNN

| | Logistic Regression | KNN |
|---|---|---|
| **Philosophy** | Learn a global mathematical function | Memorize and look up neighbors |
| **Decision boundary** | Always linear | Any shape |
| **Training time** | Slow (gradient descent) | Zero |
| **Prediction time** | Fast — $O(n)$ | Slow — $O(m \times n)$ |
| **High dimensions** | Handles well | Breaks down (curse of dimensionality) |
| **Interpretability** | High (weights) | None |
| **New data** | Requires retraining | Just append — instant |

---

## 5. Support Vector Machines (SVM)

### What It Does
Finds the **linear decision boundary with the maximum margin** — the widest possible gap between the two classes.

### The Simple Example
Classify emails as Spam or Not-Spam:

```
Feature 2
|
|  ○ ○  ‖         ‖  × ×
|    ○  ‖  MARGIN ‖  ×
|       ‖         ‖
+------------------→ Feature 1

SVM finds the center line of the widest possible street.
Only the points on the street edges (support vectors) matter.
```

### Key Concepts in Plain English

**Margin** = the total width of the empty zone between the two classes. Wider = safer = better generalization.

**Support Vectors** = the training points sitting exactly on the margin edges. Delete all other points — the boundary doesn't move. These are the only points that matter.

**Kernel Trick** = mathematically lifts data into higher dimensions where it becomes linearly separable — without explicitly computing the transformation.

```
Non-separable in 2D:           Separable in 3D (after kernel):

  × ○ × ○ ×                       ×  ×
     (mixed up)                  ──────── ← flat hyperplane
                                   ○  ○
```

### SVM Hyperparameters

| Parameter | Effect |
|---|---|
| **C (large)** | Hard margin — no tolerance for mistakes → narrow margin → overfitting risk |
| **C (small)** | Soft margin — allows some mistakes → wide margin → underfitting risk |
| **γ (large)** | Each point has tiny local influence → wiggly boundary → overfitting |
| **γ (small)** | Each point has wide influence → smooth boundary → underfitting |

### LR vs. KNN vs. SVM — The Triangle

| | LR | KNN | SVM |
|---|---|---|---|
| **Decision boundary** | Linear | Any shape | Linear (+ kernel = any) |
| **Key idea** | Probability | Proximity | Maximum margin |
| **Training** | Gradient descent | Nothing | Quadratic programming |
| **Scales to large data** | ✅ | ❌ | ❌ (>100K points) |
| **Non-linear data** | Feature engineering needed | Naturally handles it | Kernel trick |
| **Output** | Probability | Class label | Decision score |
| **Outlier robustness** | Sensitive (all points vote) | Sensitive (outlier can be a neighbor) | Robust (only SVs matter) |

---

## 6. Naive Bayes

### What It Does
Uses **Bayes' theorem** to compute the probability of each class given the features, then predicts the most probable class. Assumes all features are **independent** given the class.

### The Simple Example
Diagnose Flu or Cold from symptoms:

```
Training data tells us:
  P(Fever=Yes | Flu)  = 0.80    P(Fever=Yes | Cold)  = 0.20
  P(Cough=Yes | Flu)  = 0.80    P(Cough=Yes | Cold)  = 0.20
  P(Fatigue=Yes | Flu)= 0.80    P(Fatigue=Yes | Cold) = 0.20

New patient: Fever, Cough, Fatigue all = Yes

Score(Flu)  = 0.5 × 0.8 × 0.8 × 0.8 = 0.256
Score(Cold) = 0.5 × 0.2 × 0.2 × 0.2 = 0.004

→ Predict: Flu (98.5% confident)
```

### The Three Flavors

| Flavor | Feature Type | Use Case |
|---|---|---|
| **Bernoulli NB** | Binary (yes/no) | Symptom detection, document classification |
| **Multinomial NB** | Counts (word frequencies) | Spam detection, text classification |
| **Gaussian NB** | Continuous numbers | Medical measurements, sensor data |

### Why It Works Despite the "Naive" Assumption
- Only needs to **rank** classes correctly, not produce perfect probabilities
- Correlated features inflate **both** class scores equally — the decision is preserved
- Low variance compensates for high bias — works well with small data
- Strong features (e.g., the word "lottery" in spam) dominate and overwhelm the correlation error

### Naive Bayes vs. Logistic Regression

| | Naive Bayes | Logistic Regression |
|---|---|---|
| **Type** | Generative (models $P(x\|c)$) | Discriminative (models $P(c\|x)$) |
| **Assumption** | Feature independence | No distributional assumption |
| **Small data** | Wins — converges fast | Needs more data |
| **Large data** | LR eventually surpasses NB | Reaches better asymptotic accuracy |
| **Speed** | Extremely fast | Fast |
| **Text/NLP** | Excellent | Good |

---

## 7. Decision Trees

### What It Does
Learns a sequence of **yes/no questions** about features to split the data, building a tree structure that eventually reaches a class prediction at each leaf.

### The Simple Example
Predict if someone will buy a product:

```
                   [Age > 30?]
                  /           \
               Yes              No
                |                |
         [Income > 50k?]    → "Won't Buy"
          /         \
        Yes          No
         |            |
     "Will Buy"   "Won't Buy"
```

Each internal node = a question about one feature.
Each leaf = a prediction.
The tree learned these questions automatically from data.

### How the Tree Decides Which Feature to Split On

At every node, the algorithm tries every possible split and picks the one that produces the **purest** child nodes — measured by:

**Gini Impurity** — probability of misclassifying a randomly chosen element:

$$\text{Gini} = 1 - \sum_{k=1}^{K} p_k^2$$

**Information Gain (Entropy)** — reduction in uncertainty:

$$\text{Entropy} = -\sum_{k=1}^{K} p_k \log_2 p_k$$

A node where all examples belong to one class has Gini = 0 and Entropy = 0 — perfectly pure.

### Growing vs. Pruning

```
OVERFITTING:                  AFTER PRUNING:
Tree grows until              Tree stops early —
every leaf has 1 point        leaves have multiple points

Training accuracy: 100%       Training accuracy: 85%
Test accuracy: 61%            Test accuracy: 83% ✓
```

| Hyperparameter | What It Controls |
|---|---|
| `max_depth` | Maximum levels deep the tree can grow |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required at a leaf |

### Key Properties
- **Interpretable** — you can read the tree and understand every decision
- **No feature scaling needed** — splits are based on order, not magnitude
- **Handles non-linear boundaries** — each split can carve out any region
- **Unstable** — small changes in training data can produce completely different trees (high variance)
- **Tends to overfit** — especially with no depth limit

### Decision Tree vs. All Previous

| | LR | KNN | SVM | NB | Decision Tree |
|---|---|---|---|---|---|
| **Boundary** | Linear | Any | Linear+Kernel | Linear (probabilistic) | Piecewise rectangular |
| **Interpretable** | ✅ | ❌ | ❌ | Partial | ✅✅ (most interpretable) |
| **Feature scaling** | Needed | **Mandatory** | Needed | Not needed | **Not needed** |
| **Handles non-linearity** | ❌ | ✅ | ✅ (kernel) | ❌ | ✅ |
| **Overfitting risk** | Low | High (K=1) | Low | Low | **Very High** |

---

## 8. Random Forests

### What It Does
Builds **many decision trees** on random subsets of data and features, then combines their predictions by majority vote (classification) or averaging (regression).

### The Simple Example — Why One Tree Isn't Enough

```
One Decision Tree:
  Training data has a noisy patient →
  Tree grows a weird branch to fit the noise →
  Test accuracy: 74%

100 Random Trees (Random Forest):
  Each tree sees a different random sample →
  Each tree makes a different error →
  Errors cancel out when you take a majority vote →
  Test accuracy: 89% ✓
```

### The Two Randomness Sources

**1. Bootstrap Sampling (Bagging):**
Each tree is trained on a **random sample with replacement** from the training data (~63% of unique examples). Different trees see different data → different errors.

**2. Feature Randomness:**
At each split, only a **random subset of features** is considered (typically $\sqrt{n}$ features). This prevents all trees from always splitting on the same dominant feature → more diversity.

```
Full Dataset (1000 patients, 10 features)

Tree 1: trained on 630 random patients, splits on random 3 features each time
Tree 2: trained on 630 different random patients, different 3 features
Tree 3: trained on 630 different random patients, different 3 features
...
Tree 100: same idea

Final prediction: majority vote of all 100 trees
```

### Why Randomness Helps — Bias-Variance Perspective

| | Single Deep Tree | Random Forest |
|---|---|---|
| **Bias** | Low (fits data well) | Low (many complex trees) |
| **Variance** | **High** (overfits noise) | **Low** (errors average out) |
| **Generalization** | Poor | **Excellent** |

Each tree has high variance. But **uncorrelated** high-variance models average to low variance. The key word is **uncorrelated** — if all trees were identical, averaging wouldn't help. Randomness makes them different from each other, so their errors cancel.

### Out-of-Bag (OOB) Error — Free Cross-Validation

Since each tree only sees ~63% of data, the remaining ~37% (out-of-bag samples) can be used to evaluate that tree. This gives a **free, honest estimate of test error** without a separate validation set.

### Feature Importance

Random Forest naturally ranks features by how much each one **improves the splits** across all trees. This is one of the most reliable feature importance measures in classical ML.

```python
rf.feature_importances_
# → [0.35, 0.28, 0.15, 0.12, 0.10]
# Feature 1 does 35% of the "work" across all trees
```

### Decision Tree vs. Random Forest

| | Decision Tree | Random Forest |
|---|---|---|
| **Variance** | **High** — unstable | Low — stable |
| **Overfitting** | **Very prone** | Resistant |
| **Interpretability** | ✅ Can read the tree | ❌ 100 trees — unreadable |
| **Training speed** | Fast | Slower (100× trees) |
| **Feature importance** | Unreliable | Reliable |
| **Missing values** | Struggles | Handles well |
| **Performance** | Baseline | Strong ✅ |

---

## 9. XGBoost / Gradient Boosting

### What It Does
Builds trees **sequentially** — each new tree is trained specifically to **correct the errors** (residuals) made by all previous trees combined. Trees learn from each other's mistakes.

### The Simple Example — Learning From Mistakes

```
Dataset: predict house prices

Tree 1: Predicts $200k for a house actually worth $250k
        → Error = $50k

Tree 2: Trained on the ERRORS of Tree 1
        → Learns that houses like this are typically underpriced
        → Predicts correction: +$35k

Tree 3: Trained on remaining errors
        → Predicts: +$10k

...

Final prediction = Tree1 + Tree2 + Tree3 + ... = $200k + $35k + $10k + ... ≈ $248k
```

Each tree is a **small, weak learner** (shallow tree). But many weak learners **added together** form a powerful model.

### Random Forest vs. Gradient Boosting — The Core Difference

```
RANDOM FOREST (Parallel):          GRADIENT BOOSTING (Sequential):

Tree1  Tree2  Tree3  ...           Tree1 → error1
  ↓      ↓      ↓                     ↓
  Vote together                    Tree2 (fixes error1) → error2
  (all at once)                        ↓
                                    Tree3 (fixes error2) → ...
                                        ↓
                                    Final = Sum of all trees
```

| | Random Forest | Gradient Boosting / XGBoost |
|---|---|---|
| **How trees are built** | Independently, in parallel | Sequentially, each fixes previous errors |
| **Each tree's role** | Equal contributor | Specialist in fixing current errors |
| **Speed** | Fast (parallelizable) | Slower (sequential) |
| **Overfitting risk** | Low | Higher (needs careful tuning) |
| **Performance ceiling** | High | **Higher** |
| **Hyperparameter sensitivity** | Low | High |
| **Winner on tabular data** | Strong baseline | **State of the art** |

### Why XGBoost is Special

XGBoost (eXtreme Gradient Boosting) is an engineered, optimized implementation of gradient boosting that adds:

| XGBoost Feature | Plain Meaning | Benefit |
|---|---|---|
| **Regularization (L1/L2)** | Penalizes tree complexity | Controls overfitting |
| **Tree pruning** | Removes unhelpful splits | Cleaner trees |
| **Parallel processing** | Builds tree levels in parallel | Much faster than vanilla GBM |
| **Sparse data handling** | Handles missing values natively | No imputation needed |
| **Column subsampling** | Like Random Forest's feature randomness | More diversity, less overfitting |
| **Learning rate (η)** | Shrinks each tree's contribution | Slower but more precise learning |

### Key Hyperparameters to Know

| Parameter | What It Controls | High Value → | Low Value → |
|---|---|---|---|
| `n_estimators` | Number of trees | More complex | Underfit |
| `learning_rate` | How much each tree contributes | Overfit | Underfit |
| `max_depth` | Depth of each tree | Overfit | Underfit |
| `subsample` | Fraction of rows per tree | — | More regularization |
| `colsample_bytree` | Fraction of features per tree | — | More regularization |

**Key rule:** Lower learning rate + more trees = better generalization (but slower training).

---

## 10. Ensemble Methods — The Big Picture

### What Is an Ensemble?

An ensemble combines **multiple models** to produce a better prediction than any single model alone. This is why Random Forests and Gradient Boosting dominate tabular ML competitions.

### The Three Ensemble Strategies

**Strategy 1 — Bagging (Bootstrap Aggregating)**
Train multiple models on random subsets of data independently. Average/vote their predictions.

```
Goal: Reduce VARIANCE
Mechanism: Errors are uncorrelated → they cancel out when averaged
Example: Random Forest
```

**Strategy 2 — Boosting**
Train models sequentially. Each model focuses on the examples the previous models got wrong.

```
Goal: Reduce BIAS
Mechanism: Weak learners combine into a strong learner
Example: Gradient Boosting, XGBoost, AdaBoost
```

**Strategy 3 — Stacking**
Train multiple different models (LR, SVM, RF, XGBoost). Feed their predictions as input to a **meta-model** that learns the best way to combine them.

```
Level 0 (base models):  LR → 0.72    RF → 0.85    SVM → 0.79
                                ↓         ↓          ↓
Level 1 (meta-model):  [0.72, 0.85, 0.79] → Final prediction

Meta-model learns: "trust RF 60%, SVM 30%, LR 10%"
```

### Bagging vs. Boosting vs. Stacking

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| **Models trained** | In parallel | Sequentially | In parallel, then a meta-model |
| **What it reduces** | Variance | Bias | Both |
| **Overfitting risk** | Low | Medium-High | High (if meta-model overfits) |
| **Example** | Random Forest | XGBoost, AdaBoost | BlendedModel |
| **Complexity** | Simple | Moderate | Complex |
| **When to use** | High-variance base model | High-bias base model | Competition / maximum performance |

---

## 11. Head-to-Head Algorithm Comparisons

### The Master Comparison Table

| Property | LR | Logistic R | KNN | SVM | NB | DT | RF | XGBoost |
|---|---|---|---|---|---|---|---|---|
| **Task** | Regression | Classification | Both | Both | Classification | Both | Both | Both |
| **Output** | Number | Probability | Label | Score | Probability | Label | Label | Label/Number |
| **Boundary** | Linear | Linear | Any | Linear+Kernel | Linear | Rectangular | Complex | Complex |
| **Training speed** | Fast | Fast | Zero | Slow | Very fast | Fast | Moderate | Moderate |
| **Prediction speed** | Fast | Fast | Slow | Fast | Fast | Fast | Fast | Fast |
| **Feature scaling** | Needed | Needed | **Mandatory** | Needed | Not needed | Not needed | Not needed | Not needed |
| **Interpretable** | ✅ | ✅ | ❌ | ❌ | Partial | ✅✅ | Partial | ❌ |
| **Non-linear data** | ❌ | ❌ | ✅ | ✅ (kernel) | ❌ | ✅ | ✅ | ✅ |
| **High dimensions** | ✅ | ✅ | ❌ | ✅ | ✅ (NLP) | ❌ | ✅ | ✅ |
| **Small data** | ✅ | ✅ | Okay | ✅ | ✅✅ | Okay | ✅ | Needs tuning |
| **Large data (>1M)** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Missing values** | Struggles | Struggles | Struggles | Struggles | Struggles | Handles | Handles | **Natively** |
| **Overfitting risk** | Low | Low | High (K=1) | Low | Low | **Very High** | Low | Medium |
| **Performance** | Baseline | Baseline | Moderate | Good | Fast/Good | Weak | Strong | **Best** |

---

### The Most Common Interview Comparisons

**Q: Linear Regression vs. Logistic Regression?**

Same linear model. Linear Regression predicts a continuous number — output is $\vec{w}\cdot\vec{x}+b$. Logistic Regression predicts a class — it wraps that same expression in a sigmoid to produce a probability, then applies a threshold. Same input, different output transformation, different loss function (MSE vs. cross-entropy).

**Q: Logistic Regression vs. SVM?**

Both draw a linear boundary. LR finds the boundary that **maximizes the likelihood** of the training labels (every point votes). SVM finds the boundary that **maximizes the margin** (only support vectors matter — outliers far from the boundary are completely ignored). LR gives calibrated probabilities; SVM gives a raw decision score.

**Q: Decision Tree vs. Random Forest?**

A Decision Tree is a single tree that tends to overfit — it has high variance and changes drastically if you change the training data. Random Forest trains 100+ such trees on random subsets, then averages. The individual errors cancel out, giving low variance and much better generalization. You trade interpretability for accuracy.

**Q: Random Forest vs. XGBoost?**

Random Forest: trees built in **parallel**, independently, errors averaged. Reduces variance.
XGBoost: trees built **sequentially**, each one fixing the previous tree's mistakes. Reduces bias iteratively. XGBoost generally outperforms Random Forest on tabular data, but requires more hyperparameter tuning and is more prone to overfitting if not regularized.

**Q: Bagging vs. Boosting?**

Bagging trains models independently and averages — targets **variance reduction**. Boosting trains models sequentially where each one focuses on previous mistakes — targets **bias reduction**. Bagging is more robust; Boosting achieves higher performance ceiling but can overfit.

**Q: Why does Naive Bayes work despite the independence assumption being wrong?**

Because it only needs to **rank classes correctly**, not produce accurate probabilities. Correlated features inflate both class scores equally, so the ratio (and thus the decision) is preserved. Plus: low variance from fewer parameters compensates for the bias from the wrong assumption — especially with small data.

---

## 12. What to Use and Why — The Decision Guide

### Step-by-Step Framework

```
START HERE
    ↓
What is your task?
├── Predict a number (price, temperature, score)
│       ↓
│   → Linear Regression (start here)
│     + try Ridge/Lasso if overfitting
│     + try Random Forest / XGBoost for non-linear patterns
│
└── Predict a class (yes/no, category)
        ↓
    How much data do you have?
    ├── < 1,000 samples
    │       ↓
    │   → Naive Bayes (text/symptoms)
    │     or SVM with RBF kernel (tabular)
    │     or Logistic Regression (baseline)
    │
    ├── 1,000 – 100,000 samples
    │       ↓
    │   → Random Forest (robust default)
    │     or XGBoost (if you want best performance)
    │     or Logistic Regression (if interpretability needed)
    │
    └── > 100,000 samples
            ↓
        What type of data?
        ├── Tabular → XGBoost / LightGBM
        └── Images / Text / Audio → Neural Networks
```

### When Each Algorithm Wins

| Algorithm | Best Scenario | Avoid When |
|---|---|---|
| **Linear Regression** | Continuous output, linear relationship, need interpretability | Non-linear data without feature engineering |
| **Logistic Regression** | Strong baseline, interpretability required, legal/medical accountability | Non-linear boundaries without feature engineering |
| **KNN** | Small dataset, low dimensions, quick prototype, recommendation systems | Large dataset, high dimensions, need fast predictions |
| **SVM** | Small-medium data, high-dimensional features, genomics/bioinformatics | >100K rows (too slow), need probability outputs |
| **Naive Bayes** | Text classification, spam, very small data, need speed | When feature independence is completely violated and accuracy matters |
| **Decision Tree** | Need full interpretability, explain every decision, rule extraction | Production models (overfits — use RF instead) |
| **Random Forest** | General-purpose strong baseline, feature importance needed | Need interpretability, memory-constrained |
| **XGBoost** | Tabular data competitions, maximum performance, structured data | Raw images/audio/text, need simple/fast model |

### The Practical Hierarchy (What Practitioners Actually Do)

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
        → This is the industry standard for tabular data

Step 4: If nothing works → Neural Networks
        → Only justified for large data or unstructured input
        → Much harder to train and debug
```

### The Golden Rule

> **Start simple. Upgrade only when the simpler model is provably insufficient.**

A Logistic Regression that is well-understood, debugged, and trusted is worth more than an XGBoost that is a black box. Complexity has a cost — in debugging, in maintenance, and in trust.

---

## 13. Interview Cheat Sheet

### Core Definitions — Say These Confidently

**Overfitting:** Model memorizes training data including noise. High training accuracy, low test accuracy. Fix: regularization, more data, simpler model.

**Underfitting:** Model too simple to capture the pattern. Both training and test accuracy are low. Fix: more complex model, more features.

**Bias-Variance Tradeoff:** Every model has $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$. Increasing complexity decreases bias but increases variance. The goal is the lowest total error.

**Cross-Validation:** Rotate test folds across the training data to get a stable estimate of model performance without touching the test set.

**Feature Scaling:** Required for LR, KNN, SVM — algorithms that use distance or gradient descent. Not required for tree-based methods (DT, RF, XGBoost) — they split on order, not magnitude.

---

### The "Why" Quick-Fire Answers

| Question | 10-Second Answer |
|---|---|
| Why sigmoid in Logistic Regression? | Squeezes any number into (0,1) so we get a valid probability |
| Why maximize margin in SVM? | The widest margin gives the most generalization — farthest from both classes |
| Why odd K in KNN? | Avoids tie votes in binary classification |
| Why log-probabilities in Naive Bayes? | Prevents numerical underflow when multiplying many small numbers |
| Why Laplace smoothing? | Prevents zero probabilities from annihilating the entire class score |
| Why Random Forest beats Decision Tree? | Averaging uncorrelated trees cancels out variance |
| Why Boosting beats Bagging? | Sequential error correction reduces bias, not just variance |
| Why XGBoost is fast? | Parallelizes within each tree level + efficient data structures |
| Why feature scaling doesn't matter for trees? | Splits are based on rank/order of values, not their magnitude |
| Why KNN fails in high dimensions? | All points become equidistant — the concept of "nearest" loses meaning |

---

### Algorithm Selection by Scenario — Viva-Style

| Scenario | Best Choice | Why |
|---|---|---|
| Spam email classification | Multinomial Naive Bayes | Fast, works with word counts, excellent on text |
| Medical diagnosis (small data, accountability) | Logistic Regression | Interpretable weights, probability output, legal explainability |
| House price prediction | Linear Regression or XGBoost | LR for linear relationship, XGBoost for non-linear patterns |
| Image classification | Neural Network (CNN) | Classical ML can't process raw pixels meaningfully |
| Fraud detection (imbalanced classes) | XGBoost with class weights | Handles imbalance, captures complex patterns |
| Customer churn prediction | Random Forest or XGBoost | Tabular data, non-linear patterns, feature importance useful |
| Genomics / bioinformatics | SVM with RBF kernel | High-dimensional, small sample size — SVM's sweet spot |
| Real-time recommendation (billion scale) | KNN via FAISS (approximate) | ANN search powers modern embedding-based retrieval |
| Credit scoring (regulatorily required) | Logistic Regression | Must be interpretable and auditable by law |
| Kaggle competition (tabular data) | XGBoost / LightGBM | Empirically wins the most competitions |

---

### The One Paragraph That Ties Everything Together

> You start with the simplest model — **Linear or Logistic Regression** — because interpretability and speed matter and a simple model that works is always better than a complex one that barely outperforms it. When the data is non-linear, you move to **KNN** if the dataset is small and low-dimensional, **SVM** if it's medium-sized and high-dimensional, or **Naive Bayes** if it's text or you have very little data. When you need raw power on tabular data, **Decision Trees** give you interpretability at the cost of overfitting — so you almost always use **Random Forest** instead, which buys robustness through diversity. When you want the best possible performance, **XGBoost** wins by learning sequentially from its own mistakes. Everything from Random Forest onward is an **Ensemble Method** — combining weak learners to produce a strong one, either in parallel (Bagging) or in sequence (Boosting). No algorithm is universally best. The right choice always depends on your data size, data type, interpretability requirements, and how much you're willing to trade performance for simplicity.

---