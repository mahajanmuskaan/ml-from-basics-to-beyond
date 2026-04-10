# ML Algorithms — Revision Guide II
### Naive Bayes · Decision Tree · Random Forest · XGBoost
#### Probabilistic Models → Tree Logic → Ensemble Power → Gradient Boosting

---

> 🏥 **One real-world anchor used throughout this entire guide:**
> A hospital has data on **10,000 patients** — Age, BMI, Blood Pressure, Glucose, Insulin, Smoking_Status, Family_History, Diet_Quality.
>
> | Question | Task | Algorithm(s) |
> |---|---|---|
> | *"Will this patient develop Diabetes?"* | Classification | Naive Bayes / Decision Tree / Random Forest / XGBoost |
> | *"What is this patient's Blood Sugar level?"* | Regression | Decision Tree / Random Forest / XGBoost |
>
> Same patients. Same features. These four algorithms approach the problem in completely different ways — from probabilistic reasoning to tree logic to ensemble wisdom to sequential error correction.

---

## Table of Contents

- [Chapter 1 — The Golden Rule & Data Split](#chapter-1)
- [Chapter 2 — Naive Bayes](#chapter-2)
  - [2.1 Core Philosophy](#21-core-philosophy)
  - [2.2 Bayes' Theorem — The Foundation](#22-bayes-theorem--the-foundation)
  - [2.3 The Naive Assumption](#23-the-naive-assumption)
  - [2.4 Three Variants of Naive Bayes](#24-three-variants-of-naive-bayes)
  - [2.5 Laplace Smoothing](#25-laplace-smoothing--preventing-zero-probabilities)
  - [2.6 Hyperparameter: var_smoothing](#26-hyperparameter-var_smoothing)
  - [2.7 Properties, Strengths, Weaknesses](#27-naive-bayes--properties-strengths-weaknesses)
  - [2.8 When to Use Naive Bayes](#28-when-to-use-naive-bayes)
- [Chapter 3 — Decision Tree](#chapter-3)
  - [3.1 Core Philosophy](#31-core-philosophy)
  - [3.2 Splitting Criteria](#32-how-the-tree-learns--splitting-criteria)
  - [3.3 Tree Building — Full Algorithm](#33-tree-building--the-full-algorithm)
  - [3.4 Hyperparameters — The Pruning Dials](#34-hyperparameters--the-pruning-dials)
  - [3.5 Visualising a Decision Tree](#35-visualising-and-interpreting-a-decision-tree)
  - [3.6 Properties, Strengths, Weaknesses](#36-decision-tree--properties-strengths-weaknesses)
- [Chapter 4 — Random Forest](#chapter-4)
  - [4.1 Core Philosophy](#41-core-philosophy)
  - [4.2 Two Randomisation Tricks](#42-two-randomisation-tricks--the-heart-of-random-forest)
  - [4.3 Prediction — How the Forest Votes](#43-prediction--how-the-forest-votes)
  - [4.4 Why Averaging Reduces Variance](#44-why-averaging-reduces-variance--the-math)
  - [4.5 Hyperparameters](#45-hyperparameters)
  - [4.6 Feature Importance](#46-feature-importance-in-random-forest)
  - [4.7 Properties, Strengths, Weaknesses](#47-random-forest--properties-strengths-weaknesses)
- [Chapter 5 — XGBoost](#chapter-5)
  - [5.1 Core Philosophy](#51-core-philosophy)
  - [5.2 Gradient Boosting — Core Mechanism](#52-gradient-boosting--the-core-mechanism)
  - [5.3 XGBoost Objective Function](#53-the-xgboost-objective-function)
  - [5.4 Key Hyperparameters](#54-key-xgboost-hyperparameters)
  - [5.5 XGBoost vs. Random Forest](#55-xgboost-vs-random-forest--the-key-differences)
  - [5.6 LightGBM and CatBoost](#56-lightgbm-and-catboost--brief-mentions)
- [Chapter 6 — Side-by-Side Comparison](#chapter-6)
- [Chapter 7 — Bias-Variance for All Four](#chapter-7)
- [Chapter 8 — Evaluation Metrics Quick Reference](#chapter-8)
- [Chapter 9 — Quick Revision: All Formulae in One Place](#chapter-9)
- [Chapter 10 — Viva / Interview Q&A Bank](#chapter-10)

---

<a name="chapter-1"></a>

# CHAPTER 1 — The Golden Rule & Data Split

Before any algorithm, the same golden rule applies:

```
10,000 Patients
│
├── 8,000 patients  →  Training Pool  (all learning happens here)
│
└── 2,000 patients  →  Test Set 🔒   (locked vault — opened exactly once at the end)
```

**Feature Scaling for these algorithms:**

| Algorithm | Needs Scaling? | Reason |
|---|---|---|
| Naive Bayes | ❌ No | Works with probabilities — scale doesn't affect conditional distributions |
| Decision Tree | ❌ No | Splits based on thresholds — relative ordering unchanged by scaling |
| Random Forest | ❌ No | Ensemble of trees — same reasoning |
| XGBoost | ❌ No | Tree-based — split thresholds are scale-invariant |

> 💡 This is one of the key advantages of tree-based algorithms over linear models and SVMs — **no feature scaling required**. They are also naturally robust to outliers for the same reason.

---

<a name="chapter-2"></a>

# CHAPTER 2 — Naive Bayes

<a name="21-core-philosophy"></a>

## 2.1 Core Philosophy

> **"What is the probability that this patient is Diabetic, given what I know about them — assuming each symptom acts independently?"**

Naive Bayes is a **generative, probabilistic, parametric** model. It does not learn a decision boundary directly. Instead, it learns *how each class generates data*, then uses Bayes' theorem to flip it into a classification.

Compare this with Logistic Regression, which directly models P(y|x). Naive Bayes takes the longer route through the joint distribution.

---

<a name="22-bayes-theorem--the-foundation"></a>

## 2.2 Bayes' Theorem — The Foundation

```
P(Diabetic | features)  =  P(features | Diabetic)  x  P(Diabetic)
                           ──────────────────────────────────────────
                                       P(features)
```

In ML notation:

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
| `P(x)` | **Evidence** | Normalisation constant — same for all classes, can be ignored |

**For classification, compare posteriors:**

```
Predict Diabetic  if  P(Diabetic | x)  >  P(Not Diabetic | x)

Simplifies to (dropping P(x) which cancels):
Predict Diabetic  if  P(x | Diabetic) x P(Diabetic)  >  P(x | Not Diabetic) x P(Not Diabetic)
```

---

<a name="23-the-naive-assumption"></a>

## 2.3 The Naive Assumption

The "Naive" in Naive Bayes is this:

> **Assumption: All features are conditionally independent given the class label.**

```
P(x | y)  =  P(x1 | y)  x  P(x2 | y)  x  P(x3 | y)  x  ...  x  P(xn | y)

P(features | Diabetic)
  =  P(Glucose=180 | Diabetic)
  x  P(BMI=32      | Diabetic)
  x  P(Age=55      | Diabetic)
  x  P(Smoking=Yes | Diabetic)
  x  ...
```

**Is this assumption realistic?** Almost never. BMI and Glucose are correlated. Age and Blood Pressure are correlated.

**Then why does it work?** Because even with incorrect probability estimates, the *ranking* of classes (which posterior is larger) is often correct. The model is wrong about the magnitude of probabilities but right about which class is more likely. This is enough for classification.

---

<a name="24-three-variants-of-naive-bayes"></a>

## 2.4 Three Variants of Naive Bayes

The variants differ only in how they model `P(xi | y)` — the likelihood of each feature given the class.

---

### Variant 1 — Gaussian Naive Bayes (for continuous features)

**Assumption:** Each feature follows a Normal distribution within each class.

**Gaussian likelihood formula:**

```
P(xi | y)  =  (1 / sqrt(2 * pi * sigma^2_iy))  x  exp( -(xi - mu_iy)^2  /  (2 * sigma^2_iy) )

where:
  mu_iy    = mean of feature i among patients in class y
  sigma_iy = std deviation of feature i among patients in class y
```

**Training step — learn parameters from 8,000 training patients:**

```
Class = Diabetic (y=1):        2,400 patients
  mu_Glucose_1  = 158.3 mg/dL,    sigma_Glucose_1 = 28.4
  mu_BMI_1      = 30.7,            sigma_BMI_1     =  4.2
  mu_Age_1      = 54.2,            sigma_Age_1     = 10.1

Class = Not Diabetic (y=0):    5,600 patients
  mu_Glucose_0  = 108.6 mg/dL,    sigma_Glucose_0 = 18.7
  mu_BMI_0      = 25.3,            sigma_BMI_0     =  3.8
  mu_Age_0      = 41.8,            sigma_Age_0     = 12.3

Prior probabilities:
  P(Diabetic)     =  2400/8000  =  0.30
  P(Not Diabetic) =  5600/8000  =  0.70
```

**Prediction for new patient Priya (Glucose=165, BMI=31, Age=52):**

```
Score_Diabetic    =  P(Glucose=165|D) x P(BMI=31|D) x P(Age=52|D) x P(D)
                  =  0.0134  x  0.0948  x  0.0383  x  0.30  =  [very small positive]

Score_NotDiabetic =  P(Glucose=165|ND) x P(BMI=31|ND) x P(Age=52|ND) x P(ND)
                  =  0.0008  x  0.0312  x  0.0272  x  0.70  =  [much smaller]

Since Score_Diabetic >> Score_NotDiabetic  →  Predict: DIABETIC
```

**In practice, use log-probabilities to avoid numerical underflow:**

```
log P(y | x)  ∝  log P(y)  +  Sum_i[ log P(xi | y) ]

(sum of logs instead of product — prevents floating point going to zero)
```

---

### Variant 2 — Multinomial Naive Bayes (for count/frequency features)

Used primarily for **text classification** (word frequency counts).

**Likelihood formula:**

```
P(word_i | class_y)  =  (count(word_i in class_y) + alpha)
                         ─────────────────────────────────────────────
                         (total words in class_y  +  alpha x vocab_size)

alpha = Laplace smoothing parameter (prevents zero probabilities for unseen words)
```

**Example — Spam Detection:**

```
Training emails:
  Spam (y=1):    "win money now click prize free..."
  Not Spam (y=0):"meeting tomorrow project deadline..."

P("win" | Spam)    = 0.045  (appears often in spam)
P("win" | NotSpam) = 0.002  (rarely in normal email)

New email contains "win", "prize", "click":
P(Spam | email) ∝ 0.6 x 0.045 x 0.038 x 0.041
                >> P(NotSpam | email)  →  classified as SPAM
```

---

### Variant 3 — Bernoulli Naive Bayes (for binary features)

Used when features are binary (present = 1, absent = 0).

**Likelihood formula:**

```
P(xi | y)  =  p_iy^xi  x  (1 - p_iy)^(1-xi)

where  p_iy  =  probability of feature i being present in class y
```

**Example — binary features in our dataset:**

```
Feature: Smoking_Status (0 or 1)
  P(Smoking=1 | Diabetic)     = 0.48
  P(Smoking=1 | Not Diabetic) = 0.23

Feature: Family_History (0 or 1)
  P(Family_History=1 | Diabetic)     = 0.61
  P(Family_History=1 | Not Diabetic) = 0.29
```

---

<a name="25-laplace-smoothing--preventing-zero-probabilities"></a>

## 2.5 Laplace Smoothing — Preventing Zero Probabilities

**The problem:** If a feature value never appeared in training data for a particular class:

```
P(Glucose > 200 | Not Diabetic)  =  0 / 5600  =  0

→  Score_NotDiabetic  =  ...  x  0  x  ...  =  0   (any product with zero = zero)
→  Model can NEVER predict Not Diabetic, no matter what other features say
```

**Laplace Smoothing** adds a pseudocount alpha (usually 1) to all counts:

```
P(xi | y)  =  (count(xi, y)  +  alpha)
               ─────────────────────────────
               (count(y)  +  alpha  x  K)

where K = number of possible values for feature xi
```

This ensures no probability is ever exactly zero — preventing the entire product from collapsing.

---

<a name="26-hyperparameter-var_smoothing"></a>

## 2.6 Hyperparameter: var_smoothing (Gaussian NB)

For Gaussian NB, numerical stability is added by smoothing the variance:

```
sigma^2_smoothed  =  sigma^2_actual  +  var_smoothing  x  max(sigma^2 across all features)
```

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian NB — for continuous features
gnb = GaussianNB(var_smoothing=1e-9)   # default — usually fine
gnb.fit(X_train, y_train)

print("Class priors:", gnb.class_prior_)    # [0.70, 0.30]
print("Feature means:\n", gnb.theta_)       # shape (n_classes, n_features)
print("Feature variances:\n", gnb.var_)     # shape (n_classes, n_features)

# Multinomial NB — for text/count features
mnb = MultinomialNB(alpha=1.0)              # alpha = Laplace smoothing
mnb.fit(X_count_train, y_train)
```

---

<a name="27-naive-bayes--properties-strengths-weaknesses"></a>

## 2.7 Naive Bayes — Properties, Strengths, Weaknesses

| Property | Detail |
|---|---|
| **Type** | Generative, probabilistic, parametric |
| **Training** | Compute means, variances, counts — no gradient descent |
| **Training speed** | Extremely fast — O(n x d) |
| **Inference speed** | Extremely fast — multiply probabilities |
| **Missing values** | Handles naturally — skip missing features in the product |
| **Output** | Calibrated posterior probabilities |
| **Scaling needed?** | No |
| **Non-linear boundaries?** | Yes — product of Gaussians creates curved boundaries |

**Strengths:**

- **Incredibly fast** — trains and predicts almost instantly, even on millions of samples
- **Works well with small data** — needs very few samples to estimate means and variances
- **Handles high-dimensional data** — text classification with 100,000 features works fine
- **Real-time/online learning** — can update parameters incrementally as new data arrives
- **Naturally multi-class** — no one-vs-rest trick needed

**Weaknesses:**

- **Independence assumption** — the fatal flaw on most tabular data. BMI and Glucose are correlated, Age and BP are correlated. Correlated features cause the model to double-count evidence.
- **Distributional assumptions** — Gaussian NB assumes normality. If Glucose is skewed, the assumption is wrong.
- **Zero frequency problem** — requires Laplace smoothing to handle unseen feature values.

---

<a name="28-when-to-use-naive-bayes"></a>

## 2.8 When to Use Naive Bayes

```
✅ USE when:
  - Text classification (spam, sentiment, news categorisation)
  - Real-time prediction needed (email filtering, live recommendations)
  - Very small training datasets where complex models overfit
  - Features are genuinely approximately independent (happens in NLP)
  - You need a fast baseline before trying complex models
  - Online/streaming learning (incremental updates with each new observation)

❌ AVOID when:
  - Features are highly correlated (most tabular datasets)
  - You need precise probability estimates (naivety distorts calibration)
  - Complex non-linear feature interactions are the key signal
```

---

<a name="chapter-3"></a>

# CHAPTER 3 — Decision Tree

<a name="31-core-philosophy"></a>

## 3.1 Core Philosophy

> **"Split the patients into groups by asking yes/no questions about their features, until each group is as pure (homogeneous) as possible in terms of their class label."**

A Decision Tree is **non-parametric** (no distributional assumptions), **white-box** (fully interpretable), and learns **axis-aligned decision boundaries** — splits parallel to the feature axes.

```
DECISION TREE FOR DIABETES:

                    Glucose < 140?
                   /              \
                YES               NO
                │                  │
         BMI < 27?              Age > 50?
         /       \               /      \
       YES        NO           YES       NO
        │          │            │         │
  NOT DIABETIC  DIABETIC    DIABETIC  NOT DIABETIC
   (92% conf)  (74% conf)  (81% conf)  (88% conf)
```

Every path from root to leaf is a classification rule expressed in plain language. This is the essence of interpretability.

---

<a name="32-how-the-tree-learns--splitting-criteria"></a>

## 3.2 How the Tree Learns — Splitting Criteria

The tree is built **greedily** — at each node, find the single best (feature, threshold) pair that maximally reduces impurity.

**The split search:**

```
For each feature f in {Glucose, BMI, Age, BP, ...}:
  For each possible threshold t in {all unique values of f}:
    Left  = patients where f < t
    Right = patients where f >= t
    Compute: Information Gain = Impurity(Parent) - weighted_avg Impurity(children)

Choose (f*, t*) with maximum Information Gain.
```

---

### Impurity Measure 1 — Gini Index (sklearn default)

Measures the probability of misclassifying a randomly chosen sample.

```
Gini(node)  =  1  -  Sum_k( p_k^2 )

where p_k = proportion of class k samples at this node.

Example: node with 70% Diabetic, 30% Not Diabetic:
Gini  =  1  -  (0.70^2 + 0.30^2)
      =  1  -  (0.49  + 0.09)
      =  0.42

Gini = 0.0  →  Pure node (all one class)  — perfect
Gini = 0.5  →  Maximum impurity (50/50 split)  — worst possible
```

**Information Gain using Gini — worked example:**

```
Parent node: 200 patients (140 Diabetic, 60 Not Diabetic)
  Gini(parent) = 1 - (0.70^2 + 0.30^2) = 0.42

Candidate split A: Glucose < 140
  Left:  100 patients (20D, 80ND)  → Gini = 1 - (0.20^2 + 0.80^2) = 0.32
  Right: 100 patients (120D, 20ND) → Gini = 1 - (0.80^2 + 0.20^2) = 0.32
  Weighted Gini = (100/200)*0.32 + (100/200)*0.32 = 0.32
  Gain = 0.42 - 0.32 = 0.10  ✅

Candidate split B: Age < 40
  Left:  80 patients  (55D, 25ND)  → Gini = 0.48
  Right: 120 patients (85D, 35ND)  → Gini = 0.49
  Weighted Gini = (80/200)*0.48 + (120/200)*0.49 = 0.486
  Gain = 0.42 - 0.486 = -0.066  ← WORSE

→ Choose Glucose < 140 as the split.
```

---

### Impurity Measure 2 — Entropy and Information Gain

```
Entropy(node)  =  - Sum_k( p_k  x  log2(p_k) )

Example: node with 70% Diabetic, 30% Not Diabetic:
Entropy  =  -(0.70 x log2(0.70)  +  0.30 x log2(0.30))
          =  -(0.70 x (-0.515)   +  0.30 x (-1.737))
          =  0.881 bits

Entropy = 0    →  Pure node (zero uncertainty)
Entropy = 1.0  →  Maximum impurity (50/50 — maximum uncertainty)
```

**Information Gain formula (used in ID3):**

```
IG(parent, split)  =  Entropy(parent)  -  Sum[ (|child| / |parent|)  x  Entropy(child) ]
```

**Gini vs. Entropy — key differences:**

| Property | Gini | Entropy |
|---|---|---|
| Computation | Faster (no logarithm) | Slightly slower |
| Range (binary) | 0 to 0.5 | 0 to 1.0 |
| Sensitivity | Slightly less sensitive near pure nodes | Slightly more sensitive |
| Practical result | Nearly identical trees (~2% differ on split choice) | Nearly identical |
| Recommendation | Use sklearn default (Gini) | Use only if specific reason |

---

### Impurity Measure 3 — Variance Reduction (for Regression Trees)

For regression tasks (predicting blood sugar level), impurity is replaced by variance:

```
Variance(node)  =  (1/n)  x  Sum( (yi - y_mean)^2 )

Variance Reduction  =  Var(parent)  -  Sum[ (|child| / |parent|)  x  Var(child) ]
```

- Choose the split that maximally reduces variance
- Prediction at each leaf = **mean of target values in that leaf**

---

<a name="33-tree-building--the-full-algorithm"></a>

## 3.3 Tree Building — The Full Algorithm

```
RECURSIVE SPLITTING (CART — Classification and Regression Trees):

BuildTree(node, data, depth):

  Step 1 — Check stopping criteria:
    - All samples are the same class          → make leaf
    - max_depth reached                       → make leaf
    - Fewer samples than min_samples_split    → make leaf
    - No split improves impurity              → make leaf
    Leaf label = majority class (or mean for regression)

  Step 2 — Find best split:
    For each feature, for each threshold:
      Compute Information Gain
    Select (feature*, threshold*) with maximum gain

  Step 3 — Split data into Left and Right subsets

  Step 4 — Recursively call BuildTree(Left, depth+1)
  Step 5 — Recursively call BuildTree(Right, depth+1)
```

---

<a name="34-hyperparameters--the-pruning-dials"></a>

## 3.4 Hyperparameters — The Pruning Dials

Without constraints, a decision tree splits until every leaf has exactly one patient — perfect training accuracy, terrible test accuracy.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion         = 'gini',   # or 'entropy'
    max_depth         = 5,        # Maximum depth of the tree
    min_samples_split = 20,       # Min samples needed to split a node
    min_samples_leaf  = 10,       # Min samples required at a leaf
    max_features      = None,     # Features considered at each split
    class_weight      = 'balanced'
)
```

| Hyperparameter | Low value → | High value → |
|---|---|---|
| `max_depth` | Underfit (shallow, simple) | Overfit (deep, complex) |
| `min_samples_split` | Overfit (splits on tiny groups) | Underfit (few splits) |
| `min_samples_leaf` | Overfit (single-sample leaves) | Underfit (large leaves) |
| `max_features` | Less variance, more bias | More variance, more complexity |

**Visualising depth effect:**

```
max_depth = 2          max_depth = 5             max_depth = None (full)

   Glucose?               Glucose?                  Glucose?
   /      \               /        \                ....deep....
Diab.  BMI?          BMI?       Age+BP?             ............
        /   \          /\           /    \          ...hundreds of
      ND  Diab.    ND  Diab.    Diab.  Family?      ...tiny leaves
                                        /   \       (memorises noise)
                                     ND  Diab.

Train: 71%             Train: 88%                  Train: 100%
Test:  69%             Test:  84% ✅               Test:   67%
UNDERFIT               SWEET SPOT                  OVERFIT
```

---

<a name="35-visualising-and-interpreting-a-decision-tree"></a>

## 3.5 Visualising and Interpreting a Decision Tree

```python
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# Text representation
print(export_text(model, feature_names=list(X_train.columns)))

# Visual representation
plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=list(X_train.columns),
          class_names=['Not Diabetic', 'Diabetic'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Decision Tree — Diabetes Classification')
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

<a name="36-decision-tree--properties-strengths-weaknesses"></a>

## 3.6 Decision Tree — Properties, Strengths, Weaknesses

**Strengths:**

- **Fully interpretable** — every prediction has a clear, traceable rule path
- **No scaling needed** — splits are threshold-based, scale-invariant
- **Handles mixed types** — categorical and numerical features naturally
- **Non-linear boundaries** — complex patterns through hierarchical splits
- **Feature selection built-in** — irrelevant features never get chosen for splits
- **Fast prediction** — O(depth) per sample at inference

**Weaknesses:**

- **Overfitting** — without pruning, grows arbitrarily complex
- **High variance** — small data change can produce a completely different tree
- **Axis-aligned boundaries only** — cannot learn diagonal boundaries without many splits
- **Not great for regression** — predictions are piecewise constant (step functions), not smooth

> ⚠️ **The Variance Problem — Why Single Trees Are Unstable:**
> Train on Dataset A → Tree predicts Priya as DIABETIC.
> Train on Dataset B (same data, 5 patients swapped) → Tree predicts Priya as NOT DIABETIC.
> A tiny data change → completely different tree → completely different prediction.
> **This is exactly what Random Forest was designed to solve.**

---

<a name="chapter-4"></a>

# CHAPTER 4 — Random Forest

<a name="41-core-philosophy"></a>

## 4.1 Core Philosophy

> **"The wisdom of the crowd beats the opinion of any individual expert."**

Random Forest builds many decision trees — each trained on a slightly different version of the data — and combines their predictions. The key insight: while each individual tree has high variance, **the average of many independent high-variance estimators has low variance**.

This is **ensemble learning** — combining multiple weak/moderate learners to form a strong learner.

---

<a name="42-two-randomisation-tricks--the-heart-of-random-forest"></a>

## 4.2 Two Randomisation Tricks — The Heart of Random Forest

Random Forest introduces two sources of randomness to ensure trees are **diverse (uncorrelated)**. Correlated trees give you nothing — averaging ten trees that make the same mistakes doesn't help.

---

### Trick 1 — Bootstrap Sampling (Bagging)

Each tree is trained on a **bootstrap sample** — random sampling of the training data **with replacement**, same size as the original.

```
Original training data: 8,000 patients [P1, P2, P3, ..., P8000]

Tree 1 bootstrap sample (8,000 drawn WITH replacement):
  [P1, P1, P3, P7, P12, P12, P15, ..., P8000, P8000, P45]
   ↑↑ duplicated                              ↑↑ duplicated
   Some patients appear 2-3 times; ~37% never appear (Out-of-Bag)

Tree 2 bootstrap sample:  [P4, P4, P9, P10, P1, P22, ...]  ← different
Tree 3 bootstrap sample:  [P2, P8, P8, P11, P6, ...]       ← different again
```

Each tree sees a different subset → makes different errors → errors partially cancel.

**Out-Of-Bag (OOB) samples:** The ~37% of patients NOT selected for a tree's bootstrap. These provide a free, unbiased validation estimate — no separate validation set needed.

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")   # unbiased test accuracy estimate
```

---

### Trick 2 — Feature Subsampling (Random Subspace Method)

At **each split of each tree**, only a random subset of `max_features` features is considered.

```
Typical settings:
  Classification: max_features = sqrt(n_features) = sqrt(8) ≈ 3 features per split
  Regression:     max_features = n_features / 3

Example with 8 features {Glucose, BMI, Age, BP, Insulin, Smoking, Family, Diet}:

Tree 1, root split:   Consider only {Glucose, Age, Smoking}  → best = Glucose < 140
Tree 1, next node:    Consider only {BMI, Insulin, Diet}     → best = BMI < 27
Tree 2, root split:   Consider only {BP, Family, Smoking}    → best = Family = Yes
Tree 2, next node:    Consider only {Glucose, Age, BMI}      → best = Glucose < 155
```

Without this trick, Glucose would dominate every split in every tree → all trees look similar → highly correlated → averaging helps little. Feature subsampling forces tree **diversity**.

> This is the key difference between **Bagging** (just resampling data) and **Random Forest** (resampling data + randomising features at each split).

---

<a name="43-prediction--how-the-forest-votes"></a>

## 4.3 Prediction — How the Forest Votes

**Classification — Majority Vote:**

```
100 trees predict for new patient Priya:
  73 trees say → DIABETIC
  27 trees say → NOT DIABETIC

Final prediction: DIABETIC
Probability:      73/100 = 0.73
```

**Regression — Average:**

```
100 trees predict blood sugar level:
  Tree 1: 161,  Tree 2: 158,  Tree 3: 172,  ...,  Tree 100: 163

Final prediction = mean = 162.4 mg/dL
```

---

<a name="44-why-averaging-reduces-variance--the-math"></a>

## 4.4 Why Averaging Reduces Variance — The Math

Each tree has variance `sigma^2` and pairwise correlation `rho` with other trees.

```
Variance of the Forest (average of n trees):

  Var(Forest)  =  rho * sigma^2  +  (1 - rho) * sigma^2 / n

When rho = 1 (perfectly correlated trees):
  Var(Forest) = sigma^2          ← no improvement (all same mistakes)

When rho = 0 (perfectly independent trees):
  Var(Forest) = sigma^2 / n      ← variance shrinks by factor n

With bootstrap + feature subsampling:  rho ≈ 0.05–0.15

Example (100 trees, sigma^2=1, rho=0.1):
  Var(Forest) = 0.1*1 + 0.9*(1/100) = 0.1 + 0.009 = 0.109

Single tree variance: 1.0
Forest variance:      0.109   ← 9x reduction in variance!
```

---

<a name="45-hyperparameters"></a>

## 4.5 Hyperparameters

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators      = 100,        # Number of trees
    max_depth         = 10,         # Depth of each tree (None = fully grown)
    min_samples_split = 10,         # Min samples to split a node
    min_samples_leaf  = 5,          # Min samples at a leaf
    max_features      = 'sqrt',     # Features per split (key for tree diversity)
    bootstrap         = True,       # Use bootstrap sampling
    oob_score         = True,       # Compute OOB error estimate
    class_weight      = 'balanced', # Handle class imbalance
    n_jobs            = -1,         # Use all CPU cores
    random_state      = 42
)
```

| Hyperparameter | Effect | Typical range |
|---|---|---|
| `n_estimators` | More trees = lower variance (diminishing returns after ~200) | 100–500 |
| `max_depth` | Deeper = more complex = more overfitting risk | 5–20 or None |
| `max_features` | Lower = more diverse trees = lower variance, higher bias | 'sqrt', 'log2', 0.3–0.7 |
| `min_samples_leaf` | Higher = simpler trees = lower variance | 1–20 |
| `bootstrap` | True = bagging; False = each tree uses all data | True |

**The n_estimators diminishing returns curve:**

```
n_estimators =  10:   Test F1 = 0.81  (too few — high variance)
n_estimators =  50:   Test F1 = 0.86
n_estimators = 100:   Test F1 = 0.88
n_estimators = 200:   Test F1 = 0.89
n_estimators = 500:   Test F1 = 0.89  ← diminishing returns after ~200
n_estimators = 1000:  Test F1 = 0.89  ← same result, 5x the compute

→ More trees never hurt (no overfitting risk).
  Unlike max_depth, n_estimators only has diminishing compute returns.
```

---

<a name="46-feature-importance-in-random-forest"></a>

## 4.6 Feature Importance in Random Forest

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title('Random Forest Feature Importances — Diabetes Prediction')
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.show()
```

**Sample output:**

```
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

---

<a name="47-random-forest--properties-strengths-weaknesses"></a>

## 4.7 Random Forest — Properties, Strengths, Weaknesses

**Strengths:**

- **Low variance** — bagging + feature subsampling make it very robust
- **Excellent out-of-the-box** — good performance with minimal tuning
- **OOB error** — free internal validation, no separate validation set needed
- **Handles high dimensionality** — effective even when features >> samples
- **Parallel training** — each tree trains independently (`n_jobs=-1`)
- **Robust to outliers** — tree splits are threshold-based
- **Feature importance** — built-in global feature ranking

**Weaknesses:**

- **Black-box** — 100+ trees cannot be read directly (needs SHAP for explanation)
- **Memory intensive** — stores all trees in memory
- **Slower inference** — must traverse 100+ trees per prediction
- **Biased feature importance** — impurity-based importance is biased toward high-cardinality features

---

<a name="chapter-5"></a>

# CHAPTER 5 — XGBoost (Extreme Gradient Boosting)

<a name="51-core-philosophy"></a>

## 5.1 Core Philosophy

> **"Don't train all trees simultaneously — train them sequentially. Each new tree focuses specifically on correcting the mistakes made by all previous trees."**

XGBoost is a **boosting** algorithm — the opposite of bagging (Random Forest):

```
BAGGING (Random Forest):
  Tree 1  ──────────────┐
  Tree 2  ──────────────┤→ Average → Final Prediction
  Tree 3  ──────────────┤
  ...                   │
  Tree 100──────────────┘
  (All built independently, simultaneously)

BOOSTING (XGBoost):
  Tree 1  →  has errors
  Tree 2  →  built to fix Tree 1's errors           →  fewer errors
  Tree 3  →  built to fix Trees 1+2 errors          →  fewer errors
  ...
  Tree 100 → built to fix residual errors of 1-99
  Final    =  weighted sum of all 100 trees
```

---

<a name="52-gradient-boosting--the-core-mechanism"></a>

## 5.2 Gradient Boosting — The Core Mechanism

XGBoost uses **gradient descent in function space** — instead of minimising loss by adjusting weights, minimise loss by adding new trees.

### Step-by-Step Walkthrough (Classification)

**Setup:** Loss function = Log Loss. Base rate of diabetes = 30%.

```
ROUND 0 — Initial Prediction:
  F0(x) = log(0.30/0.70) = -0.847  (log-odds)
  → Everyone gets 30% predicted probability. Not great, but a start.

ROUND 1 — Compute Residuals (Pseudo-Residuals):
  r_i  =  y_i  -  p_hat_i    (true label minus current prediction)

  Patient Priya (y=1, Diabetic):       r = 1 - 0.30 = +0.70  (under-predicted)
  Patient Arjun (y=0, Not Diabetic):   r = 0 - 0.30 = -0.30  (over-predicted)

ROUND 1 — Train Tree 1 to predict these residuals:
  Tree 1 learns: "given features, how much should we correct the prediction?"
  Example:   Glucose > 155 AND BMI > 30  →  residual ≈ +0.65
             Glucose < 120               →  residual ≈ -0.25

ROUND 1 — Update Predictions:
  F1(x)  =  F0(x)  +  eta  x  Tree1(x)      (eta = learning rate, e.g. 0.1)

  Priya: -0.847 + 0.1 x 0.65 = -0.782  →  sigmoid(-0.782) = 0.314
  (Now predicting 31.4% — slightly better. Priya is actually diabetic.)

ROUND 2 — Compute new residuals from F1:
  Priya's residual: 1 - 0.314 = +0.686  (still under-predicting)
  Train Tree 2 on these new residuals.
  F2(x) = F1(x) + eta x Tree2(x)

...Continue for T rounds...

ROUND T — Final Prediction:
  F_T(x)  =  F0(x)  +  eta  x  Sum_t[ Tree_t(x) ]
  p_hat   =  sigmoid( F_T(x) )
```

> **The key insight:** Each tree is a correction to the previous ensemble's mistakes. The ensemble iteratively focuses on the hardest patients — those where the current prediction is most wrong.

---

<a name="53-the-xgboost-objective-function"></a>

## 5.3 The XGBoost Objective Function

XGBoost uses a more sophisticated objective than traditional gradient boosting:

```
Obj  =  Sum_i[ L(yi, y_hat_i) ]   +   Sum_t[ Omega(ft) ]
        ─────────────────────────       ───────────────────
              Training Loss                Regularisation

where:
  L(yi, y_hat_i)  = loss per patient (Log Loss for classification, MSE for regression)

  Omega(ft)  =  gamma * T  +  (1/2) * lambda * Sum_j( w_j^2 )
                 ─────────      ─────────────────────────────
                 penalty on     L2 penalty on
                 number of      leaf weights
                 leaves T
```

**Second-order Taylor expansion — what makes XGBoost "eXtreme":**

Standard GBM uses only the first derivative (gradient) of the loss. XGBoost uses **both first and second derivatives**:

```
Approximation at current prediction F_{t-1}:

L(yi, F_{t-1} + ft)  ≈  L(yi, F_{t-1})
                       + gi * ft(xi)                 ← first-order gradient
                       + (1/2) * hi * ft(xi)^2       ← second-order Hessian

where:
  gi  =  first derivative of loss w.r.t. F_{t-1}
  hi  =  second derivative of loss w.r.t. F_{t-1}
```

This allows XGBoost to **analytically compute** optimal leaf weights (closed-form solution):

```
Optimal leaf weight:
  w*_j  =  -(Sum of gi in leaf j)  /  (Sum of hi in leaf j  +  lambda)

Gain from a split (left L + right R vs parent P):
  Gain  =  (1/2) * [ G_L^2/(H_L + lambda)  +  G_R^2/(H_R + lambda)  -  G_P^2/(H_P + lambda) ]  -  gamma
```

The `-gamma` at the end means a split is only made if gain exceeds gamma — this is **automatic pruning** built into the gain calculation itself.

> **Why second-order matters:** Traditional GBM only uses gradient direction. XGBoost uses both direction and curvature — like the difference between gradient descent and Newton's method. Fewer trees needed for the same performance.

---

<a name="54-key-xgboost-hyperparameters"></a>

## 5.4 Key XGBoost Hyperparameters

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    # Boosting
    n_estimators        = 300,    # number of trees (boosting rounds)
    learning_rate       = 0.05,   # eta — step size (shrinkage)

    # Tree structure
    max_depth           = 6,      # depth of each individual tree
    min_child_weight    = 1,      # min sum of instance weight in a leaf
    gamma               = 0,      # min gain required to make a split

    # Randomisation (reduces overfitting)
    subsample           = 0.8,    # fraction of training samples per tree
    colsample_bytree    = 0.8,    # fraction of features per tree

    # Regularisation
    reg_alpha           = 0,      # L1 on leaf weights
    reg_lambda          = 1,      # L2 on leaf weights

    # Imbalance
    scale_pos_weight    = 1,      # set to neg/pos ratio for imbalanced data

    eval_metric         = 'logloss',
    random_state        = 42
)
```

---

### The Most Critical Hyperparameters

**1. learning_rate (eta) + n_estimators — Always Tune Together**

```
F_t(x)  =  F_{t-1}(x)  +  eta  x  tree_t(x)

Low eta (0.01):   tiny corrections → need MORE trees → smoother, less overfit
High eta (0.3):   big corrections  → need FEWER trees → can overshoot, more overfit

RULE: Lower eta + more trees almost always outperforms high eta + few trees.

Performance comparison:
  eta=0.3,  50 trees:   Test AUC = 0.87
  eta=0.1,  150 trees:  Test AUC = 0.90
  eta=0.05, 300 trees:  Test AUC = 0.92  ✅
  eta=0.01, 1500 trees: Test AUC = 0.92  (same but 5x slower)
```

**2. subsample + colsample_bytree — Stochastic Boosting**

```
subsample = 0.8:
  Each tree trained on 80% randomly sampled patients (without replacement)
  → Reduces overfitting, adds randomness similar to bagging

colsample_bytree = 0.8:
  Each tree uses 80% of features
  → Reduces correlation between trees (borrowing Random Forest's trick)
```

**3. Early Stopping — The Most Practical Tool**

```python
model = xgb.XGBClassifier(
    n_estimators          = 1000,   # set high — early stopping decides actual number
    learning_rate         = 0.05,
    early_stopping_rounds = 50,     # stop if no improvement for 50 rounds
    eval_metric           = 'logloss'
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=50)

print(f"Best iteration: {model.best_iteration}")
```

```
Round 100:  val-logloss = 0.321
Round 150:  val-logloss = 0.298
Round 200:  val-logloss = 0.287  ← best so far
Round 250:  val-logloss = 0.291  ← worse
Round 300:  val-logloss = 0.295  ← worse
...
Round 251: STOPPING (50 rounds without improvement)
Best model = Round 200
```

> Early stopping finds the exact number of trees needed — no manual searching, prevents overfitting, saves compute time.

---

<a name="55-xgboost-vs-random-forest--the-key-differences"></a>

## 5.5 XGBoost vs. Random Forest — The Key Differences

| Property | Random Forest | XGBoost |
|---|---|---|
| **Tree building** | Parallel — all trees built independently | Sequential — each tree corrects previous |
| **Error focus** | Each tree sees different data (bootstrap) | Each tree explicitly targets current errors |
| **Bias-Variance** | Primarily reduces Variance | Reduces both Bias AND Variance |
| **Fixing underfitting** | Hard — cannot decrease bias easily | Easy — just add more rounds |
| **Fixing overfitting** | max_depth, max_features | learning_rate, regularisation, gamma |
| **Training speed** | Fast (parallel, CPU-efficient) | Slower (sequential) but highly optimised |
| **Regularisation** | Implicit (bootstrap noise, feature sampling) | Explicit — L1/L2 on leaf weights + gamma |
| **Imbalanced data** | `class_weight='balanced'` | `scale_pos_weight` |
| **Out-of-the-box** | Great — robust to default settings | Good — learning_rate matters a lot |
| **Tuned performance** | Very good | Usually better than RF |

---

<a name="56-lightgbm-and-catboost--brief-mentions"></a>

## 5.6 LightGBM and CatBoost — Brief Mentions

**LightGBM** (Microsoft, 2017):

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators     = 300,
    learning_rate    = 0.05,
    num_leaves       = 31,    # controls complexity (instead of max_depth)
    max_depth        = -1,    # -1 = no limit
    subsample        = 0.8,
    colsample_bytree = 0.8
)
```

Key innovations: **leaf-wise tree growth** (grows the most gainful leaf rather than full level) + **GOSS sampling** (keeps all high-gradient samples, subsamples low-gradient ones). Result: significantly faster training than XGBoost with similar accuracy.

**CatBoost** (Yandex, 2017):

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations    = 300,
    learning_rate = 0.05,
    depth         = 6,
    cat_features  = ['Smoking_Status', 'Diet_Quality']  # no encoding needed!
)
```

Key innovation: handles categorical features **natively** without one-hot or label encoding, using an ordered boosting strategy that reduces prediction shift bias.

---

<a name="chapter-6"></a>

# CHAPTER 6 — Side-by-Side Comparison: All Four Algorithms

## 6.1 Algorithm Philosophy Summary

| | Naive Bayes | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| **Core question** | Which class generated these features? | Which yes/no questions separate the classes? | What does the majority of 100 trees say? | What mistakes does the current ensemble make? |
| **Type** | Generative, probabilistic | Discriminative, deterministic | Ensemble (Bagging) | Ensemble (Boosting) |
| **Parametric?** | Yes | No | No | No |
| **Builds on** | Bayes' theorem | Greedy splitting | Parallel trees | Sequential error correction |

## 6.2 Technical Properties

| Property | Naive Bayes | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| **Scaling needed** | No | No | No | No |
| **Handles missing values** | Yes (naturally) | No (needs imputation) | No (needs imputation) | Yes (learned direction) |
| **Handles categoricals** | Yes (Multinomial/Bernoulli) | Yes (with encoding) | Yes (with encoding) | Yes (encoding or CatBoost) |
| **Non-linear boundaries** | Yes (curved) | Yes (axis-aligned) | Yes (complex) | Yes (complex) |
| **Interpretable** | Moderate | Yes (if shallow) | No (needs SHAP) | No (needs SHAP) |
| **Training speed** | Very Fast | Fast | Moderate (parallel) | Moderate (sequential) |
| **Inference speed** | Very Fast | Fast | Moderate | Moderate |
| **Overfit risk** | Low | HIGH | Low-Moderate | Moderate (needs tuning) |
| **Underfit risk** | Moderate (independence assumption) | Low | Low | Low |
| **Imbalanced data** | Moderate | Poor | `class_weight` | `scale_pos_weight` |

## 6.3 When to Use Each

| Situation | Best Choice |
|---|---|
| Text classification, spam, NLP | Naive Bayes |
| Need fully interpretable model (legal, medical) | Decision Tree (shallow) |
| Quick powerful baseline, minimal tuning | Random Forest |
| Maximum tabular performance, competition | XGBoost / LightGBM |
| Very small dataset (< 500 samples) | Naive Bayes or Decision Tree |
| Features are approximately independent | Naive Bayes |
| Real-time / streaming learning | Naive Bayes |
| Categorical features with many values | CatBoost |
| Large dataset (> 1M rows) | LightGBM (faster than XGBoost) |

## 6.4 Performance on Our Diabetes Dataset

```
MODEL COMPARISON (after hyperparameter tuning + cross-validation):

                  CV F1    Test F1   Test AUC   Train Time   Interpretable?
Naive Bayes       0.74     0.72      0.81       < 1 second   Moderate
Decision Tree     0.80     0.78      0.85         2 seconds  YES (shallow)
Random Forest     0.87     0.86      0.92        45 seconds  NO (SHAP needed)
XGBoost           0.91     0.89      0.95        90 seconds  NO (SHAP needed)
LightGBM          0.91     0.89      0.95        30 seconds  NO (SHAP needed)

Performance hierarchy for tabular data:
  Naive Bayes < Decision Tree < Random Forest ≈ LightGBM ≈ XGBoost
  (For text: Naive Bayes ≈ Logistic Reg >> Tree-based models)
```

---

<a name="chapter-7"></a>

# CHAPTER 7 — Bias-Variance for All Four Algorithms

## 7.1 Where Each Model Lives on the Spectrum

```
← HIGH BIAS (Underfit)                              HIGH VARIANCE (Overfit) →

Naive Bayes      Decision Tree   Random Forest   XGBoost       Decision Tree
(independence    (max_depth=2)   (n=100,          (n=500,       (no max_depth,
 assumption                       max_depth=10)    lr=0.01)      no pruning)
 violated)

Fix:             Fix:            Fix:            Fix:          Fix:
More features,   Deeper tree,    More trees,     More rounds,  max_depth,
feature eng.     remove min      lower           lower lr,     min_samples,
                 constraints     max_features    subsample     gamma
```

## 7.2 Diagnosing Each Algorithm — The 2-Number Test

| Algorithm | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Naive Bayes | Train=76%, Test=74% (small gap, both moderate) | High Bias — independence violated | Switch to RF or tree-based model |
| Decision Tree | Train=100%, Test=67% (large gap) | High Variance — Overfit | max_depth=5, min_samples_leaf=10 |
| Decision Tree | Train=72%, Test=70% (small gap, both low) | High Bias — Underfit | Increase max_depth, remove min constraints |
| Random Forest | Train=94%, Test=88% (moderate gap) | Slightly Overfit | Increase min_samples_leaf, decrease max_depth |
| XGBoost | Train=99%, Test=71% (very large gap) | Severe Overfit | Lower lr (0.3→0.05), add subsample, increase gamma |
| XGBoost | Train=82%, Test=81% (tiny gap, both high) | Sweet Spot ✅ | Deploy |

---

<a name="chapter-8"></a>

# CHAPTER 8 — Evaluation Metrics Quick Reference

## 8.1 Classification Metrics (Same for All Four)

All four algorithms ultimately produce `y_hat in {0, 1}`. The confusion matrix applies uniformly.

```
From 2,000 test patients (600 Diabetic, 1,400 Not Diabetic):

                   PREDICTED
             Not Diabetic    Diabetic
ACTUAL Not D   TN = 1,312     FP = 88
ACTUAL Diab.   FN =    78     TP = 522
```

**Computed metrics:**

```
Accuracy   =  (1312 + 522) / 2000              =  91.7%
Precision  =  522 / (522 + 88)                 =  85.6%
Recall     =  522 / (522 + 78)                 =  87.0%  ← most important in medical context
F1         =  2 x 0.856 x 0.870 / (0.856 + 0.870) =  86.3%
```

> ⚠️ In diabetes screening, **Recall** is the priority — missing a real diabetic (FN=78) is far more dangerous than a false alarm (FP=88).

## 8.2 Regression Metrics (Decision Tree, Random Forest, XGBoost)

```
For blood sugar prediction on 2,000 test patients:

MAE   =  Mean( |y_hat - y| )          =   8.3 mg/dL   (average absolute error)
RMSE  =  sqrt( Mean( (y_hat-y)^2 ) )  =  12.1 mg/dL   (penalises large errors more)
R^2   =  1  -  SS_res / SS_tot        =   0.87         (explains 87% of variance)
```

---

<a name="chapter-9"></a>

# CHAPTER 9 — Quick Revision: All Formulae in One Place

## Naive Bayes Formulae

```
── BAYES' THEOREM ────────────────────────────────────────────────

P(y | x)  =  P(x | y)  x  P(y)  /  P(x)

Naive assumption:
  P(x | y)  =  P(x1|y) x P(x2|y) x ... x P(xn|y)

── GAUSSIAN LIKELIHOOD ───────────────────────────────────────────

P(xi | y)  =  (1 / sqrt(2 * pi * sigma^2_iy))  x  exp( -(xi - mu_iy)^2 / (2*sigma^2_iy) )

── MULTINOMIAL LIKELIHOOD ────────────────────────────────────────

P(word_i | y)  =  (count(word_i, y) + alpha)  /  (count(y) + alpha x vocab_size)

── BERNOULLI LIKELIHOOD ──────────────────────────────────────────

P(xi | y)  =  p_iy^xi  x  (1 - p_iy)^(1-xi)

── LAPLACE SMOOTHING ─────────────────────────────────────────────

P(xi | y)  =  (count(xi, y) + alpha)  /  (count(y) + alpha x K)
```

## Decision Tree Formulae

```
── GINI INDEX ────────────────────────────────────────────────────

Gini(node)  =  1  -  Sum_k( p_k^2 )

Information Gain:
  IG  =  Gini(parent)  -  [ (n_L/n)*Gini(Left)  +  (n_R/n)*Gini(Right) ]

── ENTROPY ───────────────────────────────────────────────────────

Entropy(node)  =  - Sum_k( p_k x log2(p_k) )

Information Gain:
  IG  =  Entropy(parent)  -  Sum[ (|child|/|parent|) x Entropy(child) ]

── VARIANCE REDUCTION (regression) ──────────────────────────────

Var(node)         =  (1/n) x Sum( (yi - y_mean)^2 )
Variance Reduction =  Var(parent)  -  Sum[ (|child|/|parent|) x Var(child) ]
```

## Random Forest Formulae

```
── VARIANCE REDUCTION BY AVERAGING ──────────────────────────────

Var(Forest)  =  rho x sigma^2  +  (1 - rho) x sigma^2 / n

where rho = pairwise correlation between trees, n = number of trees

── TYPICAL SETTINGS ──────────────────────────────────────────────

max_features  =  sqrt(n_features)    for classification
max_features  =  n_features / 3      for regression
OOB fraction  ≈  0.37                (~37% not sampled per bootstrap)
```

## XGBoost Formulae

```
── ENSEMBLE UPDATE ───────────────────────────────────────────────

F_t(x)  =  F_{t-1}(x)  +  eta  x  h_t(x)

where eta = learning rate,  h_t = new tree

── OBJECTIVE FUNCTION ────────────────────────────────────────────

Obj  =  Sum_i[ L(yi, y_hat_i) ]  +  Sum_t[ gamma*T + (1/2)*lambda*Sum_j(w_j^2) ]

── OPTIMAL LEAF WEIGHT ───────────────────────────────────────────

w*_j  =  -(Sum_i_in_j[ g_i ])  /  (Sum_i_in_j[ h_i ]  +  lambda)

── SPLIT GAIN ────────────────────────────────────────────────────

Gain  =  (1/2)*[ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G_P^2/(H_P+lambda) ] - gamma

where G = sum of gradients in node,  H = sum of Hessians in node

── GRADIENTS (Log Loss) ──────────────────────────────────────────

g_i  =  y_hat_i  -  y_i          (first derivative — residual)
h_i  =  y_hat_i x (1 - y_hat_i)  (second derivative — Hessian)
```

## Evaluation Metrics

```
── CLASSIFICATION ────────────────────────────────────────────────

Accuracy    =  (TP + TN)  /  (TP + TN + FP + FN)
Precision   =  TP  /  (TP + FP)
Recall      =  TP  /  (TP + FN)
F1          =  2 x Precision x Recall  /  (Precision + Recall)
Specificity =  TN  /  (TN + FP)

── REGRESSION ────────────────────────────────────────────────────

MAE   =  (1/n) x Sum( |y_hat - y| )
RMSE  =  sqrt( (1/n) x Sum( (y_hat - y)^2 ) )
R^2   =  1  -  SS_res / SS_tot
```

---

<a name="chapter-10"></a>

# CHAPTER 10 — Viva / Interview Q&A Bank

> 📖 Read each answer once before a viva or interview. These cover every conceptual question asked about these four algorithms.

---

### Q1. What is the "Naive" assumption in Naive Bayes, and why does the model still work despite it being wrong?

The naive assumption is that all features are conditionally independent given the class label — P(x|y) = product of P(xi|y). In reality, features like Glucose and BMI are correlated. The model is rarely correct about this. However, the model's job is classification — deciding which class has the higher posterior, not computing accurate probabilities. Even when individual probability estimates are wrong, the **ranking** of classes (which posterior is larger) is often correct. The wrong magnitudes don't matter; the correct ordering does. This is why Naive Bayes classifies correctly even when its probability estimates are inaccurate.

---

### Q2. What is Laplace Smoothing and why is it needed?

If a feature value never appeared in training data for a particular class, its count is zero, making P(xi|y) = 0. Since Naive Bayes multiplies all feature probabilities together, a single zero makes the entire posterior zero — the model can never assign that class regardless of all other evidence. Laplace smoothing adds a pseudocount alpha (usually 1) to all counts before computing probabilities. This ensures every probability is positive, preventing the zero-product catastrophe. The tradeoff: probabilities are slightly less accurate, but the model never catastrophically assigns zero probability to a valid class.

---

### Q3. What is Information Gain in a Decision Tree and why is it used?

Information Gain measures how much a proposed split reduces impurity at a node. It is computed as: IG = Impurity(parent) − weighted average Impurity(children). The algorithm evaluates every possible (feature, threshold) pair and chooses the one with maximum IG. This greedy, top-down approach builds the tree one split at a time, always choosing the locally best split. Using Entropy, IG comes from information theory — it measures the reduction in uncertainty (Shannon entropy) achieved by learning a feature's value.

---

### Q4. What is the difference between Gini Index and Entropy as splitting criteria?

Both measure node impurity. Gini = 1 − Sum(pk²), ranges 0 to 0.5 for binary. Entropy = −Sum(pk × log2(pk)), ranges 0 to 1 for binary. Computationally, Gini is faster (no logarithm). In practice, they produce nearly identical trees — the choice of split point differs in only ~2% of cases. Entropy is slightly more sensitive to changes near pure nodes. Use sklearn's default Gini unless you have a specific reason to prefer Entropy.

---

### Q5. Why does a single Decision Tree have high variance, and how does Random Forest fix it?

A single tree has high variance because small changes in training data can produce a completely different tree structure — a different root split leads to entirely different branches below it. Random Forest reduces variance through two mechanisms: (1) **Bootstrap sampling** — each tree trains on a different random subset with replacement, so each makes different errors. (2) **Feature subsampling** — at each split, only sqrt(n_features) are considered, decorrelating the trees. The forest averages their predictions. Mathematically: Var(Forest) = rho × sigma² + (1−rho) × sigma²/n. Lower correlation rho between trees leads to greater variance reduction.

---

### Q6. What is Out-Of-Bag error in Random Forest?

About 37% of training samples are not included in any given tree's bootstrap sample — these are the Out-Of-Bag (OOB) samples for that tree. Each training sample can be evaluated by all trees for which it was OOB. Averaging these evaluations gives the OOB error — a free, unbiased estimate of the model's generalisation performance, similar to cross-validation, without requiring a separate validation set. OOB error is a good proxy for test error and is useful for hyperparameter selection during training.

---

### Q7. Explain Gradient Boosting in simple terms.

Gradient Boosting trains trees sequentially, where each new tree targets the mistakes of the previous ensemble. Start with a constant prediction (e.g., the base rate). Compute residuals — how wrong the current prediction is for each sample. Train a new shallow tree to predict these residuals. Add this tree (scaled by a small learning rate) to the ensemble. Compute new residuals and repeat. After many rounds, the ensemble has iteratively corrected its errors, focusing most on the hardest samples. The word "gradient" means we use gradient descent in function space — each tree is a step in the direction that most reduces the loss function.

---

### Q8. What is the difference between Random Forest and XGBoost?

Random Forest builds trees in **parallel** using bootstrap sampling and feature subsampling, then averages predictions. It primarily reduces variance. XGBoost builds trees **sequentially** where each tree corrects the residual errors of all previous trees — reducing both bias and variance. XGBoost also uses explicit L1/L2 regularisation on leaf weights and a second-order Taylor expansion (gradient + Hessian) for more accurate updates. Random Forest works well out-of-the-box with minimal tuning. XGBoost typically achieves higher performance with proper tuning and is the dominant algorithm in tabular data competitions.

---

### Q9. What is the role of the learning rate in XGBoost, and how does it interact with n_estimators?

The learning rate eta controls how much each new tree contributes: F_t = F_{t-1} + eta × tree_t. A small eta means each tree makes tiny corrections — you need more trees but the final model generalises better because it takes small, careful steps toward the optimum. A large eta means big corrections — fewer trees but risk overshooting. The practical rule: use a small learning rate (0.05) + many trees (200–1000) + early stopping. This combination almost always outperforms large eta + few trees. Early stopping automatically finds the optimal n_estimators for any given learning rate.

---

### Q10. When would you choose Naive Bayes over Random Forest despite RF usually performing better?

Choose Naive Bayes when: (1) **Speed is critical** — NB trains in milliseconds and predicts in microseconds, ideal for real-time systems; (2) **Very small datasets** — RF can overfit with few samples, while NB's simple statistics are more stable; (3) **Text classification** — Multinomial NB with bag-of-words is fast, effective, and a strong NLP baseline; (4) **Streaming/online learning** — NB can update its parameters incrementally as new data arrives; (5) **Fast baseline needed** — NB is often the first model tried in a new classification problem. In practice, if features are nearly independent (as in NLP), NB can even outperform RF.

---

### Q11. What is the difference between bagging and boosting?

**Bagging** (Bootstrap Aggregating) trains multiple models in **parallel**, each on a different bootstrap sample, and combines them by averaging (regression) or majority vote (classification). The goal is to reduce variance. Random Forest is the canonical bagging algorithm. **Boosting** trains models **sequentially** where each model focuses on the errors of the previous ensemble — iteratively correcting mistakes. The goal is to reduce both bias and variance. XGBoost, LightGBM, and AdaBoost are boosting algorithms. Bagging is more robust and parallelisable; boosting is typically more accurate but requires careful tuning to avoid overfitting.

---

### Q12. How does XGBoost handle missing values?

XGBoost learns a **default direction** for each split during training — it tries sending missing-value samples to both the left and right child, then keeps whichever direction maximises the gain. This learned routing is stored as part of the tree. At inference time, any missing value is automatically routed in the learned default direction. This means you do not need to impute missing values before using XGBoost — it handles them natively and often learns the optimal routing based on the rest of the data.

---

### Q13. What is the XGBoost objective function and why is regularisation built into it?

The XGBoost objective is: Obj = Sum[L(yi, y_hat_i)] + Sum[Omega(ft)] where Omega(ft) = gamma×T + (1/2)×lambda×Sum(wj²). The first term is training loss (how well we fit the data). The second term is regularisation on each tree — gamma penalises the number of leaf nodes (pruning), and lambda penalises large leaf weights (equivalent to L2/Ridge regularisation). By including regularisation directly in the objective, XGBoost automatically avoids overly complex trees without needing external cross-validation for depth control. The gamma term specifically means a split is only made when the gain exceeds gamma — automatic pruning built into the gain calculation.

---

### Q14. What is feature importance in Random Forest and what are its limitations?

Feature importance in Random Forest is the mean decrease in impurity — the total reduction in Gini or entropy across all splits that use a particular feature, averaged over all trees. It gives a global ranking of which features contribute most to the model. **Limitations:** (1) Biased toward high-cardinality and continuous features — features with many possible values get more split opportunities; (2) Shows only magnitude, not direction — you cannot tell if a feature increases or decreases predictions; (3) Misleading for correlated features — importance is split arbitrarily between correlated features; (4) Global only — says nothing about why the model made a specific individual prediction. Use permutation importance or SHAP for more reliable and complete explanations.

---

> **The one insight that connects all four algorithms:**
>
> *Naive Bayes asks what the data looked like when it was generated.*
> *Decision Trees ask which questions best separate the data.*
> *Random Forest asks what 100 differently trained experts would collectively say.*
> *XGBoost asks what the current ensemble is still getting wrong — and fixes it.*
>
> *All four ultimately answer the same question: given these features, what is the most honest prediction I can make? They just approach that honesty from very different directions.*