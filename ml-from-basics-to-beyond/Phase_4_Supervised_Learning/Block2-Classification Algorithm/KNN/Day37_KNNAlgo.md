# K-Nearest Neighbors (KNN) — Complete Notes
> A foundational non-parametric classification and regression algorithm

---

## Table of Contents
1. [Core Intuition](#1-core-intuition)
2. [Distance Metrics](#2-distance-metrics)
3. [Choosing K](#3-choosing-k)
4. [Curse of Dimensionality](#4-curse-of-dimensionality)
5. [Lazy Learner](#5-lazy-learner)
6. [Worked Example — Binary Classification](#6-worked-example--binary-classification)
7. [Distance Metrics — Worked Comparisons](#7-distance-metrics--worked-comparisons)
8. [Choosing K — All Methods](#8-choosing-k--all-methods)
9. [Complete Summary](#9-complete-summary)

---

## 1. Core Intuition

### The Fundamental Idea

> *"Tell me who your neighbors are, and I'll tell you who you are."*

To classify a new data point, find the **K training examples closest to it** and let them **vote**. Whichever class the majority belong to becomes the prediction. No gradient descent. No weights to learn. Just — find the nearest neighbors and count votes.

### Classification vs. Regression

| Task | KNN Prediction Rule |
|---|---|
| Classification | Majority vote of K neighbors' class labels |
| Regression | Average of K neighbors' output values |

**Regression formula:**

$$\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\vec{x})} y^{(i)}$$

where $\mathcal{N}_K(\vec{x})$ denotes the set of K nearest neighbors of point $\vec{x}$.

### A Concrete Example

Classifying a house as *Expensive* or *Affordable* based on size and age:

```
Age
  |
  |  ○  ○              ○ = Affordable
  |     ○   ?          × = Expensive
  |  ×    ×            ? = New house to classify
  |     ×
  +------------------→ Size
```

- **K = 3:** 2 Expensive + 1 Affordable → **Predict: Expensive**
- **K = 5:** 3 Affordable + 2 Expensive → **Predict: Affordable**

The prediction can change depending on K. Choosing K carefully matters enormously.

---

## 2. Distance Metrics

KNN's entire logic rests on the concept of **closeness** — formally defined by a distance metric. Different metrics measure closeness differently, changing which neighbors are selected and therefore what prediction is made.

### 2.1 Euclidean Distance (p = 2)

$$d(\vec{x}, \vec{z}) = \sqrt{\sum_{j=1}^{n}(x_j - z_j)^2}$$

The straight-line distance between two points. All points at equal Euclidean distance form a **circle** (2D) or **sphere** (3D).

```
     * * *
   *       *
  *    ×    *   ← equal distance = circle
   *       *
     * * *
```

**Example:** House A = [3, 4], House B = [6, 8]
$$d = \sqrt{(6-3)^2 + (8-4)^2} = \sqrt{9 + 16} = 5$$

**Use when:** Features are continuous, on comparable scales, and feature scaling has been applied.

---

### 2.2 Manhattan Distance (p = 1)

$$d(\vec{x}, \vec{z}) = \sum_{j=1}^{n}|x_j - z_j|$$

Named after Manhattan's grid layout — distance if you can only move horizontally or vertically. All points at equal Manhattan distance form a **diamond**.

```
      *
     * *
    *  ×  *   ← equal distance = diamond
     * *
      *
```

**Same example:** $d = |6-3| + |8-4| = 3 + 4 = 7$

**Use when:** Data is high-dimensional or sparse; want less sensitivity to outliers (squaring in Euclidean amplifies large differences; Manhattan does not).

---

### 2.3 Minkowski Distance — The General Form

$$d(\vec{x}, \vec{z}) = \left(\sum_{j=1}^{n}|x_j - z_j|^p\right)^{1/p}$$

Unified formula containing all common metrics as special cases:

| p value | Reduces to | Shape |
|---|---|---|
| p = 1 | Manhattan Distance | Diamond |
| p = 2 | Euclidean Distance | Circle / Sphere |
| p → ∞ | Chebyshev Distance | Square |

**Chebyshev Distance** $(p \to \infty)$: $d(\vec{x}, \vec{z}) = \max_j(|x_j - z_j|)$ — takes only the largest difference across all dimensions.

> **p is itself a hyperparameter.** Start with p = 2, try p = 1 for high-dimensional data, find the best p via cross-validation.

---

### 2.4 Critical Role of Feature Scaling

Without scaling, features with larger numerical ranges completely dominate distance calculations:

**Example — unscaled:**
$$d = \sqrt{(2000-500)^2 + (3-1)^2} = \sqrt{2{,}250{,}000 + 4} \approx 1500$$

The *size* feature dominates. Bedrooms contribute essentially nothing.

> **Feature scaling is not optional for KNN — it is a requirement.**

Apply **Z-score normalization** before computing any distances:
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Always compute $\mu$ and $\sigma$ from the **training set only**, then apply those same values to the new point.

---

## 3. Choosing K

K controls the complexity of the model. It is not learned — you choose it. It represents the fundamental **bias-variance tradeoff** in KNN.

### K Too Small — Overfitting (High Variance)

With **K = 1**, every new point is classified by its single nearest neighbor. The decision boundary contorts to perfectly surround every training example, memorizing noise and mislabeled points.

```
Decision boundary with K=1:
  |  ○ [○] ○
  |     ○  [×]
  |  [×]  ×           [ ] = region around each point
  |     [×]               = extremely jagged boundary
```

### K Too Large — Underfitting (High Bias)

With **K = m** (all training examples), every new point is classified by a vote of the entire training data. The prediction is always the majority class — regardless of where the new point sits. All local structure is ignored.

### The Sweet Spot — Bias-Variance Tradeoff

```
Error
  |
  |  *                     ← K=1: high variance (overfitting)
  |    *
  |      *   optimal K
  |        * ↓
  |          *   *
  |                *  *   ← large K: high bias (underfitting)
  +------------------------→ K
  1   3   5   7   9   11 ...
```

| K | Bias | Variance | Behaviour |
|---|---|---|---|
| Very small (1–3) | Low | High | Overfits — noise sensitive |
| Moderate (optimal) | Balanced | Balanced | Generalizes well |
| Very large | High | Low | Underfits — over-smoothed |

### Practical Rules of Thumb

- **Always use odd K** for binary classification — avoids ties in voting.
- A common starting point is $K = \sqrt{m}$ where m is training set size.
- **Always validate** — never guess K without data evidence.

---

## 4. Curse of Dimensionality

### The Core Problem

The entire logic of KNN depends on one assumption: **nearby points are similar**. In low dimensions this holds beautifully. As the number of features n grows large, this assumption **collapses completely**.

### What Happens to Space in High Dimensions

To capture 10% of data, the side length of the neighborhood hypercube must be:

$$\text{Side length needed} = (0.1)^{1/n}$$

| Dimensions n | Side length to capture 10% of data |
|---|---|
| 1 | 0.10 |
| 2 | 0.32 |
| 5 | 0.63 |
| 10 | 0.79 |
| 100 | 0.98 |

> In 100 dimensions, you must cover **98% of the range of every feature** just to capture 10% of the data. Your "nearest neighbors" are not near at all.

### All Points Become Equidistant

In high dimensions, the ratio of maximum to minimum distance between points converges to 1:

$$\lim_{n \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

The farthest point is barely farther than the nearest point. The concept of "nearest neighbor" loses all meaning.

### Practical Consequences

- You need **exponentially more training data** to maintain the same density of neighbors: $m \sim O(c^n)$
- Distance calculations become **meaningless** — all distances cluster around the same value
- KNN accuracy degrades rapidly beyond roughly **10–20 dimensions**
- Real-world datasets (images, text, genomics) have thousands to millions of dimensions — KNN is essentially unusable **without dimensionality reduction first**

### How to Fight the Curse

| Technique | What It Does |
|---|---|
| PCA | Reduces dimensions while preserving variance |
| Feature Selection | Remove irrelevant features before applying KNN |
| t-SNE / UMAP | Map data to lower-dimensional space |
| Neural Networks | Learn compact, low-dimensional representations internally |

> This is one of the core reasons Neural Networks dominate in high-dimensional settings like images and text — they learn compact representations of the data internally.

---

## 5. Lazy Learner

### Eager vs. Lazy Learners

| Type | Also Called | Examples |
|---|---|---|
| Eager Learner | Model-based | Linear Regression, Neural Networks, Decision Trees |
| Lazy Learner | Instance-based | KNN, Locally Weighted Regression |

**Eager learners** go through an explicit training phase — they process data, fit parameters, and build a compact model. Prediction is then fast and cheap.

**KNN does absolutely nothing during training.** It simply stores the entire training dataset and waits. All computation is deferred to prediction time.

```
Eager Learner (e.g., Linear Regression):
  Training:   ████████████  (heavy — fits w*, b*)
  Prediction: □             (trivial — just compute wx+b)

KNN:
  Training:   □             (nothing — just store data)
  Prediction: ████████████  (heavy — find K nearest neighbors)
```

### Advantages of Being Lazy

| Advantage | Explanation |
|---|---|
| Zero training time | Immediately usable once data is available |
| Naturally adapts to new data | Append a new example — model instantly "knows" about it, no retraining |
| Non-parametric flexibility | Makes no assumptions about decision boundary shape; adapts to any pattern |

### Disadvantages of Being Lazy

| Disadvantage | Explanation |
|---|---|
| Slow prediction | Must compute distance to every training point: $O(m \cdot n)$ per prediction |
| Memory heavy | Entire training dataset must be stored permanently |
| No model insight | No weights, no feature importances — black box by nature |

For m = 1,000,000 and n = 100: that's **100 million distance computations per prediction**.

### Speeding Up KNN — Data Structures

| Structure | Cost | Notes |
|---|---|---|
| Brute Force | $O(m \cdot n)$ | Default, impractical at scale |
| KD-Tree | $O(n \log m)$ | Good in low dimensions; breaks down in high dimensions |
| Ball Tree | Better than KD-Tree for higher dims | Still struggles beyond ~20 dimensions |
| ANN (FAISS, HNSW, ScaNN) | Near-constant | Sacrifice tiny accuracy for massive speed; power modern vector databases |

---

## 6. Worked Example — Binary Classification

**Task:** Predict if a tumor is Malignant or Benign. 6 training examples, 2 features.

| Point | Tumor Size x₁ | Age x₂ | Class y |
|---|---|---|---|
| A | 2 | 30 | Benign (0) |
| B | 3 | 40 | Benign (0) |
| C | 4 | 50 | Benign (0) |
| D | 7 | 60 | Malignant (1) |
| E | 8 | 65 | Malignant (1) |
| F | 9 | 70 | Malignant (1) |

**New patient (?):** Tumor Size = 5, Age = 45. Solve for **K = 3**.

---

### Step 1 — Z-score Normalization

Compute μ and σ **from training set only**, then apply to new point with the same values.

**Tumor Size x₁:** μ₁ = 5.5, σ₁ ≈ 2.63

**Age x₂:** μ₂ = 52.5, σ₂ ≈ 14.07

**Scaled training data:**

| Point | x₁ scaled | x₂ scaled | Class |
|---|---|---|---|
| A | −1.33 | −1.60 | Benign |
| B | −0.95 | −0.89 | Benign |
| C | −0.57 | −0.18 | Benign |
| D | +0.57 | +0.53 | Malignant |
| E | +0.95 | +0.89 | Malignant |
| F | +1.33 | +1.24 | Malignant |

**New point ? scaled:** (−0.19, −0.53)

---

### Step 2 — Compute Euclidean Distance from ? to Every Training Point

$$d(\vec{x}^{?}, \vec{x}^{(i)}) = \sqrt{(x_1^{?} - x_1^{(i)})^2 + (x_2^{?} - x_2^{(i)})^2}$$

| Point | Calculation | Distance |
|---|---|---|
| A | $\sqrt{(1.14)^2 + (1.07)^2}$ | **1.563** |
| B | $\sqrt{(0.76)^2 + (0.36)^2}$ | **0.841** |
| C | $\sqrt{(0.38)^2 + (0.35)^2}$ | **0.516** |
| D | $\sqrt{(0.76)^2 + (1.06)^2}$ | **1.304** |
| E | $\sqrt{(1.14)^2 + (1.42)^2}$ | **1.821** |
| F | $\sqrt{(1.52)^2 + (1.77)^2}$ | **2.333** |

---

### Step 3 — Rank and Select K = 3 Nearest

| Rank | Point | Distance | Class |
|---|---|---|---|
| 1st | C | 0.516 | Benign (0) |
| 2nd | B | 0.841 | Benign (0) |
| 3rd | D | 1.304 | Malignant (1) |
| 4th | A | 1.563 | Benign (0) |
| 5th | E | 1.821 | Malignant (1) |
| 6th | F | 2.333 | Malignant (1) |

---

### Step 4 — Majority Vote

```
Neighbor C  →  Benign     (0)  ✓
Neighbor B  →  Benign     (0)  ✓
Neighbor D  →  Malignant  (1)

Vote: Benign = 2  ←  MAJORITY
      Malignant = 1
```

$$\boxed{\text{Prediction: Benign (0)}}$$

---

### Step 5 — Visualize

```
Age (scaled)
  |
  |  A(-1.33,-1.60)
  |
  |        B(-0.95,-0.89)
  |
  |               C(-0.57,-0.18)
  |                    ★ ?(-0.19,-0.53)  ← New Point
  |                           D(+0.57,+0.53)
  |
  |                                 E(+0.95,+0.89)
  |
  |                                        F(+1.33,+1.24)
  +------------------------------------------→ Size (scaled)

○ = Benign (A, B, C)    × = Malignant (D, E, F)    ★ = New patient
K=3 circle around ★ captures: C, B, D → 2 Benign, 1 Malignant → Predict Benign
```

### Sensitivity to K

| K | Neighbors | Vote | Prediction |
|---|---|---|---|
| K=1 | C | Benign (1) | Benign |
| K=3 | C, B, D | Benign (2) vs Malignant (1) | **Benign** |
| K=5 | C, B, D, A, E | Benign (3) vs Malignant (2) | Benign |

The decision is stable across K values here — which gives confidence in the prediction.

---

## 7. Distance Metrics — Worked Comparisons

Using the same scaled values from the worked example above.

### Manhattan Distance (p = 1)

$$d_{\text{Manhattan}} = \sum_{j=1}^{n} |x_j^{?} - x_j^{(i)}|}$$

No squaring, no square root — just the sum of absolute differences.

| Point | Calculation | Distance |
|---|---|---|
| A | 1.14 + 1.07 | **2.21** |
| B | 0.76 + 0.36 | **1.12** |
| C | 0.38 + 0.35 | **0.73** |
| D | 0.76 + 1.06 | **1.82** |
| E | 1.14 + 1.42 | **2.56** |
| F | 1.52 + 1.77 | **3.29** |

K=3 neighbors: C, B, D → **Predict: Benign (0)**

---

### Minkowski Distance (p = 3)

$$d_{\text{Minkowski}} = \left(\sum_{j=1}^{n} |x_j^{?} - x_j^{(i)}|^3\right)^{1/3}$$

| Point | Calculation | Distance |
|---|---|---|
| A | $(1.14^3 + 1.07^3)^{1/3} = (1.482 + 1.225)^{1/3}$ | **1.392** |
| B | $(0.76^3 + 0.36^3)^{1/3} = (0.439 + 0.047)^{1/3}$ | **0.787** |
| C | $(0.38^3 + 0.35^3)^{1/3} = (0.055 + 0.043)^{1/3}$ | **0.461** |
| D | $(0.76^3 + 1.06^3)^{1/3} = (0.439 + 1.191)^{1/3}$ | **1.175** |
| E | $(1.14^3 + 1.42^3)^{1/3} = (1.482 + 2.863)^{1/3}$ | **1.632** |
| F | $(1.52^3 + 1.77^3)^{1/3} = (3.512 + 5.545)^{1/3}$ | **2.082** |

K=3 neighbors: C, B, D → **Predict: Benign (0)**

---

### Full Comparison Across All Three Metrics

| Point | Euclidean (p=2) | Manhattan (p=1) | Minkowski (p=3) | Class |
|---|---|---|---|---|
| C | 0.516 | 0.73 | 0.461 | Benign |
| B | 0.841 | 1.12 | 0.787 | Benign |
| D | 1.304 | 1.82 | 1.175 | Malignant |
| A | 1.563 | 2.21 | 1.392 | Benign |
| E | 1.821 | 2.56 | 1.632 | Malignant |
| F | 2.333 | 3.29 | 2.082 | Malignant |

All three metrics produce the **exact same ranking** in this example → all predict **Benign**.

---

### When Rankings Diverge — Key Insight

Rankings diverge when dimensions have very **unequal differences**. Consider two candidate neighbors for a new point:

- **Point X:** differences = (2.0, 0.1) — large gap on dim 1, tiny on dim 2
- **Point Y:** differences = (1.1, 1.1) — moderate gap on both dims

| Metric | Distance to X | Distance to Y | Closer Point |
|---|---|---|---|
| Manhattan | 2.0 + 0.1 = **2.1** | 1.1 + 1.1 = **2.2** | X |
| Euclidean | $\sqrt{4.01} = 2.002$ | $\sqrt{2.42} = 1.556$ | Y |
| Minkowski (p=3) | $(8.0+0.001)^{1/3} = 2.000$ | $(2.662)^{1/3} = 1.386$ | Y |

**Different metrics → different neighbors → potentially different predictions.**

### What Each Metric Really Does

```
Manhattan  →  "Total travel distance regardless of direction"
               Large difference on ONE axis is OK
               Spreads penalty evenly across all dimensions

Euclidean  →  "Straight-line distance"
               Squares amplify large differences
               Penalizes being far on any axis

Minkowski  →  As p grows, increasingly dominated by the
  (p → ∞)     single largest dimensional difference —
               the worst-case axis matters more and more
```

---

## 8. Choosing K — All Methods

### Method 1 — Rule of Thumb: K = √m

$$K = \sqrt{m} \quad \text{(round to nearest odd number)}$$

For m = 6: $K = \sqrt{6} \approx 2.449 \approx \mathbf{3}$

Always prefer **odd K** for binary classification to avoid tie votes.

| Advantage | Disadvantage |
|---|---|
| Instant — no computation needed | Completely ignores data structure |
| Good starting point | May be far from optimal |

**Reliability: Very Low.** Use only as a starting point, never as a final answer.

---

### Method 2 — Train/Test Split Evaluation

Split data into training and test sets. Try multiple K values, evaluate each on test set, pick K with lowest error.

**Key weakness:** Result depends heavily on which points end up in test set.

**Reliability: Low to Medium.**

---

### Method 3 — Cross-Validation (The Standard Method)

Instead of one fixed split, **rotate the test set** across the entire dataset multiple times and average the results. Every point gets to be a test point at least once.

**K-Fold CV Process:**
```
Split data into F equal folds
        ↓
For each fold (1 to F):
    Use that fold as test set
    Train on remaining F-1 folds
    Record error
        ↓
Average error across all F rounds = CV Error
```

> ⚠️ *Cross-validation and Grid Search will be covered properly in Andrew Ng's Course 2-Neural Networks. For now, we will learn about the rule of thumb and elbow method.*

---

### Method 4 — Elbow Method (Visual)

Plot K on the x-axis against validation error on the y-axis. Look for the **"elbow"** — the point where error stops decreasing significantly.

```
Error
  |
  |*  ← K=1: high variance
  | *
  |  *  ← elbow here
  |   * *
  |       * *  ← K too large: underfitting
  +-------------------→ K
  1  3  5  7  9  11  13
```

Best used **alongside** cross-validation, not instead of it.

**Reliability: Medium.**

---

### Method 5 — Grid Search with Cross-Validation (Most Reliable)

Systematically search over all candidate K values, evaluate each using cross-validation, select the K with minimum cross-validated error.

```
Define search space: K ∈ {1, 3, 5, 7, 9, 11, 13, ...}
        ↓
For each K:
    Run F-fold Cross-Validation
    Compute average CV error
        ↓
Pick K* = argmin(CV Error)
        ↓
Retrain final model on ALL data using K*
```

**Reliability: Highest.** This is the production standard.

---

### Complete Method Comparison

| Method | How It Works | Reliability | Cost | Use When |
|---|---|---|---|---|
| K = √m | Formula — no computation | Very Low | Zero | Quick first guess only |
| Train/Test Split | One split, check error | Low | Very Low | Tiny datasets, quick check |
| K-Fold CV | Rotate test folds, average error | High | Medium | Standard choice |
| LOOCV | Every point tested once | High | High | Small datasets |
| Elbow Method | Visual inspection of error curve | Medium | Low | Confirming CV result visually |
| **Grid Search + CV** | **Exhaustive search over K + CV** | **Highest** | **Medium** | **Always — production standard** |

### The Reliable Final Workflow

```
Step 1: Start with K = √m as a rough guess
              ↓
Step 2: Run Grid Search over K = 1, 3, 5, ..., √(2m)
        with 5-fold or 10-fold Cross-Validation
              ↓
Step 3: Plot elbow curve to visually confirm
              ↓
Step 4: Pick K* = argmin(CV Error)
        — always odd for binary classification
              ↓
Step 5: Retrain final model on ENTIRE training set using K*
```

---

## 9. Complete Summary

### The Full KNN Pipeline

```
New data point arrives
        ↓
Compute distance to ALL training examples
(Euclidean / Manhattan / Minkowski)
        ↓
Sort by distance, pick K nearest neighbors
        ↓
Classification → Majority Vote  → predicted class
Regression     → Average value  → predicted number
        ↓
Output prediction
```
