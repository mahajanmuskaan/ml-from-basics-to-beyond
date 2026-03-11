# Support Vector Machines (SVM) — Notes
> The maximum margin classifier — a mathematically principled answer to the question of which decision boundary is best

---

## Table of Contents
1. [The Core Problem — Why Not Just Any Boundary?](#1-the-core-problem--why-not-just-any-boundary)
2. [Margin Intuition](#2-margin-intuition)
3. [Support Vectors](#3-support-vectors)

---

## 1. The Core Problem — Why Not Just Any Boundary?

### The Starting Frustration

Logistic Regression draws a decision boundary to separate classes. But consider this situation — many lines can perfectly separate the two classes:

```
x2
|
|   ○  ○        /  ← Line 1 (works)
|      ○      /
|      ×    /  ← Line 2 (works too)
|   ×     /
|   ×  /  ← Line 3 (also works)
+------------------→ x1
```

All three lines classify the training data perfectly. But which one is **the best**? Which one will work best on data it has never seen?

Logistic Regression doesn't have a principled answer to this. It finds *a* boundary — not necessarily the *best* one.

### SVM's Fundamental Insight

> *"Among all decision boundaries that correctly separate the classes, the best one is the one that is farthest away from all data points — the one with the **maximum margin**."*

This is not just a heuristic. It is a mathematically principled answer grounded in **statistical learning theory**.

---

## 2. Margin Intuition

### What Is a Margin?

The margin is the **total width of the empty zone** between the decision boundary and the closest data points from each class:

```
x2
|
|  ○  ○  |        |  × ×
|     ○  |  gap   |  ×
|     ○  |←——————→|  × ×
|        |        |
|   class 0   class 1
+----------------------------→ x1

         ←margin width→
```

The decision boundary sits exactly in the **middle** of this gap. The wider the margin, the more confident and generalizable the classifier.

### Why Maximum Margin?

A wider margin means:
- New data points have more room to land on the correct side even if they are slightly different from training data
- The classifier is less sensitive to small perturbations in the data
- The decision is based on **global structure**, not local noise

**Andrew Ng's analogy** (Stanford CS229):
> *"Imagine walking a path between two walls. The safest path is the one that keeps you as far as possible from both walls. That's the maximum margin classifier."*

```
Narrow margin (risky):        Wide margin (safe):

○ |×                          ○    |    ×
○|×                              ○  | ×
○ |  ×                        ○    |    ×
  ↑                                 ↑
boundary                         boundary
(close to points —            (far from all points —
 any noise causes               robust to noise)
 misclassification)
```

### Formal Definition of Margin

For a decision boundary defined by $\vec{w}\cdot\vec{x} + b = 0$, the **signed distance** from any point $\vec{x}^{(i)}$ to the boundary is:

$$\text{distance}^{(i)} = \frac{y^{(i)}(\vec{w}\cdot\vec{x}^{(i)} + b)}{||\vec{w}||}$$

Where $||\vec{w}|| = \sqrt{\sum_j w_j^2}$ is the norm of $\vec{w}$.

The margin $\gamma$ is the **minimum distance** across all training examples:

$$\gamma = \min_{i} \frac{y^{(i)}(\vec{w}\cdot\vec{x}^{(i)} + b)}{||\vec{w}||}$$

### SVM's Optimization Goal

$$\boxed{\max_{\vec{w},\, b} \ \gamma = \max_{\vec{w},\, b} \ \min_{i} \ \frac{y^{(i)}(\vec{w}\cdot\vec{x}^{(i)}+b)}{||\vec{w}||}}$$

**Maximize the minimum distance.** This is fundamentally different from minimizing error — SVM maximizes **confidence**, not just correctness.

| Algorithm | What It Optimizes |
|---|---|
| Logistic Regression | Minimizes cross-entropy loss — finds *a* valid boundary |
| SVM | Maximizes the margin — finds the *uniquely best* boundary |

---

## 3. Support Vectors

### The Critical Insight

After finding the maximum margin boundary, something remarkable is true:

> *"The entire decision boundary is determined by only a tiny subset of the training examples — the ones sitting exactly on the edge of the margin. Remove any other point and the boundary doesn't change. These special points are called **Support Vectors**."*

```
x2
|
|  ○        ══════ ← margin boundary (upper)
|     ○ ←SV ——————  ← DECISION BOUNDARY
|        ×  ══════ ← margin boundary (lower)
|  ○  ×  ×
|        ↑ SV
+----------------------------→ x1

SV = Support Vector — sits exactly on the margin boundary
○ and × not on the margin = irrelevant to the boundary
```

The points labeled SV are the **support vectors** — they literally "support" the margin boundaries like columns holding up a structure.

### Why This Matters

| Property | Implication |
|---|---|
| Only support vectors determine the boundary | The model is **sparse** — most training data is discarded after training |
| Non-support vectors are irrelevant | SVM is **robust to outliers** far from the boundary — they never influence it |
| Removing a non-support vector changes nothing | Very different from LR, where every single point contributes to the gradient |

### Contrast with Logistic Regression

```
Logistic Regression:              SVM:

Every point ●                     Only support vectors ●
contributes to                    determine the boundary.
shaping the boundary.             Everything else is ignored.

●  ●                              ○  ○
  ●   ●  ← all pulling            ● ←SV ——— boundary
●    ●      on the line           ×  ↑SV
  ●   ●                           ○  ×  ×  ← these don't matter
```

In LR, a single outlier far from the boundary still nudges the weights during gradient descent. In SVM, that same outlier has **zero influence** on the final boundary — it is not a support vector.

---


