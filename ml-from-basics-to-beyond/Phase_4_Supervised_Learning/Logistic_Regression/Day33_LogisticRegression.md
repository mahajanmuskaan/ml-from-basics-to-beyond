# Day 33 – Logistic Regression

## 1. Introduction

Logistic Regression is a **supervised learning algorithm** used for **binary classification** problems.

- Output variable: `0` or `1`
- It predicts the **probability** that a given input belongs to class 1.
- Final prediction is based on a threshold (usually 0.5).

---

## 2. Why Not Linear Regression for Classification?

If we use linear regression:
- Output can be less than 0 or greater than 1.
- Not suitable for probability prediction.

To solve this, Logistic Regression:
- Uses a **linear model**
- Passes it through a **Sigmoid (Logistic) function**
- Converts output to range **(0, 1)**

---

## 3. Mathematical Formulation

### Step 1: Linear Score (Logit)

\[
z = w^T x + b
\]

For two features:

\[
z = w_1x_1 + w_2x_2 + b
\]

---

### Step 2: Sigmoid Function

\[
g(z) = \frac{1}{1 + e^{-z}}
\]

- Converts any real number into a value between 0 and 1.
- Output represents probability:

\[
\hat{y} = P(y=1|x)
\]

---

## 4. Key Properties of Sigmoid Function

- Output range: (0,1)
- At \( z = 0 \), output = 0.5
- Smooth and differentiable
- Monotonic increasing
- Symmetric around \( z = 0 \)

If:
- \( g(z) \ge 0.5 \) → Predict class 1
- \( g(z) < 0.5 \) → Predict class 0

---

## 5. Worked Numerical Example

Given:
- \( w = 3 \)
- \( b = -6 \)
- \( x = 3 \)

### Step 1: Compute z

\[
z = 3(3) - 6 = 9 - 6 = 3
\]

### Step 2: Apply Sigmoid

\[
g(3) = \frac{1}{1 + e^{-3}}
\]

\[
g(3) \approx \frac{1}{1 + 0.05}
\]

\[
g(3) \approx 0.95
\]

Prediction:
- Probability = 0.95
- Since > 0.5 → Class 1

---

## 6. Decision Boundary

The **decision boundary** is where:

\[
g(z) = 0.5
\]

Since \( g(0) = 0.5 \),

Decision boundary occurs at:

\[
z = 0
\]

\[
w^T x + b = 0
\]

---

### Example: Linear Decision Boundary (2D)

Given:
- \( w_1 = 1 \)
- \( w_2 = 1 \)
- \( b = -3 \)

\[
x_1 + x_2 - 3 = 0
\]

\[
x_1 + x_2 = 3
\]

This represents a **straight line** in 2D.

- Points where \( w^T x + b > 0 \) → Class 1
- Points where \( w^T x + b < 0 \) → Class 0

---

## 7. Non-Linear Decision Boundary

Logistic Regression can create **non-linear boundaries** using **feature engineering**.

### Polynomial Features Example

If we add:

\[
x_1^2, \; x_2^2
\]

Model becomes:

\[
x_1^2 + x_2^2 - 1 = 0
\]

This represents a **circle of radius 1**.

- Inside circle → Class 1
- Outside circle → Class 0

👉 The more polynomial features added, the more complex the decision boundary becomes.

---
