# 📅 Day 30 – Multiple Linear Regression (In Depth)

**📌 Phase 4: Supervised Learning**
**📖 Reference:** Andrew Ng's ML Specialization, Course 1 — Week 2: Regression with Multiple Input Variables

---

## 1. Core Intuition

In Week 1, you predicted house prices using **only one feature** — the size of the house. But in reality, a house price depends on many things:

- Size (sq. ft.)
- Number of bedrooms
- Number of floors
- Age of the house

Andrew Ng's key question:

> *"What if we have not just one, but multiple input features? Can we extend Linear Regression to handle this?"*

The answer is **Multiple Linear Regression** — the same idea as before, but now your model takes **multiple features** as input instead of just one.

---

## 2. New Notation — Multiple Features

| Symbol | Meaning |
|--------|---------|
| $n$ | Total number of features |
| $x_j$ | The $j$-th feature (e.g., $x_1$ = size, $x_2$ = bedrooms) |
| $x^{(i)}$ | All features of the $i$-th training example (a **row vector**) |
| $x^{(i)}_j$ | Feature $j$ of training example $i$ |
| $\vec{x}^{(i)}$ | Feature vector of the $i$-th training example |

### Example — House Price Dataset

| | $x_1$ (size) | $x_2$ (bedrooms) | $x_3$ (floors) | $x_4$ (age) | $y$ (price) |
|--|:--:|:--:|:--:|:--:|:--:|
| House 1 | 2104 | 5 | 1 | 45 | 460 |
| House 2 | 1416 | 3 | 2 | 40 | 232 |
| House 3 | 852  | 2 | 1 | 35 | 178 |

Here $n = 4$ features.

- $x^{(2)}$ = entire second row = $[1416, 3, 2, 40]$ — all features for house 2
- $x^{(2)}_3 = 2$ — feature 3 (floors) of house 2

---

## 3. The Model — From One Feature to Many

**Before (Simple Linear Regression — 1 feature):**
$$f_{w,b}(x) = wx + b$$

**Now (Multiple Linear Regression — $n$ features):**
$$f_{w,b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n + b$$

Each feature $x_j$ gets its own **weight** $w_j$ that controls how much that feature contributes to the prediction.

### Concrete Interpretation (Andrew Ng's House Price Example)

$$\hat{y} = 0.1x_1 + 4x_2 + 10x_3 - 2x_4 + 80$$

| Weight | Feature | Meaning |
|--------|---------|---------|
| $0.1$ | Size (sq ft) | Every extra sq. ft. adds $0.1k |
| $4$ | Bedrooms | Every extra bedroom adds $4k |
| $10$ | Floors | Every extra floor adds $10k |
| $-2$ | Age | Every extra year subtracts $2k |
| $80$ | Bias $b$ | Base price starts at $80k |

This is how the model **encodes real-world knowledge** through its weights.

---

## 4. Vector Notation — The Compact Form

Define:

$$\vec{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}, \qquad \vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

The model becomes:

$$\boxed{f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b}$$

Where the **dot product** is:

$$\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \cdots + w_nx_n = \sum_{j=1}^{n} w_jx_j$$

Andrew Ng emphasizes this notation because:
- It's cleaner mathematically
- It maps directly to **NumPy code**
- It generalizes naturally to **Neural Networks**

---

## 5. Vectorization — How This Is Actually Computed

> This is a point many beginners miss — and Andrew Ng stresses it strongly.

### Without Vectorization (Slow — Loop approach)

```python
f = 0
for j in range(n):
    f = f + w[j] * x[j]
f = f + b
```

### With Vectorization (Fast — NumPy approach)

```python
f = np.dot(w, x) + b
```

**Why vectorization is fundamentally faster:**

```
Without vectorization:   w1*x1 → w2*x2 → w3*x3 → ...   (sequential)

With vectorization:      w1*x1
                         w2*x2   ← all computed simultaneously
                         w3*x3
                         ...
                         then summed in one operation
```

- NumPy's `dot()` uses **BLAS libraries** optimized at the hardware level
- Modern CPUs and GPUs perform these operations **in parallel**

> When you have millions of examples and hundreds of features, the difference between looped and vectorized code can be **hours vs seconds**. Vectorization is not optional in practice.

---

## 6. The Cost Function for Multiple Linear Regression

Same structure — just using vectors now:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)^2$$

Where $f_{\vec{w},b}(\vec{x}^{(i)}) = \vec{w} \cdot \vec{x}^{(i)} + b$

The goal remains the same:

$$\min_{\vec{w},\, b} \; J(\vec{w}, b)$$

Now instead of finding the best single $w$, you're finding the best **entire vector** $\vec{w} = [w_1, w_2, \ldots, w_n]$ and $b$.

---

## 7. Gradient Descent for Multiple Linear Regression

Gradient Descent extends to multiple features by updating **each weight separately**. Repeat until convergence:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \quad \text{for } j = 1, 2, \ldots, n$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

**Partial derivatives:**

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

### Comparing Simple LR vs Multiple LR

| | Simple LR | Multiple LR |
|--|-----------|-------------|
| Parameters | $w$, $b$ | $w_1, w_2, \ldots, w_n$, $b$ |
| Updates per step | One $w$ update | $n$ separate $w_j$ updates |
| Derivative | Uses $x^{(i)}$ | Uses $x_j^{(i)}$ (feature $j$ only) |
| Structure | Same | Same — just repeated for each $w_j$ |

The structure is **identical** — just repeated for each feature.

---

## 8. An Alternative to Gradient Descent — Normal Equation

Andrew Ng briefly introduces the **Normal Equation** as a closed-form alternative for Linear Regression:

$$\vec{w} = (X^TX)^{-1}X^T\vec{y}$$

This directly computes the optimal $\vec{w}$ and $b$ in one mathematical operation — **no iteration needed**.

| | Normal Equation | Gradient Descent |
|--|----------------|-----------------|
| Iteration needed | No — one-shot solution | Yes — many iterations |
| Works for large $n$ | Slow — $O(n^3)$ complexity | Fast — scales well |
| Works beyond Linear Regression | ❌ No | ✅ Yes — universal |
| Used in practice | Rarely | Almost always |

> Andrew Ng's conclusion: Gradient Descent is the method you must understand deeply — it scales to Neural Networks and beyond.

---