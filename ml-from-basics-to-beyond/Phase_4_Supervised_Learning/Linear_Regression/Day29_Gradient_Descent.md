# 📅 Day 29 – Gradient Descent (In Depth)

**📌 Phase 4: Supervised Learning**
**📖 Reference:** Andrew Ng's Machine Learning Specialization, Course 1 — Supervised Machine Learning

---

## 1. Core Intuition

Imagine you are standing on a **hilly landscape** and you want to get to the **lowest valley**. You look around in all directions, find the steepest downhill direction, and take a small step that way. Then you repeat — look around, find the steepest direction, take a step — until you reach the bottom.

**That's exactly what Gradient Descent does** — but instead of a physical landscape, you are navigating the **cost function surface J(w, b)**, and instead of physical steps, you are updating the values of `w` and `b`.

> The **"lowest valley"** is the **minimum of J(w, b)** — the best possible parameters for your model.

---

## 2. The Algorithm — Formal Statement

Gradient Descent **simultaneously updates** `w` and `b` using these rules:

$$w = w - \alpha \frac{\partial J(w,b)}{\partial w}$$

$$b = b - \alpha \frac{\partial J(w,b)}{\partial b}$$

**Repeat these updates until convergence** — meaning J(w, b) stops decreasing significantly.

Every symbol here carries meaning:

### i) The `=` Symbol
This means **assignment** — you are replacing the old value of `w` with a new computed value.

### ii) The Learning Rate `α` (Alpha)
- `α` (alpha) is a **small positive number**, typically between `0.001` and `0.1`
- It **controls how big a step** you take at each iteration
- This is one of the **most critical hyperparameters** in all of Machine Learning

### iii) The Partial Derivative `∂J(w,b)/∂w`
This is the **gradient** — the slope of the cost function with respect to `w` at the current point. It tells you:
- **The direction to move** (sign of the derivative)
- **The steepness of the slope** (magnitude of the derivative)

---

## 3. Why the Minus Sign? — Simple Example

Let's use the simplest possible case. Fix `b = 0` and say your cost function is:

```
J(w) = (w - 3)²
```

This is a parabola. You can immediately see the **minimum is at w = 3** because that's where `J(w) = 0`.

```
J(w)
 |
 |  x               x
 |    x           x
 |      x       x
 |        x   x
 |          x        ← minimum at w = 3, J = 0
 +--------------------→ w
   0  1  2  3  4  5
```

> The learning algorithm **doesn't know** the minimum is at w = 3. It has to find it by itself using the gradient.

**The derivative (slope) of the Cost Function:**

```
∂J/∂w = 2(w - 3)
```

This tells us, at any given `w`, how steeply J is rising or falling.

**The update rule:** `W = W - α · ∂J/∂w`, let's use `α = 0.1`

---

### Scenario 1: Starting to the RIGHT of minimum (w = 5)

```
∂J/∂w = 2(5 - 3) = +4
```

The slope is **positive** → the cost function is rising as `w` increases → you are on the **right side of the bowl**, going uphill to the right.

**Apply update:**
```
w = 5 - 0.1 × (+4) = 5 - 0.4 = 4.6
```

`w` decreased from 5 to 4.6 → **moved LEFT towards minimum** ✅
```
J(4.6) = (4.6 - 3)² = 2.56  → cost went DOWN ✅
```

---

### Scenario 2: Starting to the LEFT of minimum (w = 1)

```
∂J/∂w = 2(1 - 3) = 2(-2) = -4
```

The slope is **negative** → the cost function is falling as `w` increases → you are on the **left side of the bowl**, going uphill to the left.

**Apply update:**
```
w = 1 - 0.1 × (-4) = 1 + 0.4 = 1.4
```

`w` increased from 1 to 1.4 → **moved RIGHT towards minimum** ✅
```
J(1.4) = (1.4 - 3)² = 2.56
J(1)   = (1 - 3)²   = 4     →  4 → 2.56, cost went DOWN ✅
```

> **The minus sign is the reason Gradient Descent always moves toward the minimum**, regardless of which side you start on.

---

## 4. The Learning Rate `α` — Cases

### ⚠️ You must always update `w` and `b` simultaneously:

```
temp_w = w - α × ∂J/∂w
temp_b = b - α × ∂J/∂b
w = temp_w
b = temp_b
```

You must use the **old values** of both `w` and `b` to compute both gradients before updating either.

---

### Case 1: α Too Small

```
J(w,b)
  ↑
  |  w→→→→→→→→→→→→→→ minimum
```

- Gradient Descent takes **very tiny steps**
- Takes many thousands of iterations to converge
- **Computationally expensive** but is *extremely* slow

---

### Case 2: α Too Large

```
J(w,b)
  ↑
  |  * ← start
  |       * (overshoots)
  |   * (oscillates back)
  |       * (overshoots again)
```

- Each step **overshoots the minimum**
- Cost may **oscillate** or even **increase**
- In worst case: **Gradient Descent diverges** — J grows indefinitely ✗

---

### Case 3: α Just Right ✅

- Steps are **large enough** to converge efficiently
- **Small enough** to not overshoot
- Cost **smoothly decreases** each iteration

---

### 🔑 One-Line Intuition — Why the Minus Sign Always Works

The slope **always points uphill**. The minus sign ensures you always go in the **opposite direction** — downhill. That's the entire secret of Gradient Descent.

> You must update `w` and `b` simultaneously — using the old values of both to compute both. That's the correct way.

---

### What Happens as You Approach the Minimum?

As you get closer to the minimum, the **slope `∂J/∂w` naturally gets smaller**. This means the update size `α · ∂J/∂w` also gets smaller — the step size **automatically shrinks** even with a fixed `α`. This is why Gradient Descent can converge without needing to reduce `α` manually.

---

## 5. Gradient Descent for Linear Regression — Differentiation

We know our cost function:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

We need to compute the partial derivatives to plug into the update rules.

**Differentiating J(w,b) w.r.t. `w`:**

$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) \cdot x^{(i)}$$

**Differentiating J(w,b) w.r.t. `b`:**

$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)})$$

---

## 6. The Full Gradient Descent Algorithm for Linear Regression

$$\boxed{w = w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)}) \cdot x^{(i)}}$$

$$\boxed{b = b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)})}$$

Repeat until convergence.

---

## 7. Batch Gradient Descent

This specific variant is called **Batch Gradient Descent** because at every single step, it uses the **average over all `m` training examples** to update `w` and `b`.

> The summation `Σᵢ₌₁ᵐ` is literally saying — *that summation is the update* `i = 1` to `m` for convergence.

**Why "Batch"?** Because at every iteration, it looks at the **entire batch** of training examples — not just one, not just a few — all of them.

---

## 8. The Big Picture — Cost Function vs Gradient Descent

| | Role |
|---|---|
| **Cost Function J(w, b)** | The **objective** — what you want to minimize |
| **Gradient Descent** | The **mechanism** — how you actually minimize it |

> Think of it this way: the cost function tells you *how wrong* you are. Gradient Descent tells you *how to fix it*.

---

## 9. Assumptions Behind Linear Regression

Linear regression works best when these assumptions roughly hold:

- **Linearity** — relationship between inputs and output is linear
- **Independence** — observations are independent
- **Homoscedasticity** — constant variance of errors
- **Normality of errors** — important for inference, less for prediction
- **No strong multicollinearity** among features

> Violating these doesn't always break the model, but performance and reliability may suffer.

---

## 10. Strengths and Limitations of Linear Regression

| Strengths | Limitations |
|-----------|-------------|
| Simple and fast | Cannot model complex non-linear patterns |
| Highly interpretable | Sensitive to outliers |
| Works well with small datasets | Assumes linear relationships |
| Strong baseline for regression problems | Performance degrades with correlated features |

---

## 11. Where Linear Regression Fits in ML

Linear Regression is often:

- The **first model** tried in any regression task
- A **benchmark** for comparing advanced models
- A **core building block** for methods like Regularized Regression (Ridge, Lasso) and Generalized Linear Models

---