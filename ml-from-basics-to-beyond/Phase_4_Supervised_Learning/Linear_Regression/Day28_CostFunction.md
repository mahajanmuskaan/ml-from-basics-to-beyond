# 📅 Day 28 – Cost Function (In Depth)

**📌 Phase 4: Supervised Learning**
**📖 Reference:** Andrew Ng's Machine Learning Specialization, Course 1 — Supervised Machine Learning

---

## 1. Core Intuition

Andrew Ng frames the Cost Function with a fundamental question:

> *"Given different values of w and b, how do we know which line fits the training data best?"*

You could have infinitely many lines passing through your data. The **Cost Function is the judge** — it gives you a single number that quantifies how wrong your current model is. The smaller that number, the better your model fits the data.

Think of it as a **feedback signal** to the learning algorithm. Without a cost function, the algorithm has no way of knowing whether it's improving or getting worse.

---

## 2. Building It From Scratch — Andrew Ng's Approach

Andrew Ng builds the cost function step by step, starting from a simplified model.

### Step 1 — Simplify the Model (Fix b = 0)

To build intuition cleanly, Andrew Ng first sets `b = 0`, so the model becomes:

```
f_w(x) = wx
```

This forces the line to pass through the origin. Now the only parameter to tune is `w`.

---

### Step 2 — Measure the Error on One Example

For a single training example `(x⁽ⁱ⁾, y⁽ⁱ⁾)`, the model predicts `ŷ⁽ⁱ⁾ = wx⁽ⁱ⁾`.

The error on that example is:

```
error⁽ⁱ⁾ = ŷ⁽ⁱ⁾ - y⁽ⁱ⁾ = f_w(x⁽ⁱ⁾) - y⁽ⁱ⁾
```

> This can be **positive** (over-predicted) or **negative** (under-predicted).

---

### Step 3 — Aggregate Error Across All Examples

We want to measure total error across all `m` training examples. A naive approach — just summing the errors — **fails** because positive and negative errors cancel each other out, giving a falsely optimistic picture.

So we **square each error** before summing:

```
Total Squared Error = Σᵢ₌₁ᵐ ( f_w(x⁽ⁱ⁾) - y⁽ⁱ⁾ )²
```

---

### Step 4 — Normalize and Define the Cost Function

To make the cost independent of `m`, we divide by `m`. We also include `1/2` for mathematical convenience:

$$\boxed{J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2}$$

> This is the **Squared Error Cost Function**, also called **Mean Squared Error (MSE)** (with the `1/2` factor).

---

## 3. Dissecting Every Part of the Formula

| Component | Role |
|-----------|------|
| `f_{w,b}(x⁽ⁱ⁾) - y⁽ⁱ⁾` | Raw error — difference between prediction and truth |
| `(·)²` | Squaring — removes sign, penalizes large errors more |
| `Σᵢ₌₁ᵐ` | Accumulates error across all training examples |
| `1/m` | Averages the error — prevents cost from growing just because `m` is large |
| `1/2` | Convenience factor — cancels the 2 that appears when differentiating |

---

## 4. Why Squared Error? — The Design Choices Explained

### ❓ Why not Absolute Error `|ŷ - y|`?

Mean Absolute Error (MAE) is intuitive, but the absolute value function is **not differentiable at zero**. This creates problems for Gradient Descent, which requires smooth derivatives everywhere. Squared Error is **smooth and differentiable** at all points.

### ❓ Why not sum raw errors?

Positive and negative errors cancel out. A model that over-predicts by 100 and under-predicts by 100 would appear to have **zero error** — clearly wrong.

### ❓ Why does squaring penalize large errors more?

| Error | Squared Error |
|-------|--------------|
| 2 | 4 |
| 4 | 16 |
| 10 | 100 |

The penalty grows **quadratically**, so the model is strongly incentivized to fix large mistakes first.

### ❓ Why the `1/2` factor?

When you differentiate `J(w, b)` with respect to `w`, the chain rule brings down a factor of `2` from the square. The `1/2` and the `2` **cancel cleanly**, giving a tidy gradient expression. It is purely a notational convenience — it does **not** change which `(w, b)` minimizes the cost.

---

## 5. Visualizing J(w) — The Simplified Case (b = 0)

When `b = 0`, `J(w)` is a **1D Parabola**. Consider a tiny training set:

| x⁽ⁱ⁾ | y⁽ⁱ⁾ |
|:----:|:----:|
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |

**Try w = 1:**
- Predictions: ŷ = 1, 2, 3
- Errors: 0, 0, 0
- `J(1) = 0` ← **perfect fit**

**Try w = 0.5:**
- Predictions: ŷ = 0.5, 1, 1.5
- Errors: -0.5, -1, -1.5
- `J(0.5) = 1/6 [(0.25) + (1) + (2.25)] = 3.5/6 ≈ 0.58`

**Try w = 0:**
- Predictions: all ŷ = 0
- `J(0) = 1/6 [1 + 4 + 9] = 14/6 ≈ 2.33`

Plotting these points traces out a **U-shaped parabola**:

```
J(w)
 |
 |  *               *
 |     *         *
 |        *   *
 |           *        ← minimum at w = 1
 +--------------------→ w
   0  0.5   1   1.5
```

> The goal of the learning algorithm is to **find the bottom of this curve**.

---

## 6. Visualizing J(w, b) — The Full 3D Picture

When both `w` and `b` are free parameters, `J(w, b)` becomes a **surface in 3D space**.

### 🏔 3D Bowl Surface

The cost function forms a **convex bowl shape**. Every point on the surface corresponds to a specific `(w, b)` pair and its associated cost. The very bottom of the bowl is the **global minimum**.

### 🗺 Contour Plot (Andrew Ng's Preferred View)

Each elliptical ring represents all `(w, b)` combinations that produce the **same cost value**. The rings get smaller as you approach the center — the minimum.

```
    b
    |
    |    (  (  (  ·  )  )  )
    |       (  (  ·  )  )
    |          ( min )
    |
    +-------------------→ w
```

| Ring | Cost | Fit |
|------|------|-----|
| Outer rings | High | Poor |
| Inner rings | Low | Better |
| Center point | Minimum | Best |

> Each specific `(w, b)` pair corresponds to **one point on the contour plot** AND **one line on the original data plot**. Andrew Ng shows these side-by-side to build intuition about how the cost landscape relates to the actual line fit.

---

## 7. The Objective — Formally Stated

The entire purpose of training is:

$$\min_{w,\, b} \; J(w, b) = \min_{w,\, b} \; \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

This is an **optimization problem**. You are searching the space of all possible `(w, b)` values to find the pair that gives the **lowest cost**.

---

## 8. Convexity — Why Linear Regression is "Safe"

> This is a property Andrew Ng highlights as critically important.

The Squared Error Cost Function for Linear Regression is **convex**. Formally, a function is convex if the line segment between any two points on the curve lies **above or on the curve**.

### ✅ What Convexity Guarantees

- There is exactly **one global minimum** — no local minima
- **Gradient Descent will always converge** to the global minimum (given an appropriate learning rate)
- You will **never get stuck** in a bad solution

> ⚠️ This is a luxury that **Neural Networks do not have** — their cost functions are non-convex with many local minima. Understanding why Linear Regression is special here sets up an important contrast for later in the course.

