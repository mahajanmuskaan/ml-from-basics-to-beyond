# 📅 Day 31 – Feature Scaling (In Depth)

**📌 Phase 4: Supervised Learning**
**📖 Reference:** Andrew Ng's ML Specialization, Course 1 — Week 2: Regression with Multiple Input Variables

---

## 1. Core Intuition

Andrew Ng introduces Feature Scaling by first showing a **problem** that arises with Multiple Linear Regression when features have very different ranges.

Consider predicting house prices with two features:

| Feature | Range |
|---------|-------|
| Size $x_1$ (sq. ft.) | 300 — 2000 |
| Bedrooms $x_2$ | 1 — 5 |

---

## 2. Why Different Feature Ranges Cause Problems

### The Effect on Parameters $w_1$ and $w_2$

Since $x_1$ (size) is large (e.g., 1000), even a small $w_1$ produces a big output → optimal $w_1$ ends up **small** (e.g., 0.1).

Since $x_2$ (bedrooms) is small (e.g., 3), you need a larger $w_2$ to produce a meaningful contribution → optimal $w_2$ ends up **large** (e.g., 50).

$$\hat{y} = \underbrace{0.1}_{\text{small}} \cdot \underbrace{1000}_{\text{large } x_1} + \underbrace{50}_{\text{large}} \cdot \underbrace{3}_{\text{small } x_2} + b$$

### What This Does to the Cost Function

The cost function $J(w_1, w_2)$ becomes an **elongated, oval-shaped** contour plot instead of a nice circular bowl.

```
Without Feature Scaling:          With Feature Scaling:

w2                                 w2
|                                  |
|  (very tall, narrow oval)        |   (circular contours)
|  ___                             |      ___
| /   \                            |     /   \
||     |                           |    |  *  |
| \___/ ← minimum                  |     \___/ ← minimum
|                                  |
+----------→ w1                    +----------→ w1
(w1 range is tiny)                 (both ranges similar)
```

### Why the Oval Shape is a Problem for Gradient Descent

On the elongated oval, Gradient Descent **bounces back and forth** across the narrow direction while making tiny progress along the wide direction — a very long, inefficient path to the minimum.

```
Gradient Descent path without scaling:

w2
|  * ← start
|   ↘
|    ↗  (bouncing)
|   ↘
|    ↗
|     * ← minimum (takes many steps)
+----------→ w1
```

On circular contours, Gradient Descent moves **directly toward the minimum** in a much straighter, faster path.

> *"When you have different features with very different ranges, scaling them to comparable ranges speeds up Gradient Descent significantly."* — Andrew Ng

---

## 3. Three Methods of Feature Scaling

Andrew Ng covers three approaches. We'll apply each to the same dataset:

| House | Size $x_1$ | Bedrooms $x_2$ |
|:-----:|:----------:|:--------------:|
| 1 | 600 | 1 |
| 2 | 1200 | 3 |
| 3 | 1800 | 5 |

---

### Method 1 — Simple Scaling (Divide by Maximum)

Divide every feature value by the **maximum value** of that feature → rescales all values to range $[0, 1]$.

$$x_{j,\text{scaled}} = \frac{x_j}{\max(x_j)}$$

**For Size** ($\max = 1800$):

$$x_1^{(1)} = \frac{600}{1800} = 0.33, \quad x_1^{(2)} = \frac{1200}{1800} = 0.67, \quad x_1^{(3)} = \frac{1800}{1800} = 1.0$$

**For Bedrooms** ($\max = 5$):

$$x_2^{(1)} = \frac{1}{5} = 0.2, \quad x_2^{(2)} = \frac{3}{5} = 0.6, \quad x_2^{(3)} = \frac{5}{5} = 1.0$$

**Result:**

| House | $x_1$ (scaled) | $x_2$ (scaled) |
|:-----:|:--------------:|:--------------:|
| 1 | 0.33 | 0.20 |
| 2 | 0.67 | 0.60 |
| 3 | 1.00 | 1.00 |

Both features now live in range $[0, 1]$. Simple and effective for basic cases.

---

### Method 2 — Mean Normalization

Subtracts the mean to **center the feature around zero**, then divides by the range.

$$x_{j,\text{scaled}} = \frac{x_j - \mu_j}{\max(x_j) - \min(x_j)}$$

**For Size:** $\mu_1 = 1200$, range $= 1800 - 600 = 1200$

$$x_1^{(1)} = \frac{600-1200}{1200} = -0.5, \quad x_1^{(2)} = \frac{1200-1200}{1200} = 0, \quad x_1^{(3)} = \frac{1800-1200}{1200} = +0.5$$

**For Bedrooms:** $\mu_2 = 3$, range $= 5 - 1 = 4$

$$x_2^{(1)} = \frac{1-3}{4} = -0.5, \quad x_2^{(2)} = \frac{3-3}{4} = 0, \quad x_2^{(3)} = \frac{5-3}{4} = +0.5$$

**Result:**

| House | $x_1$ (normalized) | $x_2$ (normalized) |
|:-----:|:-----------------:|:-----------------:|
| 1 | −0.5 | −0.5 |
| 2 | 0.0 | 0.0 |
| 3 | +0.5 | +0.5 |

Both features now centered around **zero** with comparable ranges.

---

### Method 3 — Z-score Normalization ⭐ (Most Important)

The **most widely used** method in practice. Andrew Ng covers this most carefully.

$$\boxed{x_{j,\text{scaled}} = \frac{x_j - \mu_j}{\sigma_j}}$$

**What this does:**
- Shifts the feature so its mean becomes **0**
- Scales it so its standard deviation becomes **1**
- Result: every feature follows approximately a **standard normal distribution**

**For Size:** $\mu_1 = 1200$

$$\sigma_1 = \sqrt{\frac{(600-1200)^2 + (1200-1200)^2 + (1800-1200)^2}{3}} = \sqrt{240000} \approx 489.9$$

$$x_1^{(1)} \approx -1.22, \quad x_1^{(2)} = 0, \quad x_1^{(3)} \approx +1.22$$

**For Bedrooms:** $\mu_2 = 3$, $\sigma_2 \approx 1.63$

$$x_2^{(1)} \approx -1.22, \quad x_2^{(2)} = 0, \quad x_2^{(3)} \approx +1.22$$

**Result:**

| House | $x_1$ (z-scored) | $x_2$ (z-scored) |
|:-----:|:----------------:|:----------------:|
| 1 | −1.22 | −1.22 |
| 2 | 0.00 | 0.00 |
| 3 | +1.22 | +1.22 |

Both features now have **mean = 0** and **standard deviation = 1**. This is the **gold standard** for feature scaling.

---

## 4. Before vs After — The Visual Impact

### Before Scaling
```
Contour Plot of J(w1, w2):

w2
|
50|  ___________
  | /           \
  ||             |
  | \___________/   ← very wide in w1 direction
  |
  +----------------------→ w1
  0.05  0.1  0.15  0.2
```

Gradient Descent **zigzags inefficiently**.

### After Scaling
```
Contour Plot of J(w1, w2):

w2
|
2 |     ___
  |    /   \
  |   |  *  |   ← nearly circular
  |    \___/
  |
  +----------→ w1
  -2    0    2
```

Gradient Descent moves **smoothly and directly** to the minimum.

---

## 5. How Scaling Affects Gradient Descent — Numerically

The gradient for $w_1$ is:

$$\frac{\partial J}{\partial w_1} = \frac{1}{m}\sum(\hat{y} - y) \cdot x_1^{(i)}$$

**Without scaling** — $x_1$ values are 600, 1200, 1800 → gradient gets multiplied by **large values** → gradient becomes **very large** → if $\alpha$ is small enough for $w_1$, it's way too small for $w_2$ (and vice versa).

**With scaling** — $x_1$ values are −1.22, 0, +1.22 → gradient gets multiplied by **small, comparable values** → gradients for all features stay in a **similar range** → one learning rate $\alpha$ works well for **all parameters simultaneously**.

---

## 6. Andrew Ng's Rule of Thumb — Target Range

$$-1 \leq x_j \leq 1 \quad \text{(ideal range)}$$

| Range After Scaling | Verdict |
|---------------------|---------|
| $-1$ to $1$ | ✅ Perfect |
| $-3$ to $3$ | ✅ Acceptable |
| $-0.3$ to $0.3$ | ✅ Fine |
| $0$ to $1000$ | ❌ Too large — rescale |
| $-0.0001$ to $0.0001$ | ❌ Too small — rescale |
| $-100$ to $100$ | ❌ Too large — rescale |

The goal is simply that all features live in a **comparable numerical range** — not too large, not too small.

---

## 7. ⚠️ Critical Practical Rule — Apply Scaling Consistently

> *"Whatever scaling you apply to the training data, you must apply the exact same scaling to any new data at prediction time."* — Andrew Ng

This means you must **save** the $\mu_j$ and $\sigma_j$ values from the training set and reuse them.

### Example

Trained with Z-score normalization using training statistics:
$$x_{1,\text{train}} = \frac{x_1 - 1200}{489.9}$$

New house comes in with size = 1500 sq ft. Scale it using the **same training statistics**:

$$x_{1,\text{new}} = \frac{1500 - 1200}{489.9} = \frac{300}{489.9} \approx 0.61$$

> ❌ **Never** recompute $\mu$ and $\sigma$ on the new data point. Always use training set values — otherwise your model receives inputs in a different scale than it was trained on, and predictions will be wrong.

---

## 8. Does Feature Scaling Change the Final Answer?

**No** — the final predictions your model makes are the same. Feature scaling only affects:
- **How fast** Gradient Descent converges
- **How many iterations** are needed

The optimal model (in terms of predictions) is identical with or without scaling — scaling just gets you there **faster and more reliably**.

---

## 9. Comparison of All Three Methods

| Method | Formula | Result Range | Centers at Zero? | Best For |
|--------|---------|:------------:|:----------------:|----------|
| Simple Scaling | $\frac{x_j}{\max(x_j)}$ | $[0, 1]$ | ❌ No | Quick, simple cases |
| Mean Normalization | $\frac{x_j - \mu_j}{\max - \min}$ | $[-1, 1]$ | ✅ Yes | Bounded distributions |
| Z-score Normalization | $\frac{x_j - \mu_j}{\sigma_j}$ | ~$[-3, 3]$ | ✅ Yes | **Most practical cases** |

> **Z-score normalization is the default recommendation** — it's what you'll see in Neural Networks, scikit-learn, and most ML code in practice.

---
