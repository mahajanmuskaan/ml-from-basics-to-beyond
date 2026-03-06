# 📅 Day 32 – Convergence, Learning Rate, Feature Engineering & Polynomial Regression

**📌 Phase 4: Supervised Learning**
**📖 Reference:** Andrew Ng's ML Specialization, Course 1 — Week 2

---

## PART 1 — Checking Gradient Descent Convergence

### Core Intuition

How do you know when Gradient Descent is done? How do you know it's actually working correctly?

Andrew Ng's recommended approach is to plot the **Learning Curve** — a graph of the cost $J(\vec{w}, b)$ against the number of iterations of Gradient Descent.

### What a Healthy Learning Curve Looks Like

```
J(w,b)
  |
  |*
  | *
  |  *
  |   **
  |     ***
  |        *****
  |             ******** ← flattens = Converged
  +----------------------------→
  0  100 200 300 400 500
        no. of iterations
```

- $J$ **decreases after every single iteration**
- The **rate of decrease slows down** over time
- Eventually the curve **flattens out** — this is **Convergence**

### What the Curve Tells You

- If $J$ is going **down** → Gradient Descent is working correctly
- If $J$ **flattens** → you have converged, training can stop
- The number of iterations needed **varies widely** — could be 30, could be 30,000. We can't know in advance. The learning curve tells you.

---

### Automatic Convergence Test

In addition to visual inspection, Andrew Ng introduces a **numerical stopping condition**. Define a small threshold $\varepsilon$ (epsilon):

> If $J(\vec{w}, b)$ decreases by **less than $\varepsilon$ in one iteration** → declare convergence.

A typical choice: $\varepsilon = 10^{-3}$

**Example:**

| Iteration | $J$ |
|:---------:|:---:|
| 100 | 0.541 |
| 101 | 0.540 |

$$\text{Decrease} = 0.001 = \varepsilon \;\Rightarrow\; \textbf{Stop. Converged.}$$

### Limitation of the Automatic Test

Choosing $\varepsilon$ is tricky:
- **Too large** → you stop too early before reaching the true minimum
- **Too small** → you run unnecessarily long

> Visual inspection of the learning curve is generally more reliable.

---

### Diagnosing Problems from the Learning Curve

#### Problem 1 — J is Increasing ↑

```
J(w,b)
|         x
|      x x
|   x x
| x
+----------→ Iterations
```

Cost is going **up** — something is wrong. $\alpha$ is **too large** → Gradient Descent is overshooting the minimum.

**Fix: Reduce $\alpha$**

---

#### Problem 2 — J Oscillates

```
J(w,b)
|   x       x     x
|     x   x   x
|       x
+----------→ Iterations
```

Cost bounces up and down without a clear downward trend.

**Fix: Reduce $\alpha$**

---

#### Problem 3 — J Decreases but Very Slowly

```
J(w,b)
|*
| *
|  *
|   * (barely moving)
|    *
+----------→ Iterations
```

Cost is decreasing correctly but **taking forever**.

**Fix: Increase $\alpha$**

---

## PART 2 — Learning Rate $\alpha$

### Core Intuition

- If $\alpha$ is **too small** → GD will work but will be **very slow**
- If $\alpha$ is **too large** → GD may **overshoot and never converge** — or even diverge
- There is **no single correct $\alpha$ for all problems** — you have to search for it systematically

```
J(w)         α too large — overshoots
  ↑  start
  |  *
  |       * ← jumped here (worse than start)
  |  ← jumped over minimum
  |       * ← minimum (never reached)
  +----------→ w
```

**Examples:**

| Value | Verdict |
|-------|---------|
| $\alpha = 0.0001$ | Too Small |
| $\alpha = 10$ | Too Large |
| $\alpha = 0.01$ | Just Right |

---

### Andrew Ng's Strategy — Try a Range of Values

Try $\alpha$ values across a **logarithmic range** and observe the learning curve for each:

$$0.001 \;\rightarrow\; 0.003 \;\rightarrow\; 0.01 \;\rightarrow\; 0.03 \;\rightarrow\; 0.1 \;\rightarrow\; 0.3 \;\rightarrow\; 1$$

> Each step is roughly **3× the previous one**. This efficiently covers the space.

**Debugging Trick:**

- $\alpha = 0.001$ → $J$ decreases ✅
- $\alpha = 0.01$ → $J$ decreases faster ✅
- $\alpha = 0.001$ → $J$ increases ❌

Pick the **largest $\alpha$ that still shows a consistently decreasing $J$**. This efficiently finds a good value.

> 💡 **Bug in code check:** If cost is consistently increasing even with a very small $\alpha$, there is likely a **bug in the gradient descent formula** (e.g., wrong sign in update, incorrect derivative). Just gradient descent descent itself is consistent — check the gradient formula, simultaneous update, etc.

---

## PART 3 — Feature Engineering

### What is Feature Engineering?

Feature Engineering means **transforming existing features** to create new ones that better capture the underlying pattern in the data.

> You don't need a new algorithm. You don't need a different architecture. You use a **smarter representation of your existing data**.

### Example — House Price Prediction

A model currently uses:
- $x_1$ = frontage (width of land)
- $x_2$ = depth of land

$$f(\vec{x}) = w_1x_1 + w_2x_2 + b$$

The model currently determines land value based on width and depth separately. But what actually determines land value is not just the front or depth — it's the **area**.

So we **engineer a new feature**:

$$x_3 = x_1 \times x_2 = \text{Area}$$

The new model is:

$$f(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + b$$

The model now has access to **area** directly — a feature that is much more meaningful for predicting price.

> Feature Engineering is about **combining existing features** to capture the underlying pattern that raw features alone cannot adequately represent.

---

## PART 4 — Polynomial Regression

### What is Polynomial Regression?

Linear Regression can only fit a **straight line**. But your data doesn't always follow a straight line — it might follow a curve.

If the data looks like $x^2$ or $x^3$, the same linear regression machinery can fit it. You just need to be **smarter about your features**.

### The Model

$$f(\vec{x}) = w_1x + w_2x^2 + b$$

This is a **quadratic** (degree 2) model.

```
J vs x — Quadratic:

     x
   x   x
  x     x
 x       x
x         x
```

Or a **cubic** (degree 3) model:

$$f = w_1x + w_2x^2 + w_3x^3 + b$$

```
J vs x — Cubic (goes up to come back):

            x x
         x
x   x x
  x
```

Or a **more flexible** form:

$$f = w_1x + w_2\sqrt{x} + b$$

```
J vs x — More Flexible:

      x x x
   x
 x
x
```

### Why This is Still Linear Regression

This is a crucial insight — the **same parameters** $w_1, w_2, w_3$ are still linear. The model is:

$$f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2^2 + w_3x_3^3 + b$$

Where $x_1 = x$, $x_2 = x^2$, $x_3 = x^3$ — it is **linear in its parameters**, even though it fits a curve. The **Polynomial Terms** are treated as separate features:

$$\text{Quadratic: } w_1x + w_2x^2 + b$$
$$\text{Cubic: } w_1x + w_2x^2 + w_3x^3 + b$$

### What Polynomial Regression Looks Like End-to-End

```
Raw          Feature          Gradient        Check
Features  →  Engineering   →  Descent      →  Convergence
             (Polynomial)                     → build
                                               Trained
             Feature      →  Scaling          Model
```

### ⚠️ Important Note on Feature Scaling

When you use Polynomial Regression, the scale of features changes **dramatically**:
- $x$ might range from 1 to 1000
- $x^2$ ranges from 1 to 1,000,000
- $x^3$ ranges from 1 to 1,000,000,000

**Feature Scaling becomes even more critical** with Polynomial Regression — always apply Z-score normalization after engineering polynomial features.

### The Right Choice of Polynomial

The choice of polynomial is **guided by the domain** (what makes sense for the problem) and **validated** by how well it fits the data. You should pick the shape that best fits **all** the problem, not just individual segments.

---