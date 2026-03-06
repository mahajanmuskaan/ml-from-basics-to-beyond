# Day 35 — Overfitting & Regularization

---

## 1. The Problem of Overfitting

Overfitting is one of the most critical practical challenges in ML, affecting both **Regression** and **Classification** tasks. A model can fall into one of three situations:

### The Three Situations

| Situation | Also Called | Description |
|---|---|---|
| **Underfitting** | High Bias | Model is too simple; fails to capture the pattern even in training data |
| **Just Right** | Generalization | Fits training data well *and* performs well on unseen data — the goal |
| **Overfitting** | High Variance | Model is too complex; memorizes noise, fails on new data |

```
Underfitting          Just Right            Overfitting
y                     y                     y
|  * *                |  * *                |  * *
|       * *           |       * *           |*      *  *
|           * *       |  ~curve fits~       |   *       *  *
|_____________        |           * *       |               *
+------------→ x      +------------→ x      +------------→ x
```

> **Key Insight:** *"An overfit model has learned the noise in the training data, not the true underlying pattern. It memorizes instead of generalizing."* — Andrew Ng

---

## 2. Overfitting in Classification

In **Logistic Regression**, overfitting manifests as an overly complex, wiggly decision boundary that perfectly separates training points but fails to generalize:

```
Underfit:          Just Right:         Overfit:
   x2                 x2                  x2
   |                  |                   |
   |  ○ ○             |  ○ ○              |  ○ ○
   | /    ×           |  ) ×              | ⌢  ×
   |/   × ×           | )  × ×            |○⌣× ×  ← bizarre boundary
   |  ×               |   ×               |  ×
   +------→ x1        +------→ x1         +------→ x1
```

---

## 3. How to Address Overfitting

Three main approaches:

### Approach 1 — Collect More Training Data
More data provides richer signal, making it harder for the model to simply memorize noise. Usually the most effective fix *when available*.

### Approach 2 — Feature Selection
Use fewer, more meaningful features. Including 100 features when only 10 are relevant gives the model excessive flexibility to overfit on noise.

### Approach 3 — Regularization *(most elegant & widely used)*
Keep all features but **penalize large weights** $w_j$. This constrains the model's flexibility without discarding information.

---

## 4. Regularization — The Core Idea

> *"When a model overfits, the weights $w_j$ tend to become very large. Large weights mean the model is leaning too heavily on certain features. If we penalize large weights in the cost function, we force the model to stay simpler."* — Andrew Ng

### The Regularized Cost Function

$$J(\vec{w},b) = \underbrace{-\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\Big]}_{\text{original log loss}} + \underbrace{\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2}_{\text{regularization term}}$$

The added penalty term $\frac{\lambda}{2m}\sum w_j^2$ is called **L2 Regularization** (or **Ridge Regularization**).

---

### The $\lambda$ Parameter — Regularization Strength

$\lambda$ (lambda) balances **fitting the data** vs. **keeping weights small**:

| $\lambda$ Value | Effect |
|---|---|
| $\lambda = 0$ | No regularization — model can overfit |
| Small (e.g., 0.001) | Mild dampening of weights |
| Large (e.g., 10) | Strong regularization — weights forced near zero → underfits |
| $\lambda = \infty$ | All weights → 0 → model predicts same for all inputs |

> $\lambda$ is a **hyperparameter** — chosen by the practitioner, just like the learning rate $\alpha$.

---

### Why Not Regularize $b$?

The bias term $b$ is conventionally **not regularized** — only the $w_j$ weights are. Regularizing $b$ has negligible practical impact since it is a single scalar, whereas the $w_j$ values collectively define the model's flexibility.

---

## 5. Regularized Gradient Descent

The weight update picks up an extra term from differentiation of the regularization penalty:

$$w_j := w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)x_j^{(i)} + \frac{\lambda}{m}w_j\right]$$

$$b := b - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)$$

### The Shrinkage Interpretation

Factoring $w_j$ out of the update reveals what regularization is doing at *every single step*:

$$w_j := \underbrace{\left(1 - \alpha\frac{\lambda}{m}\right)}_{\text{shrinkage factor}} w_j \;-\; \alpha\frac{1}{m}\sum_{i=1}^{m}\left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)x_j^{(i)}$$

The factor $\left(1 - \alpha\frac{\lambda}{m}\right)$ is slightly less than **1** — it **shrinks $w_j$ a little before every gradient update**. Over many iterations, this prevents weights from growing large, keeping the model well-constrained.

---

## 6. The Flow — Connecting It All

```
Raw Data (Binary Labels: 0 or 1)
          ↓
Why not Linear Regression?
→ Outputs not bounded to [0,1]; sensitive to outliers
          ↓
Logistic Regression
→ f(x) = g(w·x + b)  —  always outputs a probability in (0, 1)
          ↓
Decision Boundary
→ Threshold at 0.5  →  straight line or curve separating classes
          ↓
Cost Function: Log Loss  (not Squared Error)
→ Convex  →  Gradient Descent finds the global minimum
          ↓
Gradient Descent
→ Learn optimal w and b
          ↓
Check for Overfitting
→ Too complex?  →  Add Regularization (λ term)
          ↓
Trained Logistic Regression Model ✓
```

---