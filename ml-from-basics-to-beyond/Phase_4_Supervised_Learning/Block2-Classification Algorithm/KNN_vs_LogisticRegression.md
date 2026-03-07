# Classification: Logistic Regression vs. KNN
> A conceptual and mechanical comparison of two fundamentally different classification philosophies

---

## How Classification Works in Logistic Regression

Logistic Regression is a **model-based, eager learner**. It learns a fixed mathematical function from training data during a training phase, then uses that function to classify new points.

**Core idea:** Find a linear decision boundary that separates classes, then use probability to make the final call.

### The 3-Step Prediction Mechanism

**Step 1 — Compute a linear combination of features:**
$$z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b = \vec{w} \cdot \vec{x} + b$$

**Step 2 — Squeeze z through the sigmoid function:**
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Maps any real number to (0, 1) — interpreted as the **probability** of belonging to class 1.

**Step 3 — Apply a threshold to classify:**
$$\text{Predict class 1 if } \hat{y} \geq 0.5, \text{ else class 0}$$

The threshold 0.5 corresponds to the **decision boundary** where $z = 0$, i.e., $\vec{w} \cdot \vec{x} + b = 0$. Everything on one side is class 1, the other is class 0.

### Training
- Gradient descent minimizes log-loss (binary cross-entropy) over all training examples
- Iteratively adjusts $\vec{w}$ and $b$ until convergence
- After training, **original data is discarded** — only $\vec{w}$ and $b$ are kept

### Prediction Cost
Just one cheap arithmetic operation: $\sigma(\vec{w} \cdot \vec{x} + b)$ → $O(n)$

---

## How Classification Works in KNN

KNN is an **instance-based, lazy learner**. It learns nothing during training — stores all data and defers all computation to prediction time.

**Core idea:** A new point's class is determined entirely by the classes of its K nearest neighbors in feature space.

### The 4-Step Prediction Mechanism

**Step 1 — Training phase:** Store all training data. Nothing else.

**Step 2 — Compute distance** from the new point to every training example:
$$d(\vec{x}^{?}, \vec{x}^{(i)}) = \sqrt{\sum_{j=1}^{n}(x_j^{?} - x_j^{(i)})^2}$$

**Step 3 — Find K nearest neighbors** by sorting all distances and picking the top K.

**Step 4 — Majority vote:**
$$\hat{y} = \text{mode of } \{y^{(1)}, y^{(2)}, \ldots, y^{(K)}\}$$

Whichever class appears most among the K neighbors wins.

No probability, no weights, no mathematical function — just **distance and counting**.

### Prediction Cost
Must compute distance to every training point: $O(m \cdot n)$ — slow for large datasets.

---

## Side-by-Side Comparison

| | Logistic Regression | KNN |
|---|---|---|
| **Philosophy** | Learn a global function from data | Memorize data, classify by proximity |
| **Type** | Eager learner | Lazy learner |
| **Training** | Gradient descent over many epochs | None — just store data |
| **Prediction** | Compute $\sigma(\vec{w} \cdot \vec{x} + b)$ | Find K nearest neighbors, majority vote |
| **Decision boundary** | Always **linear** (in feature space) | Any shape — fully adapts to data |
| **Output** | Probability + threshold | Direct class label via vote |
| **Prediction speed** | $O(n)$ — very fast | $O(m \cdot n)$ — slow for large m |
| **Memory** | Only stores $\vec{w}$ and $b$ | Must store entire training dataset |
| **Interpretability** | High — weights reveal feature importance | None — black box |
| **Core assumption** | Classes are linearly separable | Nearby points belong to the same class |
| **High dimensions** | Handles well | Breaks down (curse of dimensionality) |
| **Feature scaling** | Recommended | **Mandatory** |

---

## The Deepest Conceptual Difference

**Logistic Regression asks:**
> *"What global pattern in the data separates the classes?"*

It commits to a decision boundary during training and uses it forever. One fixed line (or hyperplane) for all future predictions.

**KNN asks:**
> *"What do the local neighbors of this specific point look like?"*

It makes no global commitment. Every prediction is a fresh local query — the "model" can look completely different in different regions of the feature space.

---

## When to Use Which

| Situation | Prefer |
|---|---|
| Data is (roughly) linearly separable | Logistic Regression |
| Need fast predictions at scale | Logistic Regression |
| Need interpretable feature importances | Logistic Regression |
| Decision boundary is non-linear | KNN |
| Small dataset, low dimensions | KNN |
| Data has complex local structure | KNN |
| High-dimensional data (images, text) | Neither — use Neural Networks |

---

## Key Tradeoff in One Line

KNN gets **flexibility** (any decision boundary shape) at the cost of **speed, memory, and scalability**.
Logistic Regression gets **speed and interpretability** at the cost of **expressiveness** (linear boundary only).

---
