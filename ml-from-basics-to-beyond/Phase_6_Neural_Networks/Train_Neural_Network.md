# Training Neural Networks & Backpropagation
### Andrew Ng's ML Specialization | Course 2, Week 2

---

## Part 1: The Big Picture — What Does "Training" Mean?

### Recall: What We Learned in Week 1

- A neural network is layers of neurons
- Each neuron computes: **z = w·x + b**, then **a = g(z)**
- Forward propagation computes **ŷ** from **X**

### The Question Now

> *"How do we find the right values of W and b — across ALL layers — so that ŷ is close to the true label y?"*

**Answer: Gradient Descent + Backpropagation**

### The Training Loop (Full Picture)

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Initialize W[ℓ], b[ℓ]  ◄─────────────────────────┐       │
│          │                                           │       │
│          ▼                                           │       │
│   Forward Propagation                                │       │
│   (compute ŷ)                                        │       │
│          │                                           │       │
│          ▼                                           │       │
│   Compute Loss J(W,b)                                │       │
│          │                                           │       │
│          ▼                                           │       │
│   Backpropagation                          Repeat    │       │
│   (compute all gradients)                  until     │       │
│          │                                converged  │       │
│          ▼                                           │       │
│   Update Parameters:                                 │       │
│   W = W - α·(∂J/∂W)  ────────────────────────────►──┘       │
│   b = b - α·(∂J/∂b)                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Part 2: The Loss Function & Cost Function

Andrew Ng distinguishes carefully between **loss** and **cost**.

### Loss Function — For ONE Training Example

For **binary classification** (sigmoid output):

```
L(ŷ, y) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

- If **y = 1**: Loss = -log(ŷ) → want ŷ → 1, loss → 0 ✅
- If **y = 0**: Loss = -log(1-ŷ) → want ŷ → 0, loss → 0 ✅

This is called **Binary Cross-Entropy Loss**.

**Intuition:**

```
y=1:  if ŷ=0.99 → loss = -log(0.99) ≈ 0.01  (very small, good!)
      if ŷ=0.01 → loss = -log(0.01) ≈ 4.6   (very large, bad!)

y=0:  if ŷ=0.01 → loss = -log(0.99) ≈ 0.01  (very small, good!)
      if ŷ=0.99 → loss = -log(0.01) ≈ 4.6   (very large, bad!)
```

The loss **heavily penalizes** confident wrong predictions.

### Cost Function — Over ALL m Training Examples

$$J(W,b) = \frac{1}{m} \sum_{i} L(\hat{y}^{(i)}, y^{(i)})$$

This is the **average loss** across all training examples.

> *"The cost function J is the function we want to MINIMIZE by choosing the right W and b. Everything in training is about reducing J."* — Andrew Ng

### For Regression (Linear output)

$$J(W,b) = \frac{1}{2m} \sum_{i} (\hat{y}^{(i)} - y^{(i)})^2$$

This is **Mean Squared Error (MSE)**.

---

## Part 3: Gradient Descent — The Optimization Engine

### The Core Idea

Imagine J(W,b) as a **hilly landscape**. You want to find the **lowest valley** (minimum J).

```
    J
    │     ╲        ╱
    │      ╲      ╱
    │       ╲    ╱
    │        ╲  ╱
    │         ╲╱   ← Global minimum (want to reach here)
    └──────────────── W
```

**Gradient descent = always take a small step downhill.** The direction of "downhill" is given by the negative gradient −∂J/∂W.

### Gradient Descent Update Rule

For each parameter:

```
W[ℓ] := W[ℓ] - α · ∂J/∂W[ℓ]
b[ℓ] := b[ℓ] - α · ∂J/∂b[ℓ]
```

Where **α** = learning rate (how big a step to take) and **:=** means "update/assign".

### The Learning Rate α

```
α too large:  ──► overshoot minimum, diverge ❌
              J  ╱╲  ╱╲  ╱╲   (bouncing around)

α too small:  ──► converge, but VERY slowly ⚠️
              J  ▼▼▼▼▼▼▼▼   (tiny steps)

α just right: ──► converge smoothly ✅
              J  ╲╲╲╲╲╲╲╲
```

> *"Choosing the learning rate is one of the most important decisions you'll make when training a neural network."* — Andrew Ng

---

## Part 4: What Is Backpropagation?

### The Core Problem

After forward propagation, we know the prediction **ŷ** and the loss **J**. But gradient descent needs **∂J/∂W[ℓ]** and **∂J/∂b[ℓ]** for **every layer ℓ**.

**Naive approach:** Compute each partial derivative by perturbing each weight — O(parameters²), completely infeasible for millions of weights.

**Backpropagation:** Uses the **chain rule** to compute ALL gradients in ONE backward pass — O(parameters), very efficient.

> *"Backpropagation is just the chain rule applied cleverly and efficiently."* — Andrew Ng

---

## Part 5: The Chain Rule — The Math Foundation

### Simple Example First

Suppose:

```
J = f(a),   a = g(z),   z = wx + b
```

We want **∂J/∂w**. Using the chain rule:

```
∂J/∂w = (∂J/∂a) · (∂a/∂z) · (∂z/∂w)
```

This is how the gradient of J "flows backward" through each function.

### Notation Andrew Ng Uses

He defines a shorthand called **"delta"**:

$$\delta^{[\ell]} = \frac{\partial J}{\partial Z^{[\ell]}}$$

This is the gradient of the cost with respect to the pre-activation Z in layer ℓ. Once we have δ[ℓ], computing the weight gradients is straightforward.

---

## Part 6: Backpropagation — Step by Step

### Setup: A 3-Layer Network

```
X ──► [Layer 1] ──► A[1] ──► [Layer 2] ──► A[2] ──► [Layer 3] ──► A[3] = ŷ
                                                                       │
                                                                    Loss J
```

**Forward propagation stored these values:**

```
Z[1] = W[1]·X + b[1]          A[1] = g(Z[1])
Z[2] = W[2]·A[1] + b[2]       A[2] = g(Z[2])
Z[3] = W[3]·A[2] + b[3]       A[3] = g(Z[3]) = ŷ
```

### Backward Pass — Layer by Layer

**Step 1: Gradient at Output Layer (Layer 3)**

```
dA[3] = ∂J/∂A[3] = -(y/ŷ) + (1-y)/(1-ŷ)    ← derivative of cross-entropy loss
dZ[3] = dA[3] · g'(Z[3])                     ← g' = derivative of activation
dW[3] = (1/m) · dZ[3] · A[2]ᵀ               ← gradient w.r.t. weights
db[3] = (1/m) · Σ dZ[3]                      ← gradient w.r.t. bias
```

**Step 2: Propagate Gradient to Layer 2**

```
dA[2] = W[3]ᵀ · dZ[3]                        ← pass gradient backward
dZ[2] = dA[2] · g'(Z[2])                     ← element-wise multiply
dW[2] = (1/m) · dZ[2] · A[1]ᵀ
db[2] = (1/m) · Σ dZ[2]
```

**Step 3: Propagate Gradient to Layer 1**

```
dA[1] = W[2]ᵀ · dZ[2]
dZ[1] = dA[1] · g'(Z[1])
dW[1] = (1/m) · dZ[1] · Xᵀ
db[1] = (1/m) · Σ dZ[1]
```

### The General Backprop Formula for Layer ℓ

```
dZ[ℓ]    = dA[ℓ] ⊙ g'(Z[ℓ])          ← ⊙ = element-wise multiply
dW[ℓ]    = (1/m) · dZ[ℓ] · A[ℓ⁻¹]ᵀ
db[ℓ]    = (1/m) · Σ dZ[ℓ]
dA[ℓ⁻¹] = W[ℓ]ᵀ · dZ[ℓ]             ← pass to previous layer
```

Apply this formula from layer L down to layer 1.

---

## Part 7: Derivatives of Activation Functions

### Sigmoid: g(z) = 1/(1+e⁻ᶻ)

$$g'(z) = g(z) \cdot (1 - g(z)) = a \cdot (1 - a)$$

Very convenient — just use the already-computed activation.

```
Example: If a = 0.8 → g'(z) = 0.8 × 0.2 = 0.16
```

### ReLU: g(z) = max(0, z)

$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

Extremely simple derivative — just a switch (1 or 0).

### Linear: g(z) = z

$$g'(z) = 1 \quad \text{(always)}$$

### Why Sigmoid's Gradient is Problematic

```
For z = 10:   a = sigmoid(10) ≈ 1.0
              g'(z) = 1.0 × (1-1.0) = 0.0  ← ZERO gradient!

For z = -10:  a = sigmoid(-10) ≈ 0.0
              g'(z) = 0.0 × (1-0.0) = 0.0  ← ZERO gradient!
```

When gradients become zero → parameters stop updating → learning stops. This is the **vanishing gradient problem** — a key reason ReLU replaced sigmoid in hidden layers.

---

## Part 8: Vanishing & Exploding Gradients

### Vanishing Gradients

In deep networks, the gradient is multiplied through many layers:

```
dZ[1] = dZ[L] · W[L] · g'(Z[L]) · W[L-1] · g'(Z[L-1]) · ... · W[1] · g'(Z[1])
```

If each **g'(z) ≈ 0.1** (common with sigmoid) and **L = 10 layers**:

```
gradient ≈ 0.1^10 = 0.0000000001
```

The gradient in early layers is essentially zero → early layers learn almost nothing.

### Exploding Gradients

If each weight **W ≈ 2** and you have 10 layers:

```
gradient ≈ 2^10 = 1024
```

Gradient grows exponentially → updates become huge → parameters blow up → NaN values.

### Solutions

| Problem | Solutions |
|---------|-----------|
| Vanishing gradients | Use ReLU activation, careful weight initialization, batch normalization, skip connections (ResNets) |
| Exploding gradients | Gradient clipping, careful weight initialization, smaller learning rate |

---

## Part 9: Weight Initialization

### Why NOT Initialize to Zero?

If W = 0 and b = 0 for all layers:

```
All neurons in the same layer compute IDENTICAL outputs
→ All gradients are IDENTICAL
→ All weights update IDENTICALLY
→ All neurons remain IDENTICAL forever
→ Every layer reduces to just 1 effective neuron
```

This is called the **symmetry problem** — the network never learns different features.

> *"If you initialize all weights to zero, all neurons are symmetric and they'll always compute the same thing. You might as well have 1 neuron per layer."* — Andrew Ng

### Correct Initialization: Random + Small

```python
W[ℓ] = np.random.randn(n[ℓ], n[ℓ-1]) * 0.01   # small random values
b[ℓ] = np.zeros((n[ℓ], 1))                      # zeros fine for bias
```

**Why small?** Large weights → large z → sigmoid/tanh in flat region → vanishing gradients.

### Better Initialization Methods

**Xavier / Glorot Initialization (for tanh):**

```python
W[ℓ] = np.random.randn(n[ℓ], n[ℓ-1]) * np.sqrt(1 / n[ℓ-1])
```

**He Initialization (for ReLU) ← Recommended:**

```python
W[ℓ] = np.random.randn(n[ℓ], n[ℓ-1]) * np.sqrt(2 / n[ℓ-1])
```

**Intuition:** Scale random weights by a factor that keeps the variance of activations stable as you go deeper — preventing both vanishing and exploding gradients. TensorFlow uses He initialization automatically with ReLU.

---

## Part 10: Mini-Batch Gradient Descent

### Three Variants

**Batch Gradient Descent** — use ALL m training examples per step. Very accurate gradient, but very slow for large datasets (m = 1,000,000 → one step takes forever).

**Stochastic Gradient Descent (SGD)** — use just ONE training example per step. Very fast updates, but very noisy and doesn't fully utilize vectorization.

**Mini-Batch Gradient Descent ← Andrew Ng's Recommendation:**

```
Split training set into mini-batches of size B (typically 32, 64, 128, 256)
For each mini-batch:
    1. Forward prop on mini-batch
    2. Compute loss on mini-batch
    3. Backprop on mini-batch
    4. Update parameters
```

### Comparison

| Method | Batch Size | Speed | Noise | Memory |
|--------|------------|-------|-------|--------|
| Batch GD | m (all) | Slow | Low | High |
| Stochastic GD | 1 | Fast | High | Low |
| Mini-Batch GD | 32–256 | Fast ✅ | Med ✅ | Med ✅ |

**Mini-batch size guidelines:** use full batch for small datasets (m < 2000); try 32, 64, 128, 256 for large datasets (powers of 2 for GPU efficiency).

### One Epoch vs One Step

```
m = 1,000,000 examples, batch size = 1,000

Steps per epoch = 1,000,000 / 1,000 = 1,000 steps

After 1,000 steps → 1 epoch (one full pass through data)
```

---

## Part 11: Optimization Algorithms

### Problem with Standard Gradient Descent

Even with mini-batches, gradient descent can oscillate in some directions, move slowly in others, or get stuck in saddle points.

### Gradient Descent with Momentum

**Idea:** Build up **velocity** — like a ball rolling downhill — instead of updating purely based on the current gradient.

```
Vdw = β · Vdw + (1-β) · dW     ← exponentially weighted average of gradients
Vdb = β · Vdb + (1-β) · db

W = W - α · Vdw
b = b - α · Vdb
```

- **β** = momentum term (typically **0.9**)
- Effect: smooths out oscillations, accelerates in consistent directions

```
Without Momentum:    ↙↗↙↗↙↗↙  (oscillates, slow)
With Momentum:       ↘↘↘↘↘↘↘  (smooth, fast)
```

### RMSProp

**Idea:** Adapt the learning rate per parameter — slow down where gradients are large, speed up where they're small.

```
Sdw = β₂ · Sdw + (1-β₂) · dW²     ← running average of squared gradient
Sdb = β₂ · Sdb + (1-β₂) · db²

W = W - α · dW / (√Sdw + ε)
b = b - α · db / (√Sdb + ε)
```

**ε** = small number (like 10⁻⁸) to avoid division by zero. Large gradient → large S → smaller update (slows down). Small gradient → small S → larger update (speeds up).

### Adam Optimizer ← Andrew Ng's Top Recommendation

**Adam = Adaptive Moment Estimation = Momentum + RMSProp combined**

```
# Momentum part (first moment)
Vdw = β₁ · Vdw + (1-β₁) · dW
Vdb = β₁ · Vdb + (1-β₁) · db

# RMSProp part (second moment)
Sdw = β₂ · Sdw + (1-β₂) · dW²
Sdb = β₂ · Sdb + (1-β₂) · db²

# Bias correction
Vdw_corr = Vdw / (1 - β₁ᵗ)
Vdb_corr = Vdb / (1 - β₁ᵗ)
Sdw_corr = Sdw / (1 - β₂ᵗ)
Sdb_corr = Sdb / (1 - β₂ᵗ)

# Update
W = W - α · Vdw_corr / (√Sdw_corr + ε)
b = b - α · Vdb_corr / (√Sdb_corr + ε)
```

**Recommended hyperparameters:**

| Parameter | Recommended Value |
|-----------|------------------|
| **α** (learning rate) | Tune this |
| **β₁** (momentum) | 0.9 |
| **β₂** (RMSProp) | 0.999 |
| **ε** | 10⁻⁸ |

> *"Adam works well in most situations. It's one of those rare algorithms that just works without much tuning. I almost always start with Adam."* — Andrew Ng

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy'
)
```

---

## Part 12: Learning Rate Decay

Using a **fixed learning rate** throughout training is suboptimal — early in training you want large α for big progress; later you want small α to fine-tune without overshooting.

### Decay Methods

**Exponential Decay:**

$$\alpha = \alpha_0 \cdot \text{decay\_rate}^{\text{epoch}}$$

**Step Decay (most common):**

$$\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \cdot \text{epoch}}$$

**Example** with α₀ = 0.1, decay_rate = 1:

```
Epoch 1:  α = 0.1 / (1 + 1·1) = 0.05
Epoch 2:  α = 0.1 / (1 + 1·2) = 0.033
Epoch 3:  α = 0.1 / (1 + 1·3) = 0.025
...
```

> *"Learning rate decay is often not the first thing I tune. Get the architecture right first, then consider decay."* — Andrew Ng

---

## Part 13: The Full Backpropagation — Code Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0       # gradient is 0 where Z was ≤ 0
    return dZ

# ── Forward Pass (stores cache for backprop) ─────────────────
def forward_pass(X, parameters):
    caches = []
    A = X
    L = len(parameters)

    for l in range(L):
        W, b, activation = parameters[l]
        A_prev = A
        Z = np.dot(W, A_prev) + b

        if activation == 'relu':
            A = relu(Z)
        elif activation == 'sigmoid':
            A = sigmoid(Z)

        cache = (A_prev, W, b, Z)
        caches.append(cache)

    return A, caches    # A is ŷ

# ── Backward Pass ─────────────────────────────────────────────
def backward_pass(AL, Y, caches, activations):
    grads = {}
    m = AL.shape[1]
    L = len(caches)

    # Output layer gradient
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    for l in reversed(range(L)):
        A_prev, W, b, Z = caches[l]
        activation = activations[l]

        if activation == 'sigmoid':
            dZ = sigmoid_backward(dAL, Z)
        elif activation == 'relu':
            dZ = relu_backward(dAL, Z)

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        grads[f'dW{l+1}'] = dW
        grads[f'db{l+1}'] = db
        dAL = dA_prev    # pass gradient to previous layer

    return grads

# ── Parameter Update ──────────────────────────────────────────
def update_parameters(parameters, grads, learning_rate):
    updated = []
    for l in range(len(parameters)):
        W, b, activation = parameters[l]
        W = W - learning_rate * grads[f'dW{l+1}']
        b = b - learning_rate * grads[f'db{l+1}']
        updated.append((W, b, activation))
    return updated

# ── Full Training Loop ────────────────────────────────────────
def train(X, Y, parameters, learning_rate=0.01, epochs=1000):
    activations = [p[2] for p in parameters]
    costs = []

    for epoch in range(epochs):
        AL, caches = forward_pass(X, parameters)

        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))

        grads = backward_pass(AL, Y, caches, activations)
        parameters = update_parameters(parameters, grads, learning_rate)

        if epoch % 100 == 0:
            costs.append(cost)
            print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return parameters, costs
```

---

## Part 14: End to End in TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# ── 1. Prepare Data ───────────────────────────────────────────
X_train = np.random.randn(1000, 10)   # 1000 examples, 10 features
y_train = (np.random.rand(1000) > 0.5).astype(float)

# ── 2. Build Model ────────────────────────────────────────────
model = Sequential([
    Dense(25, activation='relu', input_shape=(10,)),   # Layer 1
    Dense(15, activation='relu'),                       # Layer 2
    Dense(1,  activation='sigmoid')                     # Output
])

# ── 3. Compile ────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── 4. Train ──────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ── 5. Evaluate ───────────────────────────────────────────────
loss, acc = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {acc:.2%}")

# ── 6. Predict ────────────────────────────────────────────────
X_test = np.random.randn(5, 10)
predictions = model.predict(X_test)
print(predictions)   # probabilities between 0 and 1
```

> *"TensorFlow computes backpropagation automatically using 'automatic differentiation'. You just call model.fit() and it handles everything. But I want you to know what it's doing underneath."* — Andrew Ng

---

## Part 15: Debugging Training — The Learning Curve

Andrew Ng teaches you to **plot cost vs. epoch** to diagnose problems.

### What Good Training Looks Like

```
Cost J
  │╲
  │ ╲
  │  ╲
  │   ╲___
  │       ╲___
  │           ╲_______
  └──────────────────── Epochs
  (Cost smoothly decreasing → ✅)
```

### Diagnosing Problems

| Symptom | Likely Cause |
|---------|-------------|
| Cost going **up** | Learning rate too large, or bug in code |
| Cost **not decreasing** | Learning rate too small, wrong activation, or poor initialization |
| Cost decreasing then **going up** | Overfitting — need regularization |
| Cost **oscillating** | Learning rate too large, or batch size too small |

---

## Part 16: Full Summary — What Backpropagation Does

```
┌────────────────────────────────────────────────────────────┐
│              FORWARD PROPAGATION                           │
│                                                            │
│  X ──► Z[1]=W[1]X+b[1] ──► A[1]=g(Z[1])                  │
│        ──► Z[2]=W[2]A[1]+b[2] ──► A[2]=g(Z[2])            │
│        ──► Z[3]=W[3]A[2]+b[3] ──► A[3]=ŷ                  │
│                                      │                     │
│                                   LOSS J                   │
│                                      │                     │
│              BACKWARD PROPAGATION    │                     │
│                                      ▼                     │
│  dW[1],db[1] ◄── dZ[1] ◄── dA[1] ◄── dZ[2] ◄── dA[2]     │
│                              ◄── dW[2],db[2]  ◄── dZ[3]   │
│                                           ◄── dW[3],db[3] │
│                                                            │
│  Parameters updated:                                       │
│  W[ℓ] = W[ℓ] - α·dW[ℓ]   for all ℓ                       │
│  b[ℓ] = b[ℓ] - α·db[ℓ]   for all ℓ                       │
└────────────────────────────────────────────────────────────┘
```

---

## Part 17: Key Takeaways

- **Cost function** J measures total prediction error. Training = minimizing J.
- **Gradient descent** updates parameters by stepping in the negative gradient direction.
- **Backpropagation** efficiently computes all gradients using the chain rule — one backward pass gives gradients for every parameter.
- **Weight initialization** must be random and small — zero initialization causes symmetry failure.
- **Vanishing gradients** (from sigmoid) kill learning in deep nets — use **ReLU** for hidden layers.
- **Mini-batch gradient descent** is the standard — batch sizes of 32–256 balance speed and stability.
- **Adam optimizer** combines Momentum + RMSProp — Andrew Ng's default recommendation.
- **Plot cost vs. epoch** to debug: rising cost = large α; flat cost = small α or bug.
- TensorFlow handles forward prop, backprop, and parameter updates automatically via `model.fit()`.

---

## Complete Glossary

| Term | Meaning |
|------|---------|
| **Loss L** | Error on a single training example |
| **Cost J** | Average loss over all training examples |
| **Gradient Descent** | Iterative optimization: move parameters in direction of −∇J |
| **Backpropagation** | Chain rule applied backward through network to compute all ∂J/∂W |
| **Learning Rate α** | Step size in gradient descent |
| **Mini-batch** | Small subset of training data used for one update step |
| **Epoch** | One full pass through the entire training set |
| **Momentum** | Running average of gradients to smooth updates |
| **RMSProp** | Adaptive learning rate — divides by running avg of squared gradient |
| **Adam** | Momentum + RMSProp combined — best general-purpose optimizer |
| **Vanishing Gradient** | Gradients shrink to ~0 in early layers; learning stops |
| **Exploding Gradient** | Gradients grow exponentially; parameters blow up |
| **He Initialization** | Smart weight init for ReLU: scale by √(2/n[ℓ⁻¹]) |
| **Symmetry Breaking** | Reason we use random (not zero) initialization |
| **Learning Curve** | Plot of J vs. epoch — used to diagnose training problems |
| **Automatic Differentiation** | TensorFlow's system to auto-compute backprop |