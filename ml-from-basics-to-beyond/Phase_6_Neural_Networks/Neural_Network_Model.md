# Neural Networks & Neural Network Model
### Andrew Ng's ML Specialization | Course 2, Week 1

---

## Part 1: Why Neural Networks?

### The Problem with Traditional ML

Andrew Ng begins by revisiting **logistic regression and linear regression** — and showing where they fall short.

Imagine predicting house prices with just 2 features (size, bedrooms) — logistic regression works fine. But with **100 features**, capturing non-linear relationships requires adding:
- Quadratic terms: x₁², x₂², x₁x₂ ...
- Cubic terms: x₁³, x₁²x₂ ...

With 100 features, quadratic pairs alone produce ~**5,000 terms**. This is computationally expensive, prone to overfitting, and impractical for images, audio, or text.

**This is exactly the gap neural networks fill.**

> *"Neural networks can automatically learn the right features from raw data — you don't have to engineer them by hand."* — Andrew Ng

### The Killer Application: Computer Vision

An image of size 1000×1000 pixels = **1,000,000 features** (pixel intensities). Using logistic regression with quadratic features would produce ~**500 billion** feature pairs — completely infeasible.

Neural networks handle this by **learning a hierarchy of features** — raw pixels → edges → shapes → object parts → full objects.

---

## Part 2: Biological Inspiration

### The Biological Neuron

```
Dendrites (inputs)
     │
     ▼
  [Cell Body] ── computes ──► Axon ──► (to next neuron)
```

- **Dendrites** receive signals from other neurons
- **Cell body** sums up the signals
- If the sum exceeds a threshold, the neuron **fires** — sends a signal via its axon
- **Synapses** are the junctions between neurons — their strength determines influence

### The Artificial Neuron

The artificial neuron mimics this:

```
x₁ ──(w₁)──┐
x₂ ──(w₂)──┤──► z = Σwᵢxᵢ + b ──► a = g(z) ──► output
x₃ ──(w₃)──┘
```

| Biological | Artificial |
|------------|------------|
| Dendrites  | Inputs x₁, x₂, x₃ |
| Synaptic strengths | Weights w₁, w₂, w₃ |
| Cell body computation | Weighted sum + bias z |
| Firing decision | Activation function g(z) |
| Axon signal | Output a |

> *"The analogy to the brain is loose. Don't take it too literally — modern deep learning theory is based on math, not neuroscience."* — Andrew Ng

---

## Part 3: From Logistic Regression to a Neural Network

### A Single Neuron IS Logistic Regression

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
a = sigmoid(z) = 1 / (1 + e⁻ᶻ)
```

This single unit is **exactly logistic regression**. The magic happens when we **chain many of these together** in layers.

---

## Part 4: The Neural Network — Layers

### The T-Shirt Demand Prediction Example

**Goal:** Predict if a T-shirt will be a top seller (1 = yes, 0 = no)

**4 Input Features:**

| Feature | Variable |
|---------|----------|
| Price | x₁ |
| Shipping Cost | x₂ |
| Marketing Spend | x₃ |
| Material Quality | x₄ |

**Architecture — 3 Hidden Neurons → 1 Output:**

```
         INPUT LAYER          HIDDEN LAYER        OUTPUT LAYER
         (Layer 0)            (Layer 1)            (Layer 2)

Price ──────────────────►  [Affordability]  ─────►
Shipping Cost ──────────►                          [Top Seller?] ──► ŷ
Marketing ──────────────►  [Awareness]     ─────►
Material ───────────────►
                         ►  [Quality]      ─────►
```

**What each hidden neuron might learn:**
- **Neuron 1 (Affordability):** High activation when price and shipping cost are low
- **Neuron 2 (Awareness):** High activation when marketing spend is high
- **Neuron 3 (Quality):** High activation when material quality is high

> *"We did NOT tell the network what affordability or awareness means. It figures out on its own that these intermediate concepts are useful for the prediction. The network invents its own features."* — Andrew Ng

This is **representation learning** — the hidden layer learns to represent data in a new, more useful space.

---

## Part 5: Neural Network Model — Formal Structure

### Layer Types

```
┌─────────────────────────────────────────────────────┐
│  INPUT    │   HIDDEN LAYER 1  │  HIDDEN LAYER 2  │  OUTPUT  │
│  LAYER    │                   │                  │  LAYER   │
│  (Layer 0)│   (Layer 1)       │   (Layer 2)      │ (Layer 3)│
│           │                   │                  │          │
│   x₁      │  a₁[1] a₂[1]     │  a₁[2] a₂[2]    │  a[3]=ŷ  │
│   x₂      │  a₃[1] a₄[1]     │  a₃[2] a₄[2]    │          │
│   x₃      │                   │                  │          │
└─────────────────────────────────────────────────────┘
```

### Notation (Andrew Ng's exact notation)

| Symbol | Meaning |
|--------|---------|
| **L** | Total number of layers (NOT counting input layer) |
| **n[ℓ]** | Number of neurons in layer ℓ |
| **a[ℓ]** | Vector of activations from layer ℓ |
| **a[0]** | = X (the input features) |
| **a[L]** | = ŷ (final output prediction) |
| **W[ℓ]** | Weight matrix for layer ℓ |
| **b[ℓ]** | Bias vector for layer ℓ |
| **aⱼ[ℓ]** | Activation of neuron j in layer ℓ |

### Counting Layers (Andrew Ng's Rule)

- The **input layer is Layer 0** and is NOT counted
- Only hidden layers + output layer are counted

```
Layer 0 (Input):   x₁, x₂, x₃, x₄    ← NOT counted
Layer 1 (Hidden):  4 neurons           ← counted (layer 1)
Layer 2 (Hidden):  4 neurons           ← counted (layer 2)
Layer 3 (Output):  1 neuron            ← counted (layer 3)
→ This is a 3-layer neural network
```

---

## Part 6: Forward Propagation — The Core Algorithm

This is the algorithm the network uses to make a prediction.

### Step 1: Computation Inside ONE Neuron

For neuron **j** in layer **ℓ**:

```
zⱼ[ℓ] = wⱼ[ℓ]ᵀ · a[ℓ⁻¹] + bⱼ[ℓ]
aⱼ[ℓ] = g(zⱼ[ℓ])
```

- **wⱼ[ℓ]** is a vector of weights (one per neuron in the previous layer)
- **a[ℓ⁻¹]** is the activation vector from the previous layer
- **bⱼ[ℓ]** is a scalar bias
- **g** is the activation function

### Step 2: Computation for ONE Full Layer (Vectorized)

```
Z[ℓ] = W[ℓ] · A[ℓ⁻¹] + b[ℓ]
A[ℓ] = g(Z[ℓ])
```

**Dimensions:**

| Matrix | Shape |
|--------|-------|
| W[ℓ] | (n[ℓ] × n[ℓ⁻¹]) — one row per neuron in this layer |
| A[ℓ⁻¹] | (n[ℓ⁻¹] × 1) |
| b[ℓ] | (n[ℓ] × 1) |
| Z[ℓ] and A[ℓ] | (n[ℓ] × 1) |

### Step 3: Full Forward Propagation (Layer by Layer)

For a 3-layer network:

```
# Layer 1
Z[1] = W[1] · X + b[1]
A[1] = g(Z[1])

# Layer 2
Z[2] = W[2] · A[1] + b[2]
A[2] = g(Z[2])

# Layer 3 (Output)
Z[3] = W[3] · A[2] + b[3]
A[3] = g(Z[3])    ← This is ŷ
```

---

## Part 7: Worked Example — Forward Propagation by Hand

### Setup

- **2 input features:** x = [200, 17] (price=200, size=17)
- **Layer 1:** 3 neurons, sigmoid activation
- **Layer 2:** 1 neuron, sigmoid activation

### Given Weights (simplified)

```
W[1] = [[ 1,  0],    b[1] = [[0],
        [ 0,  1],             [0],
        [ 1,  1]]             [0]]

W[2] = [[0.5, 0.3, 0.2]]    b[2] = [[0]]
```

### Forward Pass — Layer 1

```
Z[1] = W[1] · x + b[1]

Z₁[1] = 1×200 + 0×17 + 0 = 200
Z₂[1] = 0×200 + 1×17 + 0 = 17
Z₃[1] = 1×200 + 1×17 + 0 = 217

A[1] = sigmoid([200, 17, 217])
     = [≈1.0, ≈1.0, ≈1.0]    (sigmoid of large numbers ≈ 1)
```

### Forward Pass — Layer 2 (Output)

```
Z[2] = W[2] · A[1] + b[2]
     = 0.5×1 + 0.3×1 + 0.2×1 + 0
     = 1.0

A[2] = sigmoid(1.0) = 1/(1+e⁻¹) ≈ 0.731
```

**ŷ ≈ 0.731** → 73.1% probability of being a top seller.

---

## Part 8: Activation Functions

### Why Do We Need Them?

Without an activation function (using linear g(z) = z):

```
A[1] = W[1]·X + b[1]
A[2] = W[2]·A[1] + b[2]
     = W[2]·(W[1]·X + b[1]) + b[2]
     = (W[2]·W[1])·X + (W[2]·b[1] + b[2])
     = W'·X + b'
```

**Result: No matter how many layers, you get a linear function of X.** A 100-layer network with no activations = equivalent to 1-layer logistic regression.

> *"Activation functions are what give neural networks their non-linear power."* — Andrew Ng

---

### Activation Function 1: Sigmoid

$$g(z) = \frac{1}{1 + e^{-z}}$$

```
Output:  0 ────────────────── 1
Shape:   S-curve
Range:   (0, 1)
```

**When to use:** Output layer for **binary classification**

**Problem:** For very large or very small z, gradient ≈ 0 → **vanishing gradient problem** → learning slows down

---

### Activation Function 2: ReLU ← Andrew Ng's Default Recommendation

$$g(z) = \max(0, z)$$

```
     |      /
     |     /
     |    /
─────|───/────── z
     |  0
```

**Range:** [0, ∞)

**Advantages:** very fast to compute, gradient is 0 or 1 (no vanishing gradient), works extremely well in practice — standard for **all hidden layers**.

**One issue:** "Dying ReLU" — neurons with z < 0 always output 0. Solution: **Leaky ReLU**: g(z) = max(0.01z, z)

---

### Activation Function 3: Linear (No Activation)

$$g(z) = z$$

**When to use:** Output layer for **regression** problems where ŷ can be any real number (house prices, temperature, etc.)

**Never use in hidden layers** — makes the entire network linear.

---

### Activation Function 4: Softmax

For **multi-class classification** (e.g., classify digit 0–9):

$$g(z_j) = \frac{e^{z_j}}{\sum e^{z_k}}$$

Outputs a **probability distribution** across all classes (sum = 1).

---

### Andrew Ng's Activation Cheat Sheet

```
┌──────────────────────────────────────────────────────────┐
│  LAYER TYPE          │  TASK              │  ACTIVATION  │
├──────────────────────────────────────────────────────────┤
│  Hidden layer        │  Any               │  ReLU ✅     │
│  Output layer        │  Binary classify   │  Sigmoid     │
│  Output layer        │  Regression        │  Linear      │
│  Output layer        │  Multiclass        │  Softmax     │
└──────────────────────────────────────────────────────────┘
```

---

## Part 9: Why Deep Networks? (Intuition)

Andrew Ng explains the "depth" intuition with the **face recognition** example:

```
Layer 1:  Detects edges (horizontal, vertical, diagonal lines)
          → Very low-level features from pixel groups

Layer 2:  Combines edges into facial parts
          → Eyes, nose, mouth outlines

Layer 3:  Combines parts into full faces
          → Recognizes overall face structure

Layer 4:  Matches specific people
          → "This is Person X"
```

**Same idea for audio:**

```
Layer 1:  Low-level audio features (frequency bands)
Layer 2:  Phonemes (basic speech sounds)
Layer 3:  Words
Layer 4:  Sentences / phrases
```

This is **hierarchical representation** — each layer builds on the last.

> *"There's a fascinating finding that the first layer of a neural network trained on images often looks like edge detectors — even though you never told it to look for edges. It figured that out on its own."* — Andrew Ng

---

## Part 10: The Neural Network Model — Complete Summary

```
Input X
  │
  ▼
[Layer 1: W[1], b[1]] ──► A[1] = g(W[1]·X + b[1])
  │
  ▼
[Layer 2: W[2], b[2]] ──► A[2] = g(W[2]·A[1] + b[2])
  │
  ▼
  ...
  │
  ▼
[Layer L: W[L], b[L]] ──► A[L] = g(W[L]·A[L-1] + b[L]) = ŷ
```

**The model's parameters are:** all the W[ℓ] and b[ℓ] matrices across all layers.

**Total parameters** = sum across all layers of (n[ℓ] × n[ℓ⁻¹] + n[ℓ])

---

## Part 11: TensorFlow Implementation

### Building and Training a Neural Network

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# ── STEP 1: Build the Model ──────────────────────────────────
model = Sequential([
    Dense(units=25, activation='relu'),     # Hidden Layer 1: 25 neurons
    Dense(units=15, activation='relu'),     # Hidden Layer 2: 15 neurons
    Dense(units=1,  activation='sigmoid')   # Output Layer: binary classification
])

# ── STEP 2: Compile ──────────────────────────────────────────
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

# ── STEP 3: Train ────────────────────────────────────────────
model.fit(X_train, y_train, epochs=100)

# ── STEP 4: Predict ──────────────────────────────────────────
y_pred = model.predict(X_test)
```

**What each line means:**

| Code | Meaning |
|------|---------|
| `Sequential` | Layers stacked one after another |
| `Dense(25, 'relu')` | Fully connected layer, 25 neurons, ReLU activation |
| `compile(loss=...)` | Define how to measure error |
| `Adam` | Optimizer (fancy gradient descent) |
| `fit(X, y, epochs=100)` | Run forward prop + backprop 100 times |
| `predict(X_test)` | Run forward propagation only |

---

## Part 12: Under the Hood — NumPy Implementation

Andrew Ng insists on understanding what Keras does internally:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def dense_layer(a_in, W, b, activation='relu'):
    """
    One fully connected layer.
    a_in: input activations (n_prev,)
    W:    weight matrix (n_prev, n_this)
    b:    bias vector (n_this,)
    """
    z = np.dot(a_in, W) + b
    if activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'relu':
        return relu(z)
    else:
        return z                      # linear

def forward_propagation(X, parameters):
    """
    Full forward pass through network.
    parameters = list of (W, b, activation) tuples
    """
    a = X
    for (W, b, act) in parameters:
        a = dense_layer(a, W, b, act)
    return a

# ── Example Usage ──────────────────────────────────────────
W1 = np.random.randn(4, 25)   # 4 input features → 25 hidden neurons
b1 = np.zeros(25)

W2 = np.random.randn(25, 15)  # 25 → 15
b2 = np.zeros(15)

W3 = np.random.randn(15, 1)   # 15 → 1 output
b3 = np.zeros(1)

parameters = [
    (W1, b1, 'relu'),
    (W2, b2, 'relu'),
    (W3, b3, 'sigmoid')
]

x_sample = np.array([1.0, 0.5, 2.0, -1.0])
y_hat = forward_propagation(x_sample, parameters)
print(f"Prediction: {y_hat}")
```

> *"I want you to understand that Keras is just doing this — matrix multiplications and activation functions — under the hood. Don't treat it as a black box."* — Andrew Ng

---

## Part 13: Training Overview (Big Picture)

```
┌─────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK TRAINING LOOP               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Initialize W[ℓ], b[ℓ] randomly (small values)          │
│                    │                                        │
│  2. FORWARD PROP ──► compute ŷ                              │
│                    │                                        │
│  3. COMPUTE LOSS:  J = -[y·log(ŷ) + (1-y)·log(1-ŷ)]       │
│                    │                                        │
│  4. BACKPROP ──────► compute ∂J/∂W[ℓ], ∂J/∂b[ℓ]            │
│                    │                                        │
│  5. UPDATE:        W[ℓ] = W[ℓ] - α · ∂J/∂W[ℓ]              │
│                    b[ℓ] = b[ℓ] - α · ∂J/∂b[ℓ]              │
│                    │                                        │
│  6. REPEAT steps 2–5 for many epochs                        │
└─────────────────────────────────────────────────────────────┘
```

- **α** = learning rate (you choose this)
- **Backpropagation** uses the **chain rule** to compute gradients efficiently
- TensorFlow handles steps 4 & 5 automatically via `model.fit()`

---

## Part 14: Complete Glossary

| Term | Definition |
|------|-----------|
| **Neuron** | A single unit that computes z = w·x + b, then a = g(z) |
| **Layer** | A group of neurons that all receive the same input |
| **Hidden Layer** | A layer whose activations you never directly observe |
| **Activation** | The output value of a neuron after applying g(z) |
| **Forward Propagation** | Computing ŷ from X by passing data through all layers |
| **Weights (W)** | Parameters the network learns — control connection strength |
| **Bias (b)** | A learnable offset term in each neuron |
| **Loss** | How wrong a single prediction is |
| **Cost (J)** | Average loss over the entire training set |
| **Epoch** | One complete pass through all training examples |
| **Dense Layer** | Fully connected layer — every input connects to every neuron |
| **ReLU** | Rectified Linear Unit: g(z) = max(0, z) |
| **Backpropagation** | Algorithm that computes gradients using the chain rule |
| **Hyperparameter** | Something you set, not learned: #layers, #neurons, α |

---

## Part 15: Key Takeaways

- **Neural networks** are stacks of logistic regression units organized in layers.
- The **input layer** feeds raw features; **hidden layers** learn intermediate representations; the **output layer** produces the final prediction.
- **Forward propagation** computes predictions layer by layer: Z[ℓ] = W[ℓ]·A[ℓ⁻¹] + b[ℓ], then A[ℓ] = g(Z[ℓ]).
- **Activation functions** introduce non-linearity — without them, the entire network collapses to a linear model.
- **ReLU** is the default for hidden layers. **Sigmoid** for binary output. **Linear** for regression output.
- Deeper networks learn **hierarchical representations** — low-level → high-level features.
- In TensorFlow: `Sequential + Dense + compile + fit` is all you need. But understanding the NumPy mechanics underneath is essential.
- Training = forward prop → compute loss → backprop → update weights → repeat.