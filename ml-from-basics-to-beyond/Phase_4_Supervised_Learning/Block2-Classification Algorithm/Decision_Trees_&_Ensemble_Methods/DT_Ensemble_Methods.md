# Course 2 Week 4 — Decision Trees to XGBoost
### Andrew Ng's ML Specialization — Advanced Learning Algorithms
#### Decision Tree → Random Forest → Gradient Boosting → XGBoost

---

## The Big Picture — Why This Week Matters

Andrew Ng opens this section with an important observation:

> *"Decision Trees and Tree Ensembles are the other major category of ML algorithm alongside Neural Networks. For many real-world applications — especially on structured/tabular data — they work extremely well and are often the first thing practitioners reach for."*

**The deliberate progression this week:**

```
Decision Tree            (one tree, limited power)
        ↓
Why one tree isn't enough (variance problem)
        ↓
Tree Ensembles           (combine many trees)
        ↓
Bagging → Random Forest  (reduce variance)
        ↓
Boosting → XGBoost       (reduce bias AND variance)
        ↓
When to use Trees vs Neural Networks
```

Every concept builds directly on the previous one. By the end, XGBoost will feel like a natural conclusion — not a mysterious algorithm.

---

# PART 1 — Decision Trees

## 1.1 Core Intuition

Andrew Ng introduces Decision Trees with the most natural possible framing:

> *"A Decision Tree makes predictions by asking a series of yes/no questions about the features, following the answers down a tree until it reaches a final prediction."*

This is exactly how a doctor diagnoses, how a customer service flowchart works, how a loan officer approves applications. Decision Trees mirror human decision-making directly.

### The Anatomy of a Decision Tree

```
                [Fever > 38°C?]          ← ROOT NODE
                /              \
              YES               NO
               ↓                 ↓
        [Cough?]           [Fatigue?]   ← INTERNAL NODES
        /      \            /      \
      YES       NO        YES       NO
       ↓         ↓          ↓         ↓
     FLU       COLD       COLD     HEALTHY  ← LEAF NODES
```

| Term | Meaning |
|---|---|
| **Root Node** | The very first question — the most informative split |
| **Internal Node** | Any question in the middle of the tree |
| **Branch** | The answer to a question (Yes/No, or a value range) |
| **Leaf Node** | The final prediction — no more questions asked |
| **Depth** | How many questions are asked along the longest path |

---

## 1.2 The Dataset Used Throughout Part 1

Predicting whether an animal is a **Cat** or **Not Cat** based on 3 binary features.

| Animal | Ear Shape | Face Shape | Whiskers | Label |
|---|---|---|---|---|
| 1 | Pointy | Round | Yes | Cat |
| 2 | Floppy | Not Round | No | Not Cat |
| 3 | Pointy | Round | Yes | Cat |
| 4 | Pointy | Not Round | No | Cat |
| 5 | Floppy | Round | Yes | Not Cat |
| 6 | Pointy | Round | No | Cat |
| 7 | Floppy | Not Round | Yes | Not Cat |
| 8 | Pointy | Round | Yes | Cat |
| 9 | Floppy | Round | No | Not Cat |
| 10 | Floppy | Round | No | Not Cat |

`m = 10` animals. 5 Cats, 5 Not Cats.

---

## 1.3 The Learning Problem: Which Feature to Split On First?

The algorithm must decide which feature — Ear Shape, Face Shape, or Whiskers — to ask about first at the root node.

Andrew Ng frames this as:

> *"At each node, we want to choose the feature that best separates the data — the one that creates the purest groups after the split. Purer groups means we're more confident in our predictions."*

**Purity** means how homogeneous a group is:
- A group of all Cats → perfectly pure
- A group of 50% Cats and 50% Not Cats → perfectly impure — maximally uncertain

We need a mathematical measure of impurity.

---

## 1.4 Measuring Impurity: Entropy

### The Intuition

Andrew Ng introduces **entropy** as the measure of impurity. The name comes from information theory — entropy measures how much "uncertainty" or "surprise" exists in a set.

- A set of all Cats → zero surprise → **zero entropy**
- A set of 50% Cats, 50% Not Cats → maximum surprise → **maximum entropy**
- Everything between → partial entropy

### The Formula

```
H(p)  =  -p * log₂(p)  -  (1-p) * log₂(1-p)
```

Where `p` = fraction of positive examples (Cats) in the set.

> **Convention:** `0 * log₂(0) = 0` (mathematically, the limit as p → 0 gives 0).

### Computing Entropy at Key Points

**Root node** — 5 Cats, 5 Not Cats, so `p = 5/10 = 0.5`:

```
H(0.5)  =  -0.5 * log₂(0.5)  -  0.5 * log₂(0.5)
        =  -0.5 * (-1)  -  0.5 * (-1)
        =  0.5 + 0.5
        =  1.0   ← Maximum entropy = maximum impurity
```

**A perfectly pure set** — all Cats, `p = 1.0`:

```
H(1.0)  =  -1.0 * log₂(1.0)  -  0 * log₂(0)
        =  0 + 0
        =  0.0   ← Zero entropy = perfect purity
```

**A mostly pure set** — 4 Cats, 1 Not Cat, `p = 4/5 = 0.8`:

```
H(0.8)  =  -0.8 * log₂(0.8)  -  0.2 * log₂(0.2)
        =  -0.8 * (-0.322)  -  0.2 * (-2.322)
        =  0.258 + 0.464
        =  0.722
```

### The Entropy Curve

```
H(p)
  │
1 │         *         ← maximum at p = 0.5
  │       *   *
  │     *       *
  │   *           *
  │ *               *
0 │*_________________*──
  0    0.2  0.5  0.8   1.0  →  p (fraction of cats)

Pure sets (p=0 or p=1) → H = 0
Mixed set (p=0.5)       → H = 1 (maximum)
```

---

## 1.5 Information Gain: Choosing the Best Split

Entropy measures impurity at a single node. **Information Gain** measures how much a split *reduces* impurity — how much "information" we gain about the labels by asking a particular question.

### The Formula

```
IG  =  H(parent)  -  [ (n_left / n) * H(left)  +  (n_right / n) * H(right) ]
```

Where:
- `H(parent)` = entropy before the split
- `H(left)`, `H(right)` = entropy of each child node after the split
- `n_left / n`, `n_right / n` = fraction of examples going to each child

The weighted average of the children's entropies is called the **weighted average entropy** after the split. Information Gain is how much lower this is compared to the parent.

> **Choose the feature with the highest Information Gain.**

---

## 1.6 Computing Information Gain for All Three Features

### Split on Ear Shape (Pointy vs Floppy)

**Pointy ear group** — Animals 1,3,4,6,8 → 4 Cats, 1 Not Cat:

```
p_left  = 4/5 = 0.8   →   H(0.8) = 0.722
```

**Floppy ear group** — Animals 2,5,7,9,10 → 1 Cat, 4 Not Cats:

```
p_right = 1/5 = 0.2   →   H(0.2) = 0.722
```

```
IG(Ear Shape)  =  1.0  -  [ (5/10)*0.722  +  (5/10)*0.722 ]
               =  1.0  -  0.722
               =  0.278
```

---

### Split on Face Shape (Round vs Not Round)

**Round face group** — Animals 1,3,5,6,8,9,10 → 4 Cats, 3 Not Cats:

```
p_left = 4/7 = 0.571

H(0.571)  =  -0.571 * log₂(0.571)  -  0.429 * log₂(0.429)
          =  -0.571*(-0.807)  -  0.429*(-1.221)
          =  0.461 + 0.524
          =  0.985
```

**Not Round face group** — Animals 2,4,7 → 1 Cat, 2 Not Cats:

```
p_right = 1/3 = 0.333

H(0.333)  =  -0.333 * log₂(0.333)  -  0.667 * log₂(0.667)
          =  -0.333*(-1.585)  -  0.667*(-0.585)
          =  0.528 + 0.390
          =  0.918
```

```
IG(Face Shape)  =  1.0  -  [ (7/10)*0.985  +  (3/10)*0.918 ]
                =  1.0  -  [ 0.690 + 0.275 ]
                =  1.0  -  0.965
                =  0.035
```

---

### Split on Whiskers (Yes vs No)

**Whiskers = Yes** — Animals 1,3,5,7,8 → 3 Cats, 2 Not Cats:

```
p_left = 3/5 = 0.6   →   H(0.6) = 0.971
```

**Whiskers = No** — Animals 2,4,6,9,10 → 2 Cats, 3 Not Cats:

```
p_right = 2/5 = 0.4   →   H(0.4) = 0.971
```

```
IG(Whiskers)  =  1.0  -  [ (5/10)*0.971  +  (5/10)*0.971 ]
              =  1.0  -  0.971
              =  0.029
```

---

### Decision: Which Feature Wins?

| Feature | Information Gain | Decision |
|---|---|---|
| Ear Shape | **0.278** | ← **Best split — use this at root** |
| Face Shape | 0.035 | |
| Whiskers | 0.029 | |

**Root node split: Ear Shape.** It reduces impurity the most.

---

## 1.7 Building the Full Tree Recursively

After splitting on Ear Shape:

```
              [Ear Shape?]
              /           \
           Pointy         Floppy
            ↓                ↓
   5 animals             5 animals
   5 Cats, 0 Not Cat     0 Cat, 5 Not Cat
   p = 1.0,  H = 0       p = 0.0,  H = 0
   PURE ✅               PURE ✅
```

Both child nodes are **perfectly pure** — no further splitting needed.

**The final tree:**

```
              [Ear Shape?]
              /           \
           Pointy         Floppy
            ↓                ↓
          CAT            NOT CAT
       (100% pure)      (100% pure)
```

> In this clean dataset, Ear Shape alone perfectly separates cats from non-cats. Real datasets produce deeper, more complex trees.

---

## 1.8 When to Stop Splitting

Andrew Ng gives four stopping criteria:

| Stopping Criterion | Reason |
|---|---|
| Node is **100% pure** (H = 0) | No benefit to splitting further |
| Splitting would exceed **maximum depth** | Prevent overfitting |
| **Information Gain** below a threshold | Split adds no meaningful information |
| Node has **too few examples** | Insufficient data to split reliably |

> **Maximum depth is a hyperparameter** — larger depth = more complex tree = more overfitting risk.

---

## 1.9 One-Hot Encoding for Categorical Features

What if a feature has more than 2 values — say Ear Shape has three values: Pointy, Floppy, Oval?

**Solution — One-Hot Encoding:** Create one binary feature for each possible value.

| Original | Ear = Pointy | Ear = Floppy | Ear = Oval |
|---|---|---|---|
| Pointy | 1 | 0 | 0 |
| Floppy | 0 | 1 | 0 |
| Oval | 0 | 0 | 1 |

Now each feature is binary — the decision tree algorithm works as before. This also means Decision Trees and Neural Networks can both use the same one-hot encoded inputs.

---

## 1.10 Continuous Features

For continuous features — like tumour size, temperature, price — you need to find the **best threshold** to split on.

### Process

1. Sort all values of the feature
2. Consider thresholds midway between consecutive values
3. Compute Information Gain for each threshold
4. Pick the threshold with the highest IG

### Example — Splitting on Weight (continuous)

Sorted weights: `[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` kg

Candidate thresholds: `6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5`

For each threshold `t`: split data into `{weight < t}` and `{weight ≥ t}`, compute IG. Best threshold = the one maximising IG.

```
[Weight < 10.5 kg?]
    /          \
   YES          NO
    ↓            ↓
  lighter      heavier
  animals      animals
```

---

## 1.11 Regression Trees (Predicting Numbers)

Decision Trees work for **regression** too — not just classification.

**The only changes:**
- Leaf nodes predict the **average y value** of all training examples reaching that leaf
- Splitting criterion changes from Information Gain to **Variance Reduction**

```
Variance Reduction  =  Var(parent)  -  [ (n_left/n)*Var(left)  +  (n_right/n)*Var(right) ]
```

Choose the split that reduces variance the most — same logic as Information Gain but for continuous outputs.

---

## 1.12 Strengths and Weaknesses of a Single Decision Tree

| Strengths | Weaknesses |
|---|---|
| Highly interpretable — you can read the tree | High variance — small data changes give very different trees |
| No feature scaling needed | Prone to overfitting with deep trees |
| Handles both classification and regression | Often less accurate than ensemble methods |
| Fast to train and predict | Finds locally optimal splits — not globally optimal |
| Handles non-linear boundaries naturally | Biased toward features with more categories |

> **The high variance problem is critical.** Change a few training examples and the tree structure can change completely. This motivates Tree Ensembles.

---

# PART 2 — Tree Ensembles: Why One Tree Isn't Enough

## 2.1 The Core Problem

Andrew Ng demonstrates this powerfully:

> *"Decision Trees are very sensitive to small changes in the training data. If you change just one or two examples, you might end up with a completely different tree. This high variance makes a single Decision Tree unreliable."*

**The solution:** Don't rely on one tree. Build many trees and combine their predictions. This is the core idea of **Tree Ensembles**.

---

## 2.2 Sampling With Replacement (Bagging)

To build many different trees from the same dataset, Andrew Ng introduces **sampling with replacement** — the mechanism underlying all ensemble methods.

### The Procedure

Given a training set of `m` examples:

1. Randomly sample `m` examples **with replacement** — you can pick the same example multiple times
2. Some original examples appear multiple times, some don't appear at all
3. Train a Decision Tree on this new sample
4. Repeat `B` times → get `B` different trees

```
Original data: [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]

Bootstrap sample 1: [P1, P3, P3, P5, P7, P7, P8, P9, P10, P10]
                             ↑↑         ↑↑              ↑↑
                          duplicated  duplicated     duplicated

Bootstrap sample 2: [P2, P2, P4, P4, P5, P6, P6, P8, P9, P10]

Bootstrap sample 3: [P1, P1, P3, P5, P6, P6, P7, P8, P8, P10]
```

Each bootstrap sample is **different** → each trained Decision Tree is **different** → their errors are **different**.

> When you average many uncorrelated errors, the average error is much lower than any individual error. This is the mathematical foundation of why ensembles work.

---

# PART 3 — Random Forest

## 3.1 Core Intuition

Random Forest builds `B` decision trees using bootstrap samples. For classification, it takes a **majority vote** across all trees. For regression, it takes the **average prediction**.

> *"Even if any individual tree makes a mistake, the majority of trees will likely get it right — and the majority vote corrects individual errors."*

```
Bootstrap sample 1 → Tree 1 → Prediction: Cat
Bootstrap sample 2 → Tree 2 → Prediction: Cat
Bootstrap sample 3 → Tree 3 → Prediction: Not Cat
Bootstrap sample 4 → Tree 4 → Prediction: Cat
Bootstrap sample 5 → Tree 5 → Prediction: Cat

Majority Vote: Cat (4/5) → Final Prediction: CAT
```

---

## 3.2 The Extra Randomisation Trick

Bagging alone produces trees that are still **correlated** — if one strong feature dominates, most trees will split on it first and look similar to each other.

Andrew Ng introduces Random Forest's key innovation:

> *"At each node, instead of considering all n features for the split, randomly select only k features and find the best split among those k features."*

**Typical choice:** `k = √n` for classification, `k = n/3` for regression.

### Why This Helps

```
WITHOUT extra randomisation (Bagging only):
  Tree 1: splits on Ear → Face → Whiskers
  Tree 2: splits on Ear → Face → Whiskers   ← all trees look similar
  Tree 3: splits on Ear → Face → Whiskers   ← still correlated

WITH Random Forest randomisation:
  Tree 1: at each node, only consider {Ear, Whiskers}
          → splits on Ear → Whiskers
  Tree 2: at each node, only consider {Face, Ear}
          → splits on Face → Ear
  Tree 3: at each node, only consider {Whiskers, Face}
          → splits on Whiskers → Face
          → trees are DIVERSE!
```

By forcing trees to consider different features, they make **different types of errors**. When you combine them, their errors cancel out more effectively.

> This is the key difference between **Bagging** (just resampling data) and **Random Forest** (resampling data + randomising features at each split).

---

## 3.3 The Full Random Forest Algorithm

```
Input: Training data, number of trees B, features per split k

For b = 1 to B:
    1. Draw bootstrap sample of size m (with replacement)
    2. Build a Decision Tree on this sample:
       At each node:
           a. Randomly select k features from all n features
           b. Find the best split among these k features (using IG)
           c. Split the node
       Continue until stopping criteria met

Prediction (Classification):
    Run new example through all B trees
    Return majority vote

Prediction (Regression):
    Run new example through all B trees
    Return average prediction
```

---

## 3.4 Key Hyperparameters of Random Forest

| Hyperparameter | What It Controls | Typical Value |
|---|---|---|
| `n_estimators` (B) | Number of trees | 100 — 500 |
| `max_features` (k) | Features per split | `√n` for classification |
| `max_depth` | Maximum tree depth | None (fully grown) |
| `min_samples_split` | Min examples to split a node | 2 |
| `min_samples_leaf` | Min examples at a leaf | 1 |

> **Andrew Ng notes:** `B` (n_estimators) **does not cause overfitting** — adding more trees never hurts, it only makes the model more stable. Beyond ~100-200 trees, additional trees give diminishing returns.

---

# PART 4 — Boosting and XGBoost

## 4.1 The Limitation of Random Forest

Random Forest reduces variance by averaging uncorrelated trees. But if every individual tree has **high bias** — if each tree is a weak learner that makes systematic errors — averaging them still gives you a model with high bias.

**Boosting takes a completely different approach:**

> *"Instead of building trees independently and averaging them, build trees sequentially — each new tree focuses specifically on the mistakes that previous trees made."*

---

## 4.2 The Boosting Intuition

Andrew Ng's analogy:

> *"Think of it like a student learning from a tutor. After each practice test, the tutor identifies which problems the student got wrong and gives them more practice on exactly those problems. Over many rounds, the student gets better specifically where they are weakest."*

```
Round 1: Train Tree 1 on all data equally
         → Tree 1 makes mistakes on examples 3, 7, 9

Round 2: Train Tree 2 — give examples 3, 7, 9 HIGHER WEIGHT
         → Tree 2 focuses on those hard examples
         → Tree 2 makes mistakes on examples 1, 5

Round 3: Train Tree 3 — give examples 1, 5 HIGHER WEIGHT
         → Tree 3 focuses on those hard examples

...

Final: Weighted combination of all trees
       Trees that performed better get MORE say in the final vote
```

Each tree is built to **complement the weaknesses** of all previous trees. The ensemble progressively gets better at the hardest examples.

---

## 4.3 AdaBoost (Conceptual Foundation)

Before XGBoost, Andrew Ng covers AdaBoost to establish the boosting concept.

### AdaBoost Algorithm

1. Initialise all example weights equally: `w(i) = 1/m`
2. For `b = 1` to `B`:
   - Train a Decision Tree (often just a **stump** — depth 1) on weighted data
   - Compute weighted error: `ε_b = Σ w(i)` for misclassified examples
   - Compute tree weight: `α_b = (1/2) * ln((1 - ε_b) / ε_b)`
   - Update example weights: misclassified examples get higher weight
   - Normalise weights so they sum to 1
3. Final prediction: weighted majority vote using `α_b` as each tree's vote weight

### The Key Mechanism: Example Weight Update

```
w(i)_new  =  w(i)_old  ×  exp(-α_b)    if correctly classified   (decrease weight)
             w(i)_old  ×  exp(+α_b)    if misclassified           (increase weight)
```

- Correctly classified examples become **less important** — the model already handles them
- Misclassified examples become **more important** — the next tree must focus on them

### Tree Weight Intuition

```
α_b  =  (1/2) * ln( (1 - ε_b) / ε_b )
```

| Tree Error ε_b | α_b | Meaning |
|---|---|---|
| ε_b = 0 (perfect) | α_b = +∞ | This tree dominates everything |
| ε_b = 0.5 (random) | α_b = 0 | This tree ignored completely |
| ε_b > 0.5 (worse than random) | α_b < 0 | Flip this tree's predictions |

---

## 4.4 XGBoost: The Practical Champion

Andrew Ng introduces XGBoost (eXtreme Gradient Boosting) as the most important practical algorithm to know:

> *"XGBoost is one of the most widely used and competitive algorithms in machine learning. If you're working with structured/tabular data, XGBoost is often the first thing you should try."*

### What Makes XGBoost Different From AdaBoost

Instead of reweighting misclassified examples, XGBoost works with **residuals** — the errors of the previous ensemble — and fits new trees to those residuals directly.

---

## 4.5 Gradient Boosting: The Mathematical Foundation of XGBoost

### The Core Idea

Start with a simple initial prediction `F₀(x)` (usually the mean of y for regression). Then iteratively add trees that predict the **residual error** of the current ensemble.

### For Regression — Step by Step

**Initialise:**

```
F₀(x)  =  ȳ  =  (1/m) * Σ y(i)
```

**For b = 1 to B:**

**Step 1 — Compute residuals (errors of current ensemble):**

```
r(i)_b  =  y(i)  -  F_{b-1}(x(i))     for all i
```

**Step 2 — Fit a Decision Tree `h_b(x)` to predict the residuals** `(x(i), r(i)_b)`

**Step 3 — Update the ensemble:**

```
F_b(x)  =  F_{b-1}(x)  +  η * h_b(x)
```

Where `η` (eta) is the **learning rate** — how aggressively to trust each new tree.

---

### Numerical Example — Gradient Boosting for Regression

**Dataset:** Predicting house price (y) from size (x₁)

| House | Size x₁ | Price y |
|---|---|---|
| 1 | 1 | 3 |
| 2 | 2 | 5 |
| 3 | 3 | 7 |
| 4 | 4 | 9 |

Using `η = 0.5` and tree stumps (depth = 1).

---

#### Iteration 0 — Initial Prediction

```
F₀  =  ȳ  =  (3 + 5 + 7 + 9) / 4  =  6
```

Predict 6 for every house.

**Compute Residuals:**

| House | y | F₀ | r₁ = y − F₀ |
|---|---|---|---|
| 1 | 3 | 6 | −3 |
| 2 | 5 | 6 | −1 |
| 3 | 7 | 6 | +1 |
| 4 | 9 | 6 | +3 |

---

#### Iteration 1 — Fit Tree h₁ to Residuals

Best split on `x₁ < 2.5`:

- **Left** (Houses 1, 2): residuals = [−3, −1] → leaf prediction = `(−3 + −1) / 2 = −2`
- **Right** (Houses 3, 4): residuals = [+1, +3] → leaf prediction = `(1 + 3) / 2 = +2`

**Update Ensemble:**

```
F₁(x)  =  F₀(x)  +  0.5 × h₁(x)
```

| House | F₀ | h₁(x) | 0.5 × h₁ | F₁ | True y | New Residual |
|---|---|---|---|---|---|---|
| 1 | 6 | −2 | −1 | **5** | 3 | −2 |
| 2 | 6 | −2 | −1 | **5** | 5 | 0 |
| 3 | 6 | +2 | +1 | **7** | 7 | 0 |
| 4 | 6 | +2 | +1 | **7** | 9 | +2 |

Houses 2 and 3 are now **perfectly predicted**. The remaining residuals are smaller.

---

#### Iteration 2 — Fit Tree h₂ to New Residuals

Non-zero residuals: House 1 (−2), House 4 (+2).

Split at `x₁ < 3.5`:

- **Left** (Houses 1,2,3): non-zero residuals [−2,0,0] → leaf = `−2/3 ≈ −0.667`
- **Right** (House 4): residual [+2] → leaf = `+2`

**Update:**

```
F₂  =  F₁  +  0.5 × h₂
```

| House | F₁ | h₂ | 0.5 × h₂ | F₂ | True y |
|---|---|---|---|---|---|
| 1 | 5 | −0.667 | −0.333 | 4.667 | 3 |
| 2 | 5 | −0.667 | −0.333 | 4.667 | 5 |
| 3 | 7 | −0.667 | −0.333 | 6.667 | 7 |
| 4 | 7 | +2 | +1 | **8** | 9 |

Predictions improve with each round. Continue until residuals are negligible or max iterations reached.

---

## 4.6 What Makes XGBoost "eXtreme"

XGBoost is not just gradient boosting — it adds several critical engineering and mathematical improvements that make it dramatically faster and more accurate in practice.

---

### Improvement 1 — Second-Order Gradient (Taylor Expansion)

Standard gradient boosting uses only the **first derivative** (gradient) of the loss. XGBoost uses **both first and second derivatives** (gradient AND Hessian):

```
Standard GBM:  uses only  g_i  =  ∂L / ∂F(x(i))

XGBoost:       uses both  g_i  =  ∂L / ∂ŷ(i)
                          h_i  =  ∂²L / ∂(ŷ(i))²
```

Using the second derivative gives a better approximation of the loss function → more accurate tree fitting → fewer trees needed.

---

### Improvement 2 — Built-in Regularisation

XGBoost adds regularisation directly into the tree-building objective:

```
XGBoost Objective  =  Σ L(y(i), ŷ(i))     +     Σ Ω(h_b)
                       ──────────────────         ──────────
                       Training Loss              Regularisation
```

Where:

```
Ω(h)  =  γT  +  (1/2) * λ * Σ w_j²

  T    = number of leaf nodes  (penalises complex trees)
  w_j  = leaf weight           (penalises large leaf values)
  γ    = minimum gain to make a split
  λ    = L2 regularisation on leaf weights
```

This is the equivalent of Ridge Regression built directly into each tree — prevents overfitting without needing external cross-validation for depth control.

---

### Improvement 3 — Clever Split Finding (Approximate Algorithm)

Finding the best split requires sorting all feature values — `O(n × m log m)` per node. For millions of examples, this is slow.

XGBoost uses an **approximate algorithm** — it pre-sorts features into quantile buckets and finds the best split within buckets. Dramatically faster with minimal accuracy loss.

---

### Improvement 4 — Sparsity Awareness

Real-world data has missing values. XGBoost learns a **default direction** for each split — which branch to send missing values to — based on what minimises loss. It handles missing values natively without imputation.

---

### Improvement 5 — Column Subsampling (Like Random Forest)

XGBoost supports sampling only a fraction of features per tree — borrowing Random Forest's decorrelation trick. This adds diversity to the boosted ensemble and further reduces overfitting.

---

### Improvement 6 — Parallel and Cache-Efficient Computation

Despite being sequential (each tree depends on the previous), XGBoost parallelises the **split-finding step** within each tree. It also uses cache-aware data access patterns — the data is arranged in memory to minimise cache misses during tree construction.

---

## 4.7 The XGBoost Objective — Full Formula

The score of a tree structure in XGBoost is:

```
Score  =  -(1/2) * Σⱼ [ G_j² / (H_j + λ) ]  +  γT
```

Where:
- `G_j` = sum of first-order gradients in leaf j
- `H_j` = sum of second-order gradients (Hessians) in leaf j
- `T` = number of leaves

**Gain from splitting node j into left L and right R:**

```
Gain  =  (1/2) * [ G_L² / (H_L + λ)  +  G_R² / (H_R + λ)  -  (G_L + G_R)² / (H_L + H_R + λ) ]  -  γ
```

> The `−γ` at the end means a split is only made if the gain exceeds `γ` — this is **automatic pruning** built into the gain calculation.

**Optimal leaf weight:**

```
w*_j  =  -(G_j) / (H_j + λ)
```

The leaf value is the negative ratio of gradient sum to Hessian sum — a more principled update than simple residual averaging.

---

## 4.8 XGBoost Key Hyperparameters

| Hyperparameter | What It Controls | Typical Range |
|---|---|---|
| `n_estimators` | Number of trees B | 100 — 1000 |
| `learning_rate` (η) | Step size per tree | 0.01 — 0.3 |
| `max_depth` | Maximum tree depth | 3 — 10 |
| `subsample` | Fraction of data per tree | 0.5 — 1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.5 — 1.0 |
| `reg_lambda` | L2 regularisation on leaves | 1 (default) |
| `gamma` | Minimum gain to split | 0 — 5 |
| `min_child_weight` | Minimum Hessian sum in leaf | 1 — 10 |

### The Learning Rate — Tree Count Tradeoff

```
Small η  +  Large B   →  More accurate, slower to train
Large η  +  Small B   →  Faster, less accurate, prone to overfitting
```

Andrew Ng's practical advice:

> *"Start with a small learning rate (0.1 or less) and use early stopping — let XGBoost tell you when to stop adding trees."*

### Early Stopping

Monitor validation error after each tree. Stop when validation error stops improving:

```
Tree 1: Val error = 0.45
Tree 2: Val error = 0.38
Tree 3: Val error = 0.31
Tree 4: Val error = 0.29
Tree 5: Val error = 0.28   ← best so far
Tree 6: Val error = 0.28   ← no improvement
Tree 7: Val error = 0.29   ← getting worse

Early stopping at tree 5. Use 5 trees.
```

---

# PART 5 — Decision Trees vs Neural Networks

Andrew Ng gives the most direct and practical comparison in the entire specialisation here.

---

## When to Use Decision Trees / Tree Ensembles

### 1. Structured / Tabular Data

Decision Trees and ensembles work **extremely well** on tabular data — rows of examples with named columns. This is the data format of most real business problems: financial records, medical records, customer data, sensor readings.

```
Tabular data example:

| Age | Income | Credit Score | Default? |
|-----|--------|--------------|---------|
| 35  | 50000  | 720          | No      |
| 28  | 35000  | 650          | Yes     |
```

For this kind of data, XGBoost and Random Forest often **match or beat** Neural Networks with far less computation and tuning.

### 2. Fast Training and Iteration

Tree ensembles train in seconds to minutes on a CPU. Neural Networks need GPUs and hours or days. In a research or competition context where you're iterating quickly, trees let you test many ideas fast.

### 3. Small to Medium Datasets

Trees generalise well even with `m < 10,000` examples. Neural Networks typically need far more data to learn good representations.

### 4. Interpretability Matters

Feature importance scores, tree visualisation, SHAP values for tree models are all natural and well-developed. When you need to explain a prediction to a doctor, banker, or judge — trees are far easier to defend.

---

## When to Use Neural Networks

### 1. Unstructured Data

Images, audio, text, video — Neural Networks and specifically CNNs and Transformers dominate here. No tree-based method comes close to GPT or ResNet on their respective tasks.

### 2. Very Large Datasets

With millions of examples and hundreds of features, Neural Networks' ability to learn complex hierarchical representations gives them an edge over trees.

### 3. Transfer Learning Is Available

Pre-trained Neural Networks (BERT, ResNet, GPT) can be fine-tuned for your specific task with minimal data. No equivalent exists for trees.

### 4. Multi-modal Tasks

Tasks combining images, text, and structured data simultaneously are natural for neural architectures. Trees cannot process raw images or text.

---

## Andrew Ng's Decision Guide

```
New Task: Start Here
         ↓
Is data structured/tabular?
  YES → XGBoost or Random Forest first
        Try Neural Network if trees don't give desired accuracy
  NO (images/audio/text) → Neural Network always
         ↓
How much data?
  Small (<10k)   → Trees preferred
  Large (>100k)  → Both competitive, Neural Network may win
         ↓
Need to iterate fast?
  YES → Trees (faster training)
  NO  → Both viable
         ↓
Transfer learning available?
  YES → Neural Network (fine-tune pretrained models)
  NO  → Trees competitive on tabular data
```

---

# PART 6 — The Full Picture: Decision Tree → XGBoost

```
DECISION TREE
  → Splits data using Information Gain (entropy reduction)
  → Builds recursively until stopping criteria
  → High variance — sensitive to data changes
          ↓
  PROBLEM: One tree is unreliable
          ↓
  SOLUTION: Combine many trees
          ↓

BAGGING (Bootstrap Aggregating)
  → Create B bootstrap samples (sampling with replacement)
  → Train B independent trees
  → Average predictions (regression) or majority vote (classification)
  → Reduces variance — trees are diverse due to different samples
          ↓

RANDOM FOREST  =  BAGGING  +  Feature Randomisation
  → At each split, consider only k = √n features (not all n)
  → Trees are further decorrelated → better variance reduction
  → Most robust tree ensemble for general use
          ↓

BOOSTING: A completely different approach
  → Build trees sequentially
  → Each tree focuses on the mistakes of all previous trees
  → Reduces both bias AND variance
          ↓

GRADIENT BOOSTING
  → Fits each new tree to the RESIDUALS of the current ensemble
  → Updates: F_b(x) = F_{b-1}(x) + η × h_b(x)
          ↓

XGBOOST  =  Gradient Boosting  +  Engineering Excellence
  → Second-order gradients (Hessian) for better tree fitting
  → Built-in L2 regularisation on leaf weights
  → Approximate split finding for speed
  → Native missing value handling
  → Column subsampling like Random Forest
  → Parallelised computation
  → Most powerful tree-based algorithm in practice
```

---

# Key Takeaways — Everything Condensed

## Decision Tree

- Splits on the feature that maximises **Information Gain** = `H(parent) − weighted avg H(children)`
- **Entropy** `H(p) = -p log₂(p) - (1-p) log₂(1-p)` measures impurity (0 = pure, 1 = maximally mixed)
- Stopping criteria: pure node, max depth, min IG threshold, min samples
- **High variance** — single trees are unreliable → motivates ensembles

## Random Forest

- **Bagging** = bootstrap sampling + train independent trees + average
- **Extra randomisation:** consider only `k = √n` features per split → decorrelates trees
- Reduces variance without increasing bias
- More trees never hurts — only diminishing returns after ~200

## XGBoost

- **Sequential boosting** — each tree corrects previous ensemble's residuals
- **Second-order gradients + built-in regularisation** = better generalisation than standard GBM
- **Learning rate η + n_estimators** are the most critical hyperparameters (use together)
- **Use early stopping** to find optimal number of trees automatically
- The dominant algorithm for structured/tabular data in research and industry

## Trees vs Neural Networks

| Scenario | Best Choice |
|---|---|
| Structured/tabular data | XGBoost or Random Forest first |
| Images / Audio / Text | Neural Networks always |
| Small data (< 10k samples) | Trees preferred |
| Transfer learning available | Neural Networks |
| Need interpretability | Trees (with SHAP for black-box ensembles) |
| Need to iterate fast | Trees (CPU-trainable in seconds) |

---
