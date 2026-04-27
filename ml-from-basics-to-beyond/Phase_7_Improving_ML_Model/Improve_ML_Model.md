# Course 2 Week 3 — Advice for Applying Machine Learning
*Reference: Andrew Ng's ML Specialization, Course 2 — Advanced Learning Algorithms*

---

## The Big Picture — Why This Week is Critical

> *"It's not enough to know how to implement ML algorithms. You need to know how to make good decisions about what to try next when your model isn't working. Without this, you might waste months going in the wrong direction."*

This week is entirely about **systematic thinking** — how to diagnose what is wrong with your model and how to fix it efficiently. Every concept here applies to every algorithm you have learned so far and every algorithm you will learn in the future.

```
The Problem Most Beginners Face:
Model doesn't work → Try random things → Waste months

The Systematic Approach (This Week):
Model doesn't work → Diagnose the problem → Apply the right fix
```

---

# PART 1 — Evaluating a Model

## 1.1 — Why You Cannot Trust Training Error Alone

Consider this situation. You train a Logistic Regression model to classify tumors. After training, you check the cost function — it's very low. You check predictions on the training data — 99% accuracy. You conclude: great model.

Then you deploy it on new patients. It performs terribly.

What went wrong? **The model memorized the training data but failed to generalize.** Evaluating only on training data tells you nothing about real-world performance.

Andrew Ng's fundamental principle:

> *"The goal of a learning algorithm is to generalize to new examples it hasn't seen before — not to perform well on data it was trained on."*

---

## 1.2 — The Train / Test Split

The simplest solution: split your data into two sets before training.

$$\text{Training Set (70-80\%)} \quad + \quad \text{Test Set (20-30\%)}$$

```
Full Dataset (100 examples)
        ↓
┌──────────────────────┬──────────────┐
│   Training Set (80)  │  Test Set(20)│
│   Model learns here  │ Evaluate here│
└──────────────────────┴──────────────┘
```

**Process:**
1. Train model on training set — find optimal $\vec{w}$, $b$
2. Evaluate model on test set — compute test error
3. Test error tells you how well the model generalizes

### Computing Train and Test Errors

**For Regression** (MSE):

$$J_{\text{train}}(\vec{w},b) = \frac{1}{2m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$$

$$J_{\text{test}}(\vec{w},b) = \frac{1}{2m_{\text{test}}}\sum_{i=1}^{m_{\text{test}}}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$$

**For Classification** (fraction misclassified):

$$J_{\text{test}} = \frac{1}{m_{\text{test}}}\sum_{i=1}^{m_{\text{test}}} \mathbf{1}\{\hat{y}^{(i)} \neq y^{(i)}\}$$

Where $\mathbf{1}\{\cdot\}$ is the indicator function — equals 1 if the condition is true, 0 otherwise.

### What Train vs Test Error Tells You

| Scenario | Training Error | Test Error | Interpretation |
|---|---|---|---|
| Good model | Low | Low | Generalizes well |
| Overfitting | Low | **High** | Memorized training data |
| Underfitting | High | High | Model too simple |

---

## 1.3 — The Problem With Two-Way Split: Model Selection

Suppose you want to choose between three models:

- Model 1: Polynomial degree $d=1$ (linear)
- Model 2: Polynomial degree $d=2$ (quadratic)
- Model 3: Polynomial degree $d=3$ (cubic)

You train all three on the training set, evaluate on the test set, and pick the one with the lowest test error — say Model 2.

**The subtle problem:** By using the test set to choose between models, you have now "used up" the test set. The test error of your chosen model is now an **optimistic estimate** of real-world performance — you selected the model specifically because it performed well on this test set.

Andrew Ng explains:

> *"If you use the test set to make decisions about the model, the test set is no longer a fair measure of generalization. You've indirectly fit to the test set."*

The solution is a **three-way split**.

---

## 1.4 — Train / Validation / Test Split (The Standard)

$$\text{Training Set} \approx 60\% \quad + \quad \text{Validation Set} \approx 20\% \quad + \quad \text{Test Set} \approx 20\%$$

```
Full Dataset
        ↓
┌────────────────────┬─────────────┬─────────────┐
│   Training Set     │  Validation │   Test Set  │
│   (60%)            │  Set (20%)  │   (20%)     │
│  Learn w, b        │ Select model│ Final report│
└────────────────────┴─────────────┴─────────────┘
```

| Set | Also Called | Purpose |
|---|---|---|
| Training Set | Train set | Learn parameters $\vec{w}$, $b$ |
| Validation Set | Dev set, Cross-validation set | Select model, tune hyperparameters |
| Test Set | Hold-out set | Final unbiased performance estimate |

### The Three-Step Process

```
Step 1 — TRAIN:
    Train all candidate models on Training Set
    → Each model learns its own w*, b*

Step 2 — VALIDATE:
    Evaluate all models on Validation Set
    → Pick model with lowest validation error
    → This is where model selection happens

Step 3 — TEST:
    Evaluate chosen model on Test Set ONCE
    → This is your final, honest performance report
    → Never use this number to make further decisions
```

### Numerical Example — Selecting Polynomial Degree

Training set: 60 houses. Validation: 20 houses. Test: 20 houses.

| Model | $d$ | $J_{\text{train}}$ | $J_{\text{val}}$ |
|---|---|---|---|
| 1 | 1 | 0.45 | 0.48 |
| 2 | 2 | 0.20 | 0.22 ← |
| 3 | 3 | 0.10 | 0.35 |
| 4 | 4 | 0.05 | 0.60 |

Model 2 (degree=2) has lowest validation error → **select Model 2**.

Now evaluate Model 2 on test set: $J_{\text{test}} = 0.24$

This 0.24 is your **honest estimate** of generalization performance. Report this number.

---
---

# PART 2 — Diagnosing Bias and Variance

## 2.1 — The Most Important Diagnostic in ML

Andrew Ng calls the bias-variance diagnosis the single most important skill for applying ML:

> *"If your model isn't working well, almost always the root cause is either high bias or high variance. Once you correctly identify which one, the fix becomes clear."*

---

## 2.2 — High Bias vs High Variance — The Definitions

### High Bias (Underfitting)

The model is **too simple** to capture the underlying pattern in the data. It misses important structure — it's biased toward a simple solution.

```
Price
  |  * *          ← data follows curve
  |       * *
  |           * *
  |_______________ ← model: straight line, misses curve
+------------------→ Size

High Bias:
- Training error HIGH
- Validation error HIGH
- Both are similarly bad
```

### High Variance (Overfitting)

The model is **too complex** — it learns the noise in the training data, not the true pattern. Performance varies wildly between training and new data.

```
Price
  |  * *
  |*      *  *
  |   *       *  * ← model: wiggles through every point
  |               *
+------------------→ Size

High Variance:
- Training error LOW
- Validation error HIGH
- Big gap between the two
```

### Just Right

```
Price
  |  * *
  |       * *       ← smooth curve fits data well
  |  ~curve~
  |           * *
+------------------→ Size

Good model:
- Training error LOW
- Validation error LOW (and close to training error)
```

---

## 2.3 — The Bias-Variance Diagnostic Using Error Numbers

The relationship between training error and validation error tells you exactly what is wrong:

$$\boxed{J_{\text{train}} \text{ and } J_{\text{val}} \text{ together reveal the problem}}$$

| $J_{\text{train}}$ | $J_{\text{val}}$ | Diagnosis | Problem |
|---|---|---|---|
| High | High | **High Bias** | Underfitting |
| Low | High | **High Variance** | Overfitting |
| High | Low | Impossible | (shouldn't happen) |
| Low | Low | **Just Right** | Good model |
| High | Very High | **Both** | High Bias AND High Variance |

### Concrete Numbers — House Price Prediction

**Scenario A:**
$$J_{\text{train}} = 0.45, \quad J_{\text{val}} = 0.48$$

Both high → **High Bias** → model underfits → need more complex model.

**Scenario B:**
$$J_{\text{train}} = 0.05, \quad J_{\text{val}} = 0.55$$

Training low, validation much higher → **High Variance** → model overfits → need regularization or more data.

**Scenario C:**
$$J_{\text{train}} = 0.08, \quad J_{\text{val}} = 0.10$$

Both low and close → **Good model** → neither significantly biased nor overfitting.

**Scenario D:**
$$J_{\text{train}} = 0.40, \quad J_{\text{val}} = 0.60$$

Training high AND validation much higher still → **Both high bias and variance** → model underfits on training but also overfits the training noise.

---

## 2.4 — How Model Complexity Affects Bias and Variance

Using polynomial degree $d$ as a proxy for model complexity:

```
Error
  |
  |*  ← J_val starts high (high bias, d=1 too simple)
  | *  *
  |     *  ← J_val minimum (sweet spot)
  |        *  *  *  *  ← J_val rises again (high variance)
  |
  |_ _ _ _ _ *  *  *  *  ← J_train keeps falling as d increases
  |
  +----------------------------→ Polynomial degree d
  1    2    3    4    5    6

                ↑
         Optimal complexity
```

Key observations:
- $J_{\text{train}}$ **always decreases** as model complexity increases — more complex models fit training data better
- $J_{\text{val}}$ follows a **U-shape** — first decreases (reducing bias), then increases (increasing variance)
- The optimal model sits at the bottom of the validation curve

---

## 2.5 — Regularization and Bias-Variance

The regularization parameter $\lambda$ has the **opposite** effect on bias and variance compared to model complexity.

### How $\lambda$ Affects Errors

**Very large $\lambda$** (e.g., $\lambda = 10000$):
- All weights forced near zero
- Model essentially predicts a constant
- **High Bias** — ignores features completely

**Very small $\lambda$** (e.g., $\lambda = 0.0001$):
- Almost no regularization
- Model fits training data too tightly
- **High Variance** — overfitting

**Just right $\lambda$**:
- Balanced constraints on weights
- **Good generalization**

```
Error
  |
  |*  ← Large λ: high bias (J_train AND J_val both high)
  | *
  |  *   ← Sweet spot λ (J_val minimum)
  |   *  *  *
  |         *  *  *  ← Small λ: high variance (J_val rises)
  |
  |_ _ _ _ * _ *  *  ← J_train keeps decreasing as λ→0
  |
  +----------------------------→ λ (regularization strength)
  large                        small
```

Finding the right $\lambda$: try a range of values ($0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10$), evaluate each on the validation set, pick the $\lambda$ with lowest $J_{\text{val}}$.

---

## 2.6 — Establishing a Baseline Level of Performance

Before deciding if your model has high bias or high variance, you need to know what **"good performance"** looks like for your problem.

Andrew Ng introduces the concept of a **baseline** — a reference point against which to compare your errors.

Three common baselines:

| Baseline Type | How to Get It | Example |
|---|---|---|
| **Human level performance** | Have humans do the task | 98% accuracy on image classification |
| **Competing algorithm** | Previous known methods | 85% F1 on similar text classification |
| **Simple heuristic** | Naive rule-based approach | "Always predict majority class" = 70% |

### Why Baseline Matters — Concrete Example

Speech recognition model:
- $J_{\text{train}} = 10.8\%$
- $J_{\text{val}} = 14.8\%$

Without baseline: looks like there's a 4% gap → might assume high variance.

With baseline:
- Human level error = 10.6%

$$\text{Bias gap} = J_{\text{train}} - \text{Baseline} = 10.8\% - 10.6\% = 0.2\%$$
$$\text{Variance gap} = J_{\text{val}} - J_{\text{train}} = 14.8\% - 10.8\% = 4.0\%$$

Now it's clear: bias is nearly zero (0.2% above human), variance is the real problem (4% gap). Without the baseline, you might have incorrectly tried to reduce bias.

### The Three-Way Gap Analysis:

```
Human Level Performance: 10.6%
         ↕  0.2% gap (Avoidable Bias)
Training Error: 10.8%
         ↕  4.0% gap (Variance)
Validation Error: 14.8%
```

**Avoidable Bias** = $J_{\text{train}}$ − Baseline → how much better the training performance could be.
**Variance** = $J_{\text{val}}$ − $J_{\text{train}}$ → how much performance drops on unseen data.

Focus your efforts on whichever gap is larger.

---
---

# PART 3 — Learning Curves

## 3.1 — What Are Learning Curves?

A **Learning Curve** plots $J_{\text{train}}$ and $J_{\text{val}}$ as a function of **training set size** $m_{\text{train}}$.

You deliberately train the model on increasing amounts of data (10 examples, 20 examples, 50 examples, ...) and record both training and validation error at each size.

This reveals how your model behaves as data grows — one of the most informative diagnostics available.

---

## 3.2 — Learning Curves for High Bias Model

```
Error
  |
  |         ← J_val (starts high, decreases, then flattens)
  |*
  | * * *
  |       * * * * * ─────────── ← J_val flattens (high and flat)
  |
  |             * * * * * ────── ← J_train rises and flattens
  |       * * *                    (close to J_val — both bad)
  |* * *
  +--------------------------------→ Training Set Size m
  10  20  50  100  200  500  1000

Gap between J_train and J_val is SMALL
Both flatten at HIGH error → high bias confirmed
```

### The Critical Insight About High Bias and Data

> *"If your model has high bias, getting more training data will NOT help. The curves have already flattened — more data won't push them down."*

A high-bias model is fundamentally too simple. No amount of additional data will make a linear model fit a cubic relationship. The fix must be a **more complex model**.

---

## 3.3 — Learning Curves for High Variance Model

```
Error
  |
  |*  ← J_val starts very high (big gap from J_train)
  | *
  |  *  *
  |      *  *  *  ← J_val still decreasing — hasn't flattened yet!
  |
  |                    *  ← J_train rises gradually
  |           *  *
  |* * *
  +--------------------------------→ Training Set Size m
  10  20  50  100  200  500  1000

Gap between J_train and J_val is LARGE
J_val hasn't flattened → more data would still help
```

### The Critical Insight About High Variance and Data

> *"If your model has high variance, getting more training data WILL help — as long as the validation curve hasn't flattened yet. More data reduces the gap between training and validation performance."*

With a high-variance model, the model has learned the noise in the training set. More data dilutes the noise and forces the model to learn the true signal. The validation curve is still declining — there's room to improve.

---

## 3.4 — The Learning Curve Diagnostic Summary

| Shape of Curves | Diagnosis | Will More Data Help? |
|---|---|---|
| Both flat at high error, small gap | High Bias | No |
| Large gap, val curve still declining | High Variance | Yes |
| Both low, small gap | Good model | Marginally |

---
---

# PART 4 — Deciding What to Try Next

## 4.1 — The Systematic Fix List

This is one of the most practically valuable sections Andrew Ng teaches. Given a diagnosis, what exactly should you do?

```
Your Model Isn't Performing Well
            ↓
    ┌───────────────────┐
    │ Diagnose:         │
    │ High Bias or      │
    │ High Variance?    │
    └───────┬───────────┘
            ↓
    ┌───────┴──────────────────────┐
    │                              │
HIGH BIAS                   HIGH VARIANCE
(J_train high)             (J_train low,
    │                        J_val high)
    ↓                              ↓
• Get more features           • Get more training data
• Add polynomial features     • Reduce model complexity
• Decrease λ (less reg.)      • Increase λ (more reg.)
• Use more complex model      • Feature selection (use fewer)
• Train longer                • Early stopping
```

---

## 4.2 — Mapping Fixes to the Bias-Variance Problem

Andrew Ng walks through each common "fix" and shows exactly which problem it addresses:

| Action | Fixes High Bias? | Fixes High Variance? |
|---|---|---|
| Get more training data | ✗ No | ✓ Yes |
| Add more features | ✓ Yes | ✗ Makes worse |
| Add polynomial features | ✓ Yes | ✗ Makes worse |
| Decrease $\lambda$ | ✓ Yes | ✗ Makes worse |
| Increase $\lambda$ | ✗ Makes worse | ✓ Yes |
| Use a more complex model | ✓ Yes | ✗ Makes worse |
| Feature selection (fewer) | ✗ Makes worse | ✓ Yes |
| Early stopping | ✗ No | ✓ Yes |

Without this map, you might waste months adding features to an overfitting model, or collecting data for an underfitting model. With the diagnosis-fix map, every action you take is purposeful.

---

## 4.3 — Neural Networks and the Bias-Variance Tradeoff

Andrew Ng introduces an important insight specific to Neural Networks:

> *"With Neural Networks, you have more freedom. If your model has high bias, you can simply make the network bigger — more layers, more units. If it has high variance, you can add regularization or get more data. And importantly — a larger neural network with regularization almost never does worse than a smaller network."*

### The Neural Network Approach to Bias-Variance:

```
Train Neural Network
        ↓
High bias (J_train high)?
    YES → Make the network BIGGER (more layers/units)
    NO  ↓
High variance (J_val >> J_train)?
    YES → Get more data OR add regularization (dropout, L2)
    NO  ↓
Done — good model
```

This is fundamentally different from classical ML where you had to carefully choose model complexity. Large neural networks with regularization give you a path to reduce both bias and variance simultaneously — at the cost of more computation.

---
---

# PART 5 — Error Analysis

## 5.1 — The Process

After diagnosing bias vs variance at the aggregate level, **Error Analysis** gives you a fine-grained understanding of *which specific types of examples* your model struggles with.

Andrew Ng describes this process as:

> *"Manually examine the examples your model misclassifies and look for patterns. This often reveals what to focus on next better than any automated analysis."*

### The Process

```
Step 1: Take your validation set
Step 2: Find all misclassified examples
Step 3: Manually examine them
Step 4: Tag each one with the type of error
Step 5: Count errors per category
Step 6: Focus on the highest-impact category
```

---

## 5.2 — Numerical Example — Email Spam Classifier

Your spam classifier has 500 validation examples. It misclassifies **100 of them** (20% error rate).

You manually examine all 100 misclassified emails and tag them:

| Error Category | Count | % of Errors | % of Total Val Set |
|---|---|---|---|
| Pharma spam (unusual words) | 21 | 21% | 4.2% |
| Misspelled words (d1scounts) | 36 | 36% | 7.2% |
| Phishing emails | 18 | 18% | 3.6% |
| Embedded image spam | 25 | 25% | 5.0% |

### What Error Analysis Tells You

If you could **perfectly fix** each category, you would eliminate those errors:

| Fix | Error Reduction |
|---|---|
| Fix pharma spam | 100% → 79% fixed → 20% → ~16% error rate |
| Fix misspelled words | **Most impactful** → 20% → ~12.8% error rate |
| Fix embedded image spam | 20% → ~15% error rate |

**Conclusion:** Focus on misspelled words first — it addresses 36% of your errors. Building a better misspelling handler or adding character-level features would give the biggest improvement.

Without error analysis, you might have worked on pharma spam (only 21% of errors) instead of the more impactful misspelling problem.

---
---

# PART 6 — Data-Centric Approaches

## 6.1 — Adding Data: The Approaches

When you identify high variance and decide to get more data, Andrew Ng gives three strategies:

### Strategy 1 — Collect More of Everything

Simply gather more labeled examples. This is the most straightforward but often most expensive approach — data collection and labeling is slow and costly.

### Strategy 2 — Targeted Data Collection

Based on your **error analysis**, collect more data specifically for the categories where your model struggles.

```
Error analysis showed: 36% of errors are misspelled words
        ↓
Targeted collection: gather more emails with intentional
misspellings and deliberate obfuscation
        ↓
More efficient than random collection:
you're adding data precisely where the model is weakest
```

This is far more efficient than collecting random new data — you're directly addressing your identified weaknesses.

### Strategy 3 — Data Augmentation

**Create new training examples** by applying transformations to existing data.

For images:
- Rotate, flip, zoom, crop
- Adjust brightness, contrast
- Add noise

For audio:
- Add background noise (cafe sounds, car noise)
- Change speed slightly
- Shift pitch

For text:
- Synonym replacement
- Back-translation (English → French → English)

```
Original image of "A":
    A

Augmented versions:
    A   A   A   𝐀   A   (bold)
   (rotated)(zoomed)(noisy)(flipped)

All augmented versions still labeled as "A"
→ Model sees many more examples without new data collection
```

**Critical rule:** Data augmentation should produce examples that look like **realistic test examples**. Distorting images beyond recognition or adding random noise that never appears in real data will hurt performance.

---

## 6.2 — Data Synthesis

For some tasks, you can **generate entirely new synthetic data**:

**OCR (reading text in images):**
- Take computer fonts
- Render text in many fonts, sizes, backgrounds
- Synthesize images of characters

This was used historically to create large OCR training datasets before massive real-world data was available. Modern applications include generating synthetic financial transactions for fraud detection or synthetic medical images.

Andrew Ng's point:

> *"Don't think only about model-centric improvements — trying better algorithms, better hyperparameters. Sometimes the most impactful thing you can do is improve the quality or quantity of your data."*

---
---

# PART 7 — Transfer Learning

## 7.1 — The Core Idea

Andrew Ng introduces Transfer Learning as one of the most powerful practical ideas in modern ML:

> *"Instead of training a model from scratch on your specific task, start with a model already trained on a large related dataset. This pretrained model has already learned many useful patterns — fine-tune it on your data."*

```
Standard approach (training from scratch):
Random initialization → Train on your data → Model

Transfer Learning:
Pretrained model (trained on massive dataset)
        ↓
Keep early layers (general features already learned)
        ↓
Replace/fine-tune final layers for your specific task
        ↓
Train on your (smaller) dataset
→ Faster, better performance with less data
```

---

## 7.2 — Why Transfer Learning Works

Neural Networks trained on large datasets learn **hierarchical feature representations**:

```
Image Classification Network:
Layer 1: Learns edges, corners (very general)
Layer 2: Learns shapes, textures
Layer 3: Learns object parts (wheels, eyes, windows)
Layer 4: Learns high-level concepts (faces, cars)
Layer 5: Final classification
```

These early layers learn features that are useful for **many different tasks** — not just the original training task. A network trained on ImageNet (1 million images, 1000 categories) has learned incredibly rich image representations that transfer to any image task.

For your specific task (say, detecting tumors from X-rays), you only have 1000 labeled images. If you train from scratch, you'll overfit badly. But if you start from ImageNet weights, the early layers already know how to detect edges, shapes, and textures — you just need to fine-tune the final layers to recognize tumor-specific patterns.

---

## 7.3 — The Two Transfer Learning Approaches

### Approach 1 — Fine-tuning All Layers (When you have enough data)

```
Pretrained model weights → Initialize all layers
        ↓
Train ALL layers on your dataset
        ↓
Early layers update slightly (already good)
Later layers update more (specific to your task)
```

Use this when you have a **reasonably large dataset** (thousands of examples) and the target task is somewhat different from the pretrained task.

### Approach 2 — Fine-tuning Only the Last Layer(s) (When data is very limited)

```
Pretrained model weights → Initialize all layers
        ↓
FREEZE all early layers (do not update)
        ↓
Train ONLY the last 1-2 layers on your dataset
        ↓
Only final classification layer adapts to your task
```

Use this when you have **very limited data** (hundreds of examples). Training all layers on tiny data would overfit immediately. Frozen early layers provide a fixed rich feature extractor.

```
Transfer Learning Strategy:

Small dataset:                Large dataset:
Freeze most layers            Fine-tune all layers
Train only head               But with small learning rate
→ Avoids overfitting          for early layers
```

---

## 7.4 — Concrete Example — Medical Imaging

You want to detect diabetic retinopathy from eye scans.
Your dataset: 2000 labeled images.

**Without transfer learning:**
Train from scratch on 2000 images → overfits → 72% accuracy.

**With transfer learning:**
Start from ResNet pretrained on ImageNet (1.2M images).
Fine-tune only the last 2 layers on your 2000 images.
Early layers already know edges, textures, shapes.
→ 94% accuracy.

The pretrained model's knowledge of "what images look like" transfers even though ImageNet has cats and cars, not eye scans.

---

## 7.5 — Why Transfer Learning Matters for Your Research Path

Andrew Ng emphasizes:

> *"For most practical applications, you will almost never train a large neural network from scratch. You'll start from a pretrained model. Knowing when and how to apply transfer learning is one of the most important skills in applied ML."*

Popular pretrained models:

| Domain | Pretrained Model | Pretrained On |
|---|---|---|
| Images | ResNet, VGG, EfficientNet | ImageNet |
| Text | BERT, GPT, RoBERTa | Large text corpora |
| Audio | Wav2Vec, Whisper | Audio datasets |
| Multi-modal | CLIP, DALL-E | Image-text pairs |

---
---

# PART 8 — The Full Cycle of a Machine Learning Project

## 8.1 — The Iterative Loop

Andrew Ng introduces the complete lifecycle of an ML project — not just model training, but everything before and after:

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  1. DEFINE THE PROJECT                                 │
│     → What is the problem?                            │
│     → What data is available?                         │
│     → What does success look like?                    │
│                    ↓                                   │
│  2. COLLECT AND PREPARE DATA                          │
│     → Gather raw data                                 │
│     → Label data (supervised learning)                │
│     → Feature engineering                             │
│                    ↓                                   │
│  3. TRAIN THE MODEL                                   │
│     → Choose algorithm                                │
│     → Train on training set                          │
│     → Evaluate: bias? variance?                       │
│                    ↓                                   │
│  4. ERROR ANALYSIS & IMPROVEMENT                      │
│     → Diagnose failure modes                          │
│     → Add data, tune hyperparameters, change model    │
│                    ↓                                   │
│  5. DEPLOY & MONITOR                                  │
│     → Serve predictions to users                      │
│     → Monitor for performance degradation             │
│     → Collect new data from deployment                │
│                    ↑                                   │
│         ← ← ← ← ← ← ← ← ← ← ← ← ←                 │
│              (Iteration loop)                          │
└────────────────────────────────────────────────────────┘
```

This is not a one-time process — it is a **continuous loop**. Real ML systems are never "done" — they degrade as the world changes (data drift), as user behavior changes, and as new edge cases are discovered in deployment.

---

## 8.2 — Deployment Considerations

Andrew Ng highlights key issues that arise specifically in deployment:

### Concept Drift and Data Drift

**Data Drift:** The distribution of inputs $x$ changes over time.
- Example: A speech recognition model trained on clear audio suddenly receives more calls from noisy environments.

**Concept Drift:** The relationship between inputs and outputs changes.
- Example: Credit scoring model trained before a recession — the meaning of "creditworthy" changes during economic shifts.

```
Training data distribution:    Deployed data distribution:
    x ~ P_train(x)                 x ~ P_deploy(x)
                                        ↑
                           If these diverge → performance degrades
```

**Solution:** Monitor model performance continuously. Set up triggers to retrain when performance drops below a threshold.

---
---

# PART 9 — Skewed Datasets: Precision, Recall, and F1

## 9.1 — The Problem With Accuracy on Skewed Datasets

This is one of Andrew Ng's most important practical points:

> *"Accuracy is misleading when classes are heavily imbalanced. A model that predicts the majority class every time can achieve 99% accuracy while being completely useless."*

### The Extreme Example

Cancer detection dataset: 99% of patients are healthy, 1% have cancer.

A model that **always predicts "No Cancer"**:

$$\text{Accuracy} = \frac{99}{100} = 99\%$$

This model is completely useless — it detects zero cancer cases. Yet it has 99% accuracy. Accuracy is a terrible metric here.

---

## 9.2 — The Confusion Matrix

For binary classification, four outcomes are possible:

```
                    Predicted: Positive    Predicted: Negative
Actual: Positive  |  True Positive (TP)  |  False Negative (FN)  |
Actual: Negative  |  False Positive (FP) |  True Negative (TN)   |
```

| Term | Abbreviation | Medical Meaning |
|---|---|---|
| **True Positive** | TP | Predicted cancer, actually has cancer ✓ |
| **True Negative** | TN | Predicted no cancer, actually no cancer ✓ |
| **False Positive** | FP | Predicted cancer, actually no cancer ✗ (Type I error) |
| **False Negative** | FN | Predicted no cancer, actually has cancer ✗ (Type II error) |

### Numerical Example — Confusion Matrix

Out of 1000 test patients (10 actually have cancer, 990 don't):

| | Predicted: Cancer | Predicted: No Cancer |
|---|---|---|
| **Actually: Cancer** | TP = 8 | FN = 2 |
| **Actually: No Cancer** | FP = 30 | TN = 960 |

Accuracy = $\frac{TP + TN}{m} = \frac{8+960}{1000} = 96.8\%$ — looks good but misses 2 cancer cases!

---

## 9.3 — Precision and Recall

### Precision

$$\boxed{\text{Precision} = \frac{TP}{TP + FP}}$$

"Of all the cases I predicted as Positive — what fraction actually were Positive?"

Precision measures: **when you say yes, how often are you right?**

$$\text{Precision} = \frac{8}{8+30} = \frac{8}{38} \approx 0.211$$

Only 21.1% of predicted cancers are real. Many false alarms.

### Recall (Sensitivity)

$$\boxed{\text{Recall} = \frac{TP}{TP + FN}}$$

"Of all the actual Positive cases — what fraction did I correctly identify?"

Recall measures: **of all the real positives, how many did you catch?**

$$\text{Recall} = \frac{8}{8+2} = \frac{8}{10} = 0.800$$

Caught 80% of actual cancer cases. Missed 20%.

### The Precision-Recall Tradeoff

These two metrics **trade off against each other**:

**High threshold** (predict cancer only when very confident):
- Fewer false alarms → **Higher Precision**
- Miss more true cases → **Lower Recall**

**Low threshold** (predict cancer whenever slightly suspicious):
- Catch more true cases → **Higher Recall**
- More false alarms → **Lower Precision**

```
Threshold effect on Precision and Recall:

Precision
  |  * ← high threshold (very selective)
  |    *
  |      *
  |        *
  |          * ← low threshold (very permissive)
  +──────────────→ Recall
  0              1

Moving right along the curve = lower threshold
= more positive predictions
= higher recall, lower precision
```

### Choosing the Threshold

**For cancer detection:** You want **high recall** — missing a cancer case is far worse than a false alarm. You can always do further testing to rule out false positives.

**For email spam:** You want **high precision** — moving a legitimate email to spam is worse than letting some spam through.

The choice of threshold is a **business/domain decision**, not a purely mathematical one.

---

## 9.4 — The F1 Score: Combining Precision and Recall

When you need a single metric that balances both precision and recall, use the **F1 Score** — the harmonic mean of precision and recall:

$$\boxed{F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2PR}{P+R}}$$

### Why Harmonic Mean, Not Arithmetic Mean?

Arithmetic mean of (Precision=0.9, Recall=0.1): $\frac{0.9+0.1}{2} = 0.5$ — looks decent.

Harmonic mean: $F_1 = \frac{2 \times 0.9 \times 0.1}{0.9+0.1} = \frac{0.18}{1.0} = 0.18$ — correctly reflects that recall is terrible.

The harmonic mean **punishes extreme imbalance between P and R**. It forces both to be high to get a high F1 score.

### Numerical Example

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Always predict positive | 0.01 | 1.00 | 0.020 |
| Always predict negative | — | 0.00 | 0.000 |
| Our model | 0.211 | 0.800 | 0.336 |
| Improved model | 0.650 | 0.750 | 0.697 |
| Goal | >0.8 | >0.8 | >0.8 |

F1 correctly identifies that "always predict positive" has terrible precision despite perfect recall.

### When to Use Which Metric

| Metric | Use When |
|---|---|
| **Accuracy** | Classes balanced, all errors equal cost |
| **Precision** | False positives are costly (spam, fraud alerts) |
| **Recall** | False negatives are costly (cancer, safety systems) |
| **F1 Score** | Need balance of both, single metric required |

---
---

# PART 10 — The Complete Diagnostic Flowchart

Andrew Ng's complete decision process for improving any ML model:

```
Train your model
        ↓
Compute J_train, J_val, J_test
        ↓
        ┌─────────────────────────────────────────────┐
        │                                             │
   J_train HIGH?                              J_train LOW?
        │                                             │
        ↓                                             ↓
   HIGH BIAS                               J_val >> J_train?
        │                                      │         │
   FIXES:                                    YES         NO
   • More features                            │          │
   • Polynomial features                      ↓          ↓
   • Decrease λ                         HIGH VARIANCE  GOOD MODEL
   • Bigger model                            │
   • Train longer                       FIXES:
                                        • More data
                                        • Fewer features
                                        • Increase λ
                                        • Dropout
                                        • Early stopping
        ↓
Are errors on skewed classes?
        ↓
Use Precision, Recall, F1 instead of Accuracy
        ↓
Error analysis: which categories of errors dominate?
        ↓
Targeted data collection or model improvement
for the highest-impact error category
        ↓
Consider Transfer Learning if:
        • Similar task has large pretrained model
        • Your data is limited
        ↓
Deploy model + monitor for data/concept drift
        ↓
Iterate
```

---

# Key Takeaways

**Evaluation:**
- Never evaluate on training data alone
- Train/Val/Test split — use validation for model selection, test set once at the end
- Training error low, val error high → overfitting. Both high → underfitting.

**Bias-Variance:**
- High Bias → model too simple → more complexity, more features, lower λ
- High Variance → model too complex → more data, regularization, fewer features
- Always establish a **baseline** before interpreting your errors
- Learning curves reveal whether more data will help (high variance) or not (high bias)

**Practical Improvements:**
- Error analysis reveals which specific mistakes to prioritize
- Targeted data collection beats random data collection
- Data augmentation synthetically expands your training set

**Transfer Learning:**
- Start from pretrained models — almost always beats training from scratch
- Fine-tune all layers for large datasets; freeze early layers for small datasets

**Skewed Datasets:**
- Accuracy is meaningless on imbalanced classes
- Use Precision (FP cost), Recall (FN cost), F1 (balance of both)
- Threshold choice is a domain decision, not a math decision
