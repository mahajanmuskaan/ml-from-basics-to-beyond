# Part 2 — Anomaly Detection

## 2.1 — Core Intuition

> *"You run a data center with 10,000 servers. Occasionally a server starts behaving strangely — maybe it's about to fail, or it's been hacked. You want to automatically flag unusual servers before something goes wrong."*
> — Andrew Ng

Most servers behave normally — predictable CPU usage, memory, network traffic. An anomalous server looks different from all the others. You can't label all 10,000 servers manually. But you can learn what "normal" looks like and flag anything that deviates significantly.

**Anomaly Detection:** learn the distribution of normal data, then flag new examples that are unlikely under that distribution.

---

## 2.2 — The Story: Detecting Faulty Aircraft Engines

You manufacture aircraft engines. You measure two features at the end of production:

- $x_1$ = Vibration intensity
- $x_2$ = Heat generated

You have 100 engines that all passed quality control (normal engines). A new engine comes off the line — does it look normal or faulty?

```
Heat (x2)
  |
  |        * * *
  |      * * * * *
  |    * * * * * * *   ← Dense cluster of normal engines
  |      * * * * *
  |        * * *
  |
  |                  ? ← New engine — far from the cluster!
  +------------------------------→ Vibration (x1)

This new engine is unusual — anomaly!
```

The question is: how unusual is "unusual enough" to flag as an anomaly?

---

## 2.3 — The Gaussian Distribution: Modeling "Normal"

### What is a Gaussian Distribution?

A Gaussian (Normal) distribution is a bell curve. Most values cluster around the mean. Values far from the mean are increasingly rare.

$$\boxed{p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}$$

Where:
- $\mu$ = mean (center of the bell curve)
- $\sigma^2$ = variance (how wide the bell curve is)
- $p(x)$ = how likely it is to observe value $x$

```
p(x)
  |
  |         █
  |        ███
  |       █████
  |      ███████
  |    ███████████
  |  ███████████████
  +-------------------→ x
         ↑
         μ (mean)

Height at x = how likely that value is
Values far from μ → very low probability → unusual
```

### Key Properties

| Property | Meaning |
|----------|---------|
| $p(\mu)$ = maximum | Mean is the most likely value |
| $p(x)$ decreases as $x$ moves away from $\mu$ | Unusual values are less probable |
| ~68% of data falls within $\mu \pm \sigma$ | One standard deviation |
| ~95% of data falls within $\mu \pm 2\sigma$ | Two standard deviations |
| ~99.7% of data falls within $\mu \pm 3\sigma$ | Three standard deviations |

Anything beyond $3\sigma$ from the mean is very rare — a potential anomaly.

---

## 2.4 — Estimating $\mu$ and $\sigma^2$ From Data

Given $m$ training examples of a single feature $x$:

$$\boxed{\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}}$$

$$\boxed{\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \mu)^2}$$

These are the sample mean and sample variance — you are fitting a Gaussian to your normal data.

### Numerical Example — Vibration Feature

Vibration readings from 10 normal engines:

$$x_1^{(1..10)} = [5.1, 4.8, 5.3, 5.0, 4.9, 5.2, 4.7, 5.1, 4.8, 5.0]$$

$$\mu_1 = \frac{5.1+4.8+5.3+5.0+4.9+5.2+4.7+5.1+4.8+5.0}{10} = \frac{49.9}{10} = 4.99$$

$$\sigma_1^2 = \frac{(5.1-4.99)^2+(4.8-4.99)^2+\cdots+(5.0-4.99)^2}{10}$$

$$= \frac{0.0121+0.0361+0.1+0.0001+0.0081+0.0441+0.0841+0.0121+0.0361+0.0001}{10} = \frac{0.333}{10} = 0.0333$$

The vibration feature follows a Gaussian with $\mu_1 = 4.99$ and $\sigma_1^2 = 0.0333$.

---

## 2.5 — Extending to Multiple Features: The Independence Assumption

Your engines have two features: vibration ($x_1$) and heat ($x_2$).

Andrew Ng makes the same assumption as Naive Bayes — treat features as **independent**. The joint probability is just the product of individual probabilities:

$$\boxed{p(\vec{x}) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2)}$$

For $n$ features, estimate $\mu_j$ and $\sigma_j^2$ separately for each feature, then multiply the probabilities.

---

## 2.6 — Full Numerical Example: Aircraft Engine Anomaly Detection

### The Training Data (10 normal engines, 2 features)

| Engine | Vibration $x_1$ | Heat $x_2$ |
|--------|-----------------|------------|
| E1     | 5.1             | 3.0        |
| E2     | 4.8             | 2.8        |
| E3     | 5.3             | 3.2        |
| E4     | 5.0             | 3.0        |
| E5     | 4.9             | 2.9        |
| E6     | 5.2             | 3.1        |
| E7     | 4.7             | 2.7        |
| E8     | 5.1             | 3.0        |
| E9     | 4.8             | 2.8        |
| E10    | 5.0             | 3.0        |

### Step 1 — Estimate Gaussian Parameters

**For Vibration ($x_1$):**

$$\mu_1 = \frac{5.1+4.8+5.3+5.0+4.9+5.2+4.7+5.1+4.8+5.0}{10} = 4.99$$

$$\sigma_1^2 = 0.033 \quad (\sigma_1 = 0.183)$$

**For Heat ($x_2$):**

$$\mu_2 = \frac{3.0+2.8+3.2+3.0+2.9+3.1+2.7+3.0+2.8+3.0}{10} = 2.95$$

$$\sigma_2^2 = \frac{(3.0-2.95)^2+\cdots+(3.0-2.95)^2}{10} = 0.023 \quad (\sigma_2 = 0.152)$$

### Step 2 — Compute p(x) for New Engines

Normal Gaussian probability formula:

$$p(x_j) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$

**Test Engine A = (5.0, 2.9) — looks normal:**

$$p(x_1=5.0) = \frac{1}{0.456} \times e^{-\frac{0.0001}{0.066}} = 2.193 \times e^{-0.0015} = 2.193 \times 0.9985 = 2.190$$

$$p(x_2=2.9) = \frac{1}{0.380} \times e^{-\frac{0.0025}{0.046}} = 2.630 \times e^{-0.054} = 2.630 \times 0.947 = 2.491$$

$$p(\vec{x}_A) = p(x_1) \times p(x_2) = 2.190 \times 2.491 = \mathbf{5.455}$$

**Test Engine B = (8.0, 5.5) — looks unusual:**

$$p(x_1=8.0) = \frac{1}{0.456} \times e^{-\frac{(8.0-4.99)^2}{0.066}} = 2.193 \times e^{-137.3} \approx \mathbf{0}$$

The vibration of 8.0 is $(8.0-4.99)/0.183 = 16.4$ standard deviations from the mean. Essentially impossible under the normal distribution.

$$p(\vec{x}_B) \approx \mathbf{0}$$

### Step 3 — Apply the Threshold $\epsilon$

Choose a threshold $\epsilon$ (epsilon):

$$\text{If } p(\vec{x}) < \epsilon \rightarrow \text{ANOMALY}$$
$$\text{If } p(\vec{x}) \geq \epsilon \rightarrow \text{NORMAL}$$

With $\epsilon = 0.02$:

| Engine               | $p(\vec{x})$ | Decision   |
|----------------------|--------------|------------|
| Engine A (5.0, 2.9)  | 5.455        | Normal ✓   |
| Engine B (8.0, 5.5)  | ≈ 0          | ANOMALY ✗  |

---

## 2.7 — Choosing the Threshold $\epsilon$

The threshold $\epsilon$ controls the sensitivity of your anomaly detector:

```
Small ε (very sensitive):
→ Flag many examples as anomalies
→ Catch more real anomalies (high recall)
→ But many false alarms (low precision)

Large ε (less sensitive):
→ Only flag very extreme examples
→ Fewer false alarms (high precision)
→ Miss some real anomalies (lower recall)
```

### How to Choose $\epsilon$ in Practice

You need a small labeled validation set — some known anomalies and known normal examples:

```
Labeled validation set:
- 100 normal engines  (y=0)
- 10 known faulty engines (y=1)

For different values of ε:
    Compute F1 score on validation set
    (F1 because anomalies are rare — skewed classes!)

Pick ε that gives best F1 score
```

This is the same Precision-Recall tradeoff from Course 2 Week 3 — anomaly detection is a classic skewed-class problem.

---

## 2.8 — Anomaly Detection vs Supervised Learning

Andrew Ng addresses a natural question:

> *"If I have some labeled anomalies, why not just use Logistic Regression or a supervised classifier instead of this probability approach?"*

The answer comes down to the nature of anomalies:

| Situation | Use Anomaly Detection | Use Supervised Learning |
|-----------|----------------------|------------------------|
| Very few anomalies (20 or fewer) | ✓ | ✗ Not enough data |
| Many types of anomalies | ✓ | ✗ Can't enumerate all types |
| Future anomalies look different | ✓ | ✗ Won't generalize |
| Many normal examples to learn from | ✓ | — |
| Many labeled anomalies (>50) | — | ✓ |
| Future anomalies look similar to past | — | ✓ |

### The Key Insight

**Anomaly Detection** learns what *normal* looks like and flags anything different. It doesn't need to know what anomalies look like — because future anomalies might be entirely new and unexpected.

**Supervised Learning** learns specific patterns of anomalies seen in training. If a new type of anomaly appears that doesn't match the training anomalies, the supervised model misses it.

**Real world scenario: Computer security**

```
Supervised: "I know what hacking looks like — I've seen these attack patterns"
→ New attack type → MISS ✗

Anomaly Detection: "Normal traffic looks like this distribution"
→ Any deviation → flagged → New attack type → CAUGHT ✓
```

---

## 2.9 — Feature Engineering for Anomaly Detection

Andrew Ng emphasizes that choosing the right features is crucial for anomaly detection.

### Making Features Gaussian

Anomaly detection assumes features follow a Gaussian distribution. If your raw feature doesn't look Gaussian — apply a transformation:

```
Original x1 (right-skewed):        After log transform:

Frequency                            Frequency
   |  ██                                |      ██
   |  ████                              |    ██████
   |  ████████                          |  ██████████
   |  ██████████████                    |  ████████████
   +----------------→ x                +----------------→ log(x)
   Skewed — not Gaussian               Bell curve — Gaussian!
```

Common transformations:
- $\log(x)$ or $\log(x+1)$ — for right-skewed features
- $\sqrt{x}$ — for moderately skewed features
- $x^{1/3}$ — for heavily skewed features

### Creating New Features That Reveal Anomalies

Sometimes combining features reveals anomalies that individual features miss.

**Server monitoring example:**
- $x_1$ = CPU load
- $x_2$ = Network traffic

A normally functioning server has both high CPU AND high traffic, or both low. An anomalous server (maybe stuck in an infinite loop) has high CPU but low traffic:

```
Normal server:              Anomalous server:
CPU high → network high     CPU high → network LOW (unusual ratio!)
CPU low  → network low
```

Create a new feature:

$$x_3 = \frac{\text{CPU load}}{\text{Network traffic}}$$

Now the anomaly is obvious — $x_3$ will be unusually high for the faulty server.

This is feature engineering applied to anomaly detection — the same principle from Week 2 of Course 1.

---

# Part 3 — Connecting the Two Algorithms

## How K-Means and Anomaly Detection Complement Each Other

These two algorithms are often used together in real systems:

```
Step 1: K-Means Clustering
    → Group your normal data into clusters
    → Understand the structure of normal behavior
    → "We have 3 types of normal server behavior"

Step 2: Anomaly Detection per Cluster
    → Fit a Gaussian to each cluster separately
    → A new point is anomalous if it doesn't fit ANY cluster well
    → More precise than fitting one global Gaussian
```

This combination is used in network intrusion detection, manufacturing quality control, and fraud detection.

---

# Part 4 — The Full Week 1 Summary

```
UNSUPERVISED LEARNING
"Find structure in data without labels"
            ↓
    ┌────────────────────────────────────┐
    │                                    │
K-MEANS CLUSTERING               ANOMALY DETECTION
"Group similar points              "Flag unusual points"
 together"
    │                                    │
Algorithm:                         Algorithm:
1. Place K centroids randomly      1. Fit Gaussian to each feature
2. Assign each point to            2. Compute p(x) for new point
   nearest centroid                3. If p(x) < ε → ANOMALY
3. Move centroid to mean
   of its cluster
4. Repeat until convergence
    │                                    │
Key Concepts:                       Key Concepts:
- Distortion function J             - Gaussian: p(x;μ,σ²)
- Random initialization problem     - Product rule for multiple features
- Multiple restarts                 - Threshold ε — precision/recall
- Elbow method for K                - vs Supervised Learning
- K-Means++ initialization          - Feature engineering
    │                                    │
Use When:                           Use When:
- Customer segmentation             - Very few known anomalies
- Image compression                 - New anomaly types expected
- Data exploration                  - Fraud, fault, intrusion detection
- Preprocessing for other ML        - Quality control
```

---

## Key Takeaways

### K-Means Clustering

- Assigns $m$ unlabeled points to $K$ clusters by iterating two steps: assign to nearest centroid, recompute centroids as cluster means
- Minimizes distortion $J$ = average squared distance to assigned centroid
- Random initialization can cause bad clustering — run multiple times, keep lowest $J$
- Choose $K$ using the elbow method or based on downstream purpose

### Anomaly Detection

- Models normal data with a Gaussian distribution: estimate $\mu_j$ and $\sigma_j^2$ per feature
- Joint probability: $p(\vec{x}) = \prod_j p(x_j; \mu_j, \sigma_j^2)$
- Flag as anomaly when $p(\vec{x}) < \epsilon$
- Choose $\epsilon$ using F1 score on a small labeled validation set
- Better than supervised learning when anomalies are rare and new types can appear