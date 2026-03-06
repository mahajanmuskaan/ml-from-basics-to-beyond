# Day 36 — Evaluation Metrics: Regression & Classification

---

## Part I — Evaluation Metrics for Regression

In regression, the model predicts **continuous numerical values** (house prices, temperature, sales revenue, etc.). To evaluate performance, we measure how far predictions deviate from the true values — these differences are called **errors** or **residuals**.

$$\text{error} = y_i - \hat{y}_i$$

The four most commonly used regression metrics are: **MAE, MSE, RMSE, and R²**.

---

### 1. Mean Absolute Error (MAE)

**Concept:** Average magnitude of errors, treating positive and negative errors equally. Large errors are *not* disproportionately penalized.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Example:**

| Actual | Predicted | Error | \|Error\| |
|--------|-----------|-------|-----------|
| 100    | 110       | −10   | 10        |
| 200    | 190       | +10   | 10        |
| 300    | 310       | −10   | 10        |

$$\text{MAE} = \frac{10 + 10 + 10}{3} = 10$$

The average prediction is off by **10 units**.

**Properties:**

| Property | Meaning |
|---|---|
| Linear penalty | Every error contributes proportionally |
| Robust to outliers | Large errors do not dominate |
| Easy to interpret | Same unit as the target variable |

**When to use:** Demand forecasting, sales prediction, inventory estimation — anywhere outliers should not dominate and interpretability matters.

---

### 2. Mean Squared Error (MSE)

**Concept:** Squares the error before averaging — large errors receive *much* larger penalties.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Example (same data):**

$$\text{MSE} = \frac{100 + 100 + 100}{3} = 100$$

```
Error = 2  →  penalty = 4
Error = 10 →  penalty = 100   ← large mistakes become very costly
```

**Properties:**

| Property | Meaning |
|---|---|
| Quadratic penalty | Large errors strongly penalized |
| Differentiable | Works seamlessly with gradient descent |
| Sensitive to outliers | Outliers dominate the metric |

**Why MSE is the standard training loss:** It is smooth, differentiable, and works naturally with gradient descent — making it the default loss for Linear Regression and many other models.

---

### 3. Root Mean Squared Error (RMSE)

**Concept:** The square root of MSE — restores the metric back to the original unit of the target variable.

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Why take the square root?** MSE has *squared units* (e.g., ₹²), which is uninterpretable. RMSE fixes this — if the target is in ₹, RMSE is also in ₹.

**Example:** If $\text{MSE} = 100$, then $\text{RMSE} = \sqrt{100} = 10$

```
Prediction Error  →  Square  →  Average  →  Square Root  →  RMSE
```

**Properties:**

| Property | Meaning |
|---|---|
| Penalizes large errors | Because errors are squared internally |
| Same unit as target | More interpretable than MSE |
| Sensitive to outliers | Large errors still dominate |

**When to use:** Weather forecasting, finance, energy demand, time series — anywhere large errors are undesirable and units must remain interpretable.

---

### 4. R² (Coefficient of Determination)

**Concept:** Instead of measuring raw error, R² measures **how much of the variance in the data the model explains** — relative to a naive baseline of always predicting the mean.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:

$$SS_{res} = \sum(y_i - \hat{y}_i)^2 \quad \text{(residual sum of squares)}$$

$$SS_{tot} = \sum(y_i - \bar{y})^2 \quad \text{(total sum of squares)}$$

**Interpretation:**

| R² Value | Meaning |
|---|---|
| 1.0 | Perfect model — explains all variance |
| 0.0 | No better than predicting the mean |
| < 0 | Worse than predicting the mean |

**Example:** $R^2 = 0.85$ → the model explains **85% of the variance** in the target.

---

### MAE vs MSE vs RMSE — Comparison

| Metric | Error Treatment | Outlier Sensitivity | Unit |
|---|---|---|---|
| MAE | Absolute | Low | Same as target |
| MSE | Squared | Very High | Squared |
| RMSE | Squared + Root | High | Same as target |

**Visual intuition:**

```
Error = 2     →  MAE = 2,  MSE = 4,    RMSE = 2
Error = 10    →  MAE = 10, MSE = 100,  RMSE = 10
```

MSE exaggerates large errors; MAE does not.

---

### Step-by-Step Algorithm

```
Given: Actual (y), Predicted (ŷ)

Step 1 — error      = y − ŷ
Step 2 — MAE        = mean(|error|)
Step 3 — MSE        = mean(error²)
Step 4 — RMSE       = sqrt(MSE)
Step 5 — SS_res     = Σ(y − ŷ)²
          SS_tot     = Σ(y − mean(y))²
          R²         = 1 − (SS_res / SS_tot)
```

---

### Interview Question: Why RMSE over MAE?

**Short answer:** RMSE penalizes large errors quadratically, making it the better choice when large mistakes are significantly more harmful than small ones.

**Detailed intuition:**

| Error | MAE penalty | RMSE penalty |
|---|---|---|
| 5 | 5 | 25 |
| 10 | 10 | 100 |

**Practical example — Hospital oxygen demand:**

| Prediction | Error | MAE | RMSE penalty |
|---|---|---|---|
| Prediction A: 950 (actual 1000) | 50 | 50 | 2,500 |
| Prediction B: 500 (actual 1000) | 500 | 500 | 250,000 |

RMSE makes the catastrophic prediction *extremely* costly — forcing the model to avoid big mistakes.

---


## Part II — Evaluation Metrics for Classification

In classification, the model predicts **discrete labels** (Spam/Not Spam, Malignant/Benign, Fraud/Legitimate). The foundation of all classification metrics is the **Confusion Matrix**.

---

### 1. Confusion Matrix

A 2×2 table summarizing correct and incorrect predictions for binary classification:

```
                     PREDICTED
                 Positive   Negative
ACTUAL Positive     TP         FN
ACTUAL Negative     FP         TN
```

| Term | Meaning |
|---|---|
| TP (True Positive) | Correctly predicted positive |
| TN (True Negative) | Correctly predicted negative |
| FP (False Positive) | Incorrectly predicted positive (Type I Error) |
| FN (False Negative) | Incorrectly predicted negative (Type II Error) |

**Example — Spam Detection:**

| Email | Actual | Predicted | Result |
|---|---|---|---|
| Email 1 | Spam | Spam | TP |
| Email 2 | Spam | Not Spam | FN |
| Email 3 | Not Spam | Spam | FP |
| Email 4 | Not Spam | Not Spam | TN |

All classification metrics are derived from these four values.

---

### 2. Accuracy

**Concept:** Overall proportion of correct predictions.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Example:** TP=50, TN=40, FP=5, FN=5 → $\text{Accuracy} = \frac{90}{100} = 0.90$

---

### 3. When Accuracy is Misleading — Class Imbalance

Accuracy breaks down when one class dominates the dataset.

**Example — Fraud Detection:**
- 1,000 transactions: 990 legitimate, 10 fraudulent
- A model that predicts *everything as legitimate* achieves **99% accuracy** — yet catches **zero fraud cases**

```
Legitimate ███████████████████████ 99%
Fraud      █ 1%
```

Accuracy rewards the majority class and ignores the minority. This is why we need Precision, Recall, F1, and ROC-AUC.

---

### 4. Precision

**Concept:** Of all predicted positives, how many were actually positive?

> *"When the model says positive, how often is it correct?"*

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Example:** TP=80, FP=20 → $\text{Precision} = \frac{80}{100} = 0.80$

→ 80% of emails flagged as spam were truly spam.

**When precision matters:** When **false positives are costly** — spam filters (don't block legitimate emails), legal classification, search engine results.

---

### 5. Recall (Sensitivity)

**Concept:** Of all actual positives, how many were correctly detected?

> *"How many real positives did the model find?"*

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Example:** TP=90, FN=10 → $\text{Recall} = \frac{90}{100} = 0.90$

→ Model detected 90% of actual disease cases.

**When recall matters:** When **missing positives is dangerous** — cancer detection, fraud detection, disease screening.

---

### Precision vs Recall — Key Difference

| Metric | Focus |
|---|---|
| Precision | Quality of positive predictions |
| Recall | Coverage of actual positives |

**Visual intuition:**
```
Actual Positives = 100
Model predicted 50 positives, 40 were correct.

Precision = 40/50 = 0.80   ← accurate when it says positive
Recall    = 40/100 = 0.40  ← but misses many actual positives
```

**The Precision–Recall Tradeoff:**

```
Lower threshold → detect more positives → ↑ Recall, ↓ Precision
Higher threshold → stricter predictions  → ↑ Precision, ↓ Recall
```

Tuning the threshold shifts the balance depending on what the application demands.

---

### 6. F1-Score

**Concept:** The **harmonic mean** of Precision and Recall — a single balanced metric.

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why harmonic mean?** It penalizes imbalance. If either metric is near zero, F1 collapses:

```
Precision = 1.0, Recall = 0.0  →  F1 = 0   ← both must be strong
```

**Example:** Precision=0.8, Recall=0.6

$$F1 = \frac{2 \times 0.8 \times 0.6}{0.8 + 0.6} = \frac{0.96}{1.4} \approx 0.69$$

**When to use F1:** Imbalanced datasets where both false positives and false negatives matter — fraud detection, medical diagnosis, information retrieval.

---

### 7. ROC Curve

**ROC** (Receiver Operating Characteristic) plots model performance across all classification thresholds:

$$\text{True Positive Rate (TPR)} = \frac{TP}{TP + FN} \quad \leftarrow \text{same as Recall}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

```
TPR
1 |        *
  |      *
  |    *
  |  *
0 |_____________
   0        1
       FPR

Each point = a different classification threshold
```

---

### 8. ROC-AUC

**AUC** (Area Under the ROC Curve) summarizes the ROC curve as a single number, measuring **how well the model separates the two classes**.

**Intuitive meaning:** AUC = the probability that the model ranks a random positive example *higher* than a random negative example.

| AUC | Meaning |
|---|---|
| 1.0 | Perfect classifier |
| 0.9 | Excellent |
| 0.8 | Good |
| 0.7 | Fair |
| 0.5 | Random guessing |

```
Perfect model              Random model
TPR ↑                      TPR ↑
1 | █████                  1 |         /
  | █████                    |       /
  | █████                    |     /
0 |_______                 0 |___/______
  0     1 FPR                0       1 FPR
  AUC = 1.0                  AUC = 0.5
```

---

### Step-by-Step Algorithm

```
Given: TP, TN, FP, FN

Step 1 — Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Step 2 — Precision = TP / (TP + FP)
Step 3 — Recall    = TP / (TP + FN)
Step 4 — F1        = 2 × (Precision × Recall) / (Precision + Recall)
Step 5 — ROC-AUC   = Compute TPR & FPR at each threshold → plot → calculate area
```

---