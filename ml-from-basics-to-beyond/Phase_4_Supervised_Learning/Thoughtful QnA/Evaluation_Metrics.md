# Evaluation Metrics — Complete Guide
### Regression + Classification | Formulae + Real-World Examples

> 🏥 **One dataset, two tasks — used throughout this entire guide:**
> A hospital has data on 1,000 patients.
> - **Regression Task:** Predict a patient's exact **Blood Sugar level** (e.g., 142 mg/dL)
> - **Classification Task:** Predict whether a patient is **Diabetic (1) or Not Diabetic (0)**

---

# SECTION A — Regression Metrics

> **When do you use these?**
> Whenever your model's output is a **continuous number** — house price, temperature, salary, blood sugar, stock price. There is no "correct" or "wrong" class — only how *far off* the prediction was.

---

## A1. Error — The Foundation of Everything

Before any metric, understand the **raw error** for a single prediction:

```
Error  =  Predicted Value  -  Actual Value
        =  y_hat  -  y
```

**Example:**

| Patient | Actual Blood Sugar | Predicted | Error |
|---|---|---|---|
| P1 | 145 mg/dL | 142 mg/dL | −3 |
| P2 | 160 mg/dL | 175 mg/dL | +15 |
| P3 | 120 mg/dL | 118 mg/dL | −2 |
| P4 | 200 mg/dL | 165 mg/dL | −35 |
| P5 | 95 mg/dL | 97 mg/dL | +2 |

Notice: errors can be **positive** (over-predicted) or **negative** (under-predicted). If you simply average the raw errors, the positives and negatives cancel out — giving a misleading score of near zero even when predictions are way off. This is why we need MAE, MSE, and RMSE.

---

## A2. MAE — Mean Absolute Error

### What it is
Take the absolute value of each error (remove the sign), then average them.

```
MAE  =  (1/n) * Sum( |y_hat_i - y_i| )
```

### Calculation from our example

```
|−3| + |+15| + |−2| + |−35| + |+2|
= 3 + 15 + 2 + 35 + 2
= 57

MAE  =  57 / 5  =  11.4 mg/dL
```

### What it means
> "On average, my model's blood sugar prediction is **off by 11.4 mg/dL**."

The unit of MAE is the **same as the target variable** (mg/dL here) — making it directly interpretable by a doctor.

### Real-World Intuition
A delivery app predicts your order will arrive in 30 minutes. It sometimes arrives in 25, sometimes in 40. The MAE of 5 minutes means — on average, the app is off by 5 minutes either way. A doctor reading MAE = 11.4 mg/dL immediately understands the practical magnitude of the error.

### Properties
| Property | Detail |
|---|---|
| Units | Same as target variable — easy to interpret |
| Outlier sensitivity | **Low** — treats a 3 mg/dL error and a 35 mg/dL error proportionally |
| When to use | When all errors are roughly equal importance. When you don't want big errors to dominate. |
| When NOT to use | When a large error (dangerous misprediction) should be penalised much more heavily |

---

## A3. MSE — Mean Squared Error

### What it is
Square each error before averaging. Squaring removes the sign AND magnifies larger errors.

```
MSE  =  (1/n) * Sum( (y_hat_i - y_i)² )
```

### Calculation from our example

```
(−3)² + (15)² + (−2)² + (−35)² + (2)²
= 9 + 225 + 4 + 1225 + 4
= 1467

MSE  =  1467 / 5  =  293.4 mg²/dL²
```

### What it means
> "The average squared error is 293.4."

Notice the unit is now **mg²/dL²** — squared units, which are NOT directly interpretable by a doctor. This is MSE's main weakness.

### Real-World Intuition
Imagine a self-driving car predicting when to brake. A 1-second error might be fine. A 5-second error could be fatal. MSE punishes the 5-second error **25x more** than the 1-second error (25² vs 1²), not just 5x more. This is exactly what you want when large errors are disproportionately dangerous.

### Properties
| Property | Detail |
|---|---|
| Units | Squared units — NOT directly interpretable |
| Outlier sensitivity | **High** — one large error dominates the score heavily |
| Mathematical advantage | Differentiable everywhere — smooth for Gradient Descent |
| When to use | When large errors are much more costly than small ones |
| When NOT to use | When you want an interpretable, human-readable error metric |

---

## A4. RMSE — Root Mean Squared Error

### What it is
Simply take the square root of MSE. This brings the units back to the original scale.

```
RMSE  =  sqrt( MSE )
       =  sqrt( (1/n) * Sum( (y_hat_i - y_i)² ) )
```

### Calculation from our example

```
RMSE  =  sqrt(293.4)  =  17.1 mg/dL
```

### What it means
> "My model's predictions are off by approximately **17.1 mg/dL**, with larger errors penalised more."

### MAE vs RMSE — The Critical Comparison

```
MAE  = 11.4 mg/dL
RMSE = 17.1 mg/dL

RMSE > MAE  →  This always happens when there are large individual errors.
The gap tells you: "There are some really bad predictions hiding in the average."
```

In our data, Patient P4 had an error of −35. MAE treats this as just another error. RMSE punishes it hard (35² = 1225), pulling the score up significantly.

### Real-World Intuition
A weather forecasting model. If it's usually off by 2°C but occasionally off by 15°C (during storms), the MAE might look decent (say 3°C average), but RMSE will be much higher because of those extreme storm-day errors. RMSE essentially says — *"I care a lot about your worst days, not just your average day."*

### Properties
| Property | Detail |
|---|---|
| Units | Same as target — interpretable like MAE |
| Outlier sensitivity | High — large errors are penalised quadratically |
| Best for | When big errors are costly and you want to surface them |
| Relationship to MAE | RMSE >= MAE always. Large gap = outlier errors present. |

---

## A5. R² — R-Squared (Coefficient of Determination)

### What it is
R² answers a different question entirely: **"How much of the variation in the target does my model explain?"**

It compares your model's performance against the dumbest possible baseline — a model that just predicts the **mean** of y for every patient.

```
SS_res  =  Sum( (y_hat_i - y_i)² )       ← your model's total squared error
SS_tot  =  Sum( (y_mean  - y_i)² )       ← dumb baseline's total squared error

R²  =  1  -  (SS_res / SS_tot)
```

### Calculation from our example

```
y_mean = (145 + 160 + 120 + 200 + 95) / 5 = 144

SS_res = 9 + 225 + 4 + 1225 + 4 = 1467      ← your model

SS_tot = (145-144)² + (160-144)² + (120-144)² + (200-144)² + (95-144)²
       = 1 + 256 + 576 + 3136 + 2401
       = 6370                                ← dumb baseline

R²  =  1  -  (1467 / 6370)  =  1  -  0.23  =  0.77
```

### What it means
> "My model explains **77% of the variance** in blood sugar levels. The remaining 23% is unexplained noise."

### Interpreting R² values

| R² value | Interpretation | Real-World Analogy |
|---|---|---|
| R² = 1.0 | Perfect — model explains everything | Weather app predicts exact temperature every day |
| R² = 0.9 | Excellent | Very strong model |
| R² = 0.7 | Good | Useful model with some unexplained variation |
| R² = 0.5 | Moderate | Model is better than guessing, but misses a lot |
| R² = 0.0 | Useless — same as just predicting the mean | Predicting everyone has blood sugar = 144 |
| R² < 0 | Model is WORSE than just predicting the mean | Actively misleading — something is very wrong |

### Real-World Intuition
Imagine trying to predict a student's exam score. If you use ONLY the number of hours they slept the night before, R² might be 0.15 — sleep explains only 15% of score variation. If you also add study hours, practice tests done, and subject difficulty, R² might jump to 0.82. R² tells you **how much of the story your features are telling**.

### Key Warning
> ⚠️ R² can be artificially inflated by adding more features — even useless ones. This is why **Adjusted R²** exists for multiple regression: it penalises for adding features that don't genuinely help.

---

## A6. Side-by-Side Regression Metrics Summary

| Metric | Formula | Unit | Outlier Sensitive? | Interpretable? | Best For |
|---|---|---|---|---|---|
| MAE | `Mean(|y_hat - y|)` | Same as y | No | ✅ Yes | General purpose, robust to outliers |
| MSE | `Mean((y_hat - y)²)` | Squared units | Yes | ❌ No | Math/optimisation, Gradient Descent |
| RMSE | `sqrt(MSE)` | Same as y | Yes | ✅ Yes | When large errors are costly |
| R² | `1 - SS_res/SS_tot` | Unitless (0 to 1) | Moderate | ✅ Yes | Understanding model's explanatory power |

> 💡 **In practice:** Always report both **RMSE** (to surface big errors) and **R²** (to understand how much variance your model captures). MAE is the most business-friendly metric — non-technical stakeholders instantly understand "off by 11 mg/dL."

---
---

# SECTION B — Classification Metrics

> **When do you use these?**
> Whenever your model outputs a **class label** — Diabetic/Not Diabetic, Spam/Not Spam, Fraud/Legitimate, Cat/Dog. The output is discrete, not continuous.

---

## B1. The Confusion Matrix — Everything Flows From Here

### Building it
Run your model on 200 test patients. For each patient, compare the predicted label vs the actual label. Tally into a 2×2 grid:

```
                          PREDICTED
                    Not Diabetic (0)    Diabetic (1)
         ┌──────────────────────────────────────────┐
A    Not │                              │           │
C  Diab. │    TN = 110  ✅              │  FP = 10  │
T  (0)   │                              │    ❌     │
U        ├──────────────────────────────────────────┤
A  Diab. │                              │           │
L  (1)   │    FN = 11                   │  TP = 69  │
         │       ❌                     │    ✅     │
         └──────────────────────────────────────────┘
```

### The 4 cells — memorise these cold

| Cell | Full Name | What happened | Type of Error |
|---|---|---|---|
| **TP = 69** | True Positive | Predicted Diabetic ✅, Actually Diabetic ✅ | Correct |
| **TN = 110** | True Negative | Predicted Not Diabetic ✅, Actually Not Diabetic ✅ | Correct |
| **FP = 10** | False Positive | Predicted Diabetic ❌, Actually NOT Diabetic | **Type I Error** |
| **FN = 11** | False Negative | Predicted Not Diabetic ❌, Actually WAS Diabetic | **Type II Error** |

> 💡 **Memory trick:** The first word (True/False) = was the prediction correct? The second word (Positive/Negative) = what did the model predict?

### The Cost of Each Error

```
FP (False Alarm):   Doctor calls healthy patient diabetic
                    → Patient gets unnecessary tests → Inconvenient, costly, stressful
                    → But patient is FINE

FN (Missed Case):   Doctor misses a real diabetic
                    → Patient goes home untreated
                    → Disease progresses → Could be FATAL
```

**This is why in medical diagnosis, FN is the more dangerous error.** Different domains have different cost asymmetries — this shapes WHICH metric you optimise.

---

## B2. Accuracy

### Formula

```
Accuracy  =  (TP + TN) / (TP + TN + FP + FN)
           =  (69 + 110) / (69 + 110 + 10 + 11)
           =  179 / 200
           =  89.5%
```

### What it means
> "My model correctly classified **89.5% of all patients**."

### Real-World Intuition
A cricket umpire makes 200 decisions in a match (out/not out). If 179 decisions are correct, accuracy = 89.5%. Simple and intuitive.

### The Accuracy Trap — When Accuracy Lies

Imagine a **new dataset**: 950 healthy patients, 50 diabetic patients (heavily imbalanced).

A completely useless model that predicts **everyone is healthy** would score:

```
Accuracy  =  950 / 1000  =  95%
```

95% accuracy — but it MISSED every single diabetic patient. This is the **accuracy paradox**. Accuracy is misleading on imbalanced datasets.

> ⚠️ **Rule:** Only trust Accuracy when your classes are roughly balanced. For imbalanced data, always check Precision, Recall, and F1.

---

## B3. Precision

### Formula

```
Precision  =  TP / (TP + FP)
            =  69 / (69 + 10)
            =  69 / 79
            =  87.3%
```

### What it means
> "Of all the patients my model **flagged as Diabetic**, **87.3%** were actually diabetic."

### Real-World Intuition — The Spam Filter

Your email spam filter flagged 100 emails as spam. Precision = 87% means 87 were actually spam, but 13 were legitimate emails wrongly sent to spam (FP). Precision asks: **"When you cry wolf, how often is it really a wolf?"**

Low Precision = too many false alarms = people lose trust in the system.

### When Precision matters most
When **False Positives are costly:**
- Spam filter: Legitimate emails deleted → user misses important mail
- Drug approval: Approving an ineffective drug → wasted resources, patient harm
- Legal: Convicting an innocent person → Type I Error (grave injustice)

---

## B4. Recall (Sensitivity / True Positive Rate)

### Formula

```
Recall  =  TP / (TP + FN)
         =  69 / (69 + 11)
         =  69 / 80
         =  86.3%
```

### What it means
> "Of all patients who **actually were Diabetic**, my model caught **86.3%** of them."

The remaining 13.7% (FN = 11 patients) were missed — they were diabetic but the model said they weren't.

### Real-World Intuition — Airport Security

Security scanner checks 500 bags for weapons. There are 20 bags with weapons. If the scanner finds 17 of them (misses 3), Recall = 17/20 = 85%. Recall asks: **"Of all the real threats, how many did you actually catch?"**

A scanner with low Recall misses actual threats — unacceptable.

### When Recall matters most
When **False Negatives are costly:**
- Cancer screening: Missed tumour → patient goes untreated → fatal
- Fraud detection: Missed fraudulent transaction → money lost
- Fire alarm: Missed fire → building burns down

> 💡 **The fundamental tradeoff:** You can almost always improve Recall by lowering the prediction threshold — flag MORE patients as diabetic. But this increases FP → Precision drops. You are constantly trading one for the other. This is the **Precision-Recall tradeoff**.

---

## B5. The Precision-Recall Tradeoff

With Logistic Regression, the default threshold is 0.5. But you can move it:

| Threshold | Effect | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| 0.7 (strict) | Predict Diabetic only when very confident | High ↑ | Low ↓ | Few | Many |
| 0.5 (default) | Balanced | 87.3% | 86.3% | 10 | 11 |
| 0.3 (lenient) | Predict Diabetic more aggressively | Low ↓ | High ↑ | Many | Few |

```
Strict threshold (0.7):   Miss fewer healthy people (low FP)
                          But miss more real diabetics (high FN)

Lenient threshold (0.3):  Catch almost every diabetic (low FN)
                          But flag many healthy people too (high FP)
```

> There is no universally "right" threshold — it depends on the **cost of each error type** in your specific domain. A doctor setting a diabetes screening threshold should weight FN much more heavily than FP.

---

## B6. F1 Score

### The Problem it Solves
You need ONE number to compare two models, but Precision and Recall point in different directions. F1 is the **harmonic mean** of both — it punishes extreme imbalances.

### Formula

```
F1  =  2 * (Precision * Recall) / (Precision + Recall)
     =  2 * (0.873 * 0.863) / (0.873 + 0.863)
     =  2 * 0.753 / 1.736
     =  86.8%
```

### Why Harmonic Mean, Not Simple Average?

Arithmetic mean of 90% Precision and 10% Recall = **50%** — misleadingly decent-looking.
Harmonic mean (F1) = **18%** — correctly reflects that this model is terrible.

The harmonic mean is always closer to the **smaller** of the two values. A model cannot "hide" a terrible Recall behind a great Precision.

### Real-World Intuition — Comparing Two Models

| Model | Precision | Recall | Arithmetic Mean | F1 Score |
|---|---|---|---|---|
| Model A | 90% | 30% | 60% | **44.4%** — Recall is a problem |
| Model B | 75% | 75% | 75% | **75%** — Balanced ✅ |

F1 correctly identifies Model B as superior despite Model A having higher precision. The arithmetic mean was fooled.

### When to use F1
- Imbalanced datasets where Accuracy is misleading
- When you want a single metric balancing Precision and Recall
- NLP tasks (information retrieval, named entity recognition)

---

## B7. Specificity (True Negative Rate)

### Formula

```
Specificity  =  TN / (TN + FP)
              =  110 / (110 + 10)
              =  110 / 120
              =  91.7%
```

### What it means
> "Of all patients who **were NOT diabetic**, my model correctly identified **91.7%** of them as healthy."

### Recall vs Specificity — The Complete Picture

```
Recall (Sensitivity)  =  How well the model catches POSITIVE cases (diabetics)
Specificity           =  How well the model catches NEGATIVE cases (healthy patients)

Together they give the complete performance picture across BOTH classes.
```

### Real-World Intuition — HIV Testing

- **High Recall (Sensitivity):** Ensure almost every HIV-positive person is detected → no missed cases
- **High Specificity:** Ensure almost every HIV-negative person is correctly cleared → no false positives causing panic

A good medical diagnostic test needs BOTH to be high. In practice, increasing sensitivity often reduces specificity — hence the need for confirmatory tests.

---

## B8. ROC Curve and AUC

### What is the ROC Curve?
ROC stands for **Receiver Operating Characteristic**. It plots:
- **Y-axis:** Recall (True Positive Rate) — how many real diabetics are caught
- **X-axis:** False Positive Rate (1 − Specificity) — how many healthy patients are wrongly flagged

As you **sweep the threshold from 1.0 down to 0.0**, you trace a curve:

```
Recall (TPR)
1.0 │              ╭─────────────── Perfect model
    │          ╭───╯
    │       ╭──╯               ← Your model's curve (AUC = 0.92)
    │    ╭──╯
    │ ╭──╯
    │╱                         ← Random guessing (AUC = 0.5, diagonal line)
    └─────────────────────────
    0.0                    1.0
              FPR (1 - Specificity)
```

### AUC — Area Under the ROC Curve

```
AUC = 0.5    →  Random guessing (no better than a coin flip)
AUC = 0.7    →  Fair model
AUC = 0.8    →  Good model
AUC = 0.9+   →  Excellent model
AUC = 1.0    →  Perfect model
```

### What it means
> "If I pick one random diabetic patient and one random healthy patient, my model assigns a higher probability of diabetes to the diabetic patient **92% of the time.**"

### Why AUC is Powerful for Comparison

```
Model A (SVM):            AUC = 0.94  ✅  Best overall discriminator
Model B (Logistic Reg):   AUC = 0.91
Model C (KNN, K=7):       AUC = 0.87
```

AUC is **threshold-independent** — it evaluates the model's inherent discriminating power across ALL possible thresholds at once. This makes it the gold standard for comparing classifiers.

> ⚠️ **AUC limitation:** On highly imbalanced datasets (1% diabetic, 99% healthy), AUC can still look impressive even for a poor model. Use **Precision-Recall AUC** instead for severely imbalanced data.

---

## B9. Log Loss (Cross-Entropy Loss)

### What it is
Unlike all other classification metrics that only look at the final label, Log Loss evaluates the **confidence** of predictions — it penalises models that are wrong AND confident.

```
Log Loss  =  -(1/n) * Sum[ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
```

### Real-World Intuition

| Actual | Model's Predicted Probability | Log Loss for this patient |
|---|---|---|
| Diabetic (1) | p = 0.95 | Very low — model was right AND confident ✅ |
| Diabetic (1) | p = 0.55 | Medium — model was right but barely confident |
| Diabetic (1) | p = 0.05 | Very HIGH — model was wrong AND very confident ❌ |

A model that says "I'm 95% sure this patient is healthy" and is wrong gets **catastrophically penalised**. Log Loss rewards models that are calibrated — confident when right, uncertain when unsure.

```
Perfect model:  Log Loss → 0
Random model:   Log Loss ≈ 0.69
Terrible model: Log Loss → very large
```

### When to use
- When **probability calibration** matters (e.g., risk scoring in medicine or finance)
- Comparing probabilistic models
- Training Logistic Regression (this IS the cost function being minimised)

---

## B10. Complete Classification Metrics Summary

| Metric | Formula | What it asks | When to prioritise |
|---|---|---|---|
| **Accuracy** | `(TP+TN) / Total` | Overall, how often is the model correct? | Balanced classes only |
| **Precision** | `TP / (TP+FP)` | When you predict Positive, how often are you right? | FP is costly (spam, false convictions) |
| **Recall** | `TP / (TP+FN)` | Of all real Positives, how many did you catch? | FN is costly (disease, fraud, fire) |
| **F1 Score** | `2*P*R / (P+R)` | Balanced view of Precision and Recall | Imbalanced data, single-metric comparison |
| **Specificity** | `TN / (TN+FP)` | Of all real Negatives, how many were correctly identified? | Paired with Recall for full picture |
| **AUC-ROC** | Area under ROC | Overall discriminating power across all thresholds | Comparing models; threshold-independent |
| **Log Loss** | `-Mean[y*log(p)]` | How confident and correct are your probability outputs? | When calibrated probabilities matter |

---

## B11. Choosing the Right Metric — Decision Guide

```
Is your dataset balanced?
│
├── YES → Accuracy is a reasonable starting metric
│         Add Precision/Recall for deeper understanding
│
└── NO (imbalanced) → NEVER use Accuracy alone
                      Use F1, AUC-ROC, or Precision-Recall curve

What is more dangerous — FP or FN?
│
├── FN is more dangerous (disease, fraud, fire, security)
│   → Optimise RECALL
│   → Lower prediction threshold
│
├── FP is more dangerous (spam filter, legal, drug approval)
│   → Optimise PRECISION
│   → Raise prediction threshold
│
└── Both matter equally
    → Optimise F1 Score

Do you need to compare models holistically?
└── Use AUC-ROC (threshold-independent, single clean comparison)

Do you care about probability outputs (not just labels)?
└── Use Log Loss
```

---

# SECTION C — The Full Picture: Regression vs. Classification

| | Regression | Classification |
|---|---|---|
| **Output type** | Continuous number (142 mg/dL) | Discrete label (Diabetic / Not) |
| **Primary metric** | RMSE or MAE | Confusion Matrix → Precision/Recall/F1 |
| **"How wrong" question** | By how many units was I off? | How many did I misclassify and which type? |
| **Error asymmetry** | Usually symmetric (off by 5 above = off by 5 below) | Asymmetric — FP and FN have different costs |
| **Baseline comparison** | R² vs. mean-predictor baseline | Accuracy vs. majority-class baseline |
| **Confidence metrics** | N/A | Log Loss, AUC-ROC |

---

# SECTION D — Quick Revision: All Formulae in One Place

```
── REGRESSION ──────────────────────────────────────────────────

MAE    =  (1/n) * Sum( |y_hat - y| )
MSE    =  (1/n) * Sum( (y_hat - y)² )
RMSE   =  sqrt( MSE )
R²     =  1  -  (SS_res / SS_tot)
       =  1  -  Sum(y_hat-y)² / Sum(y_mean-y)²

── CLASSIFICATION ───────────────────────────────────────────────

Accuracy    =  (TP + TN) / (TP + TN + FP + FN)
Precision   =  TP / (TP + FP)
Recall      =  TP / (TP + FN)
Specificity =  TN / (TN + FP)
F1 Score    =  2 * (Precision * Recall) / (Precision + Recall)
FPR         =  FP / (FP + TN)          ← X-axis of ROC curve
Log Loss    =  -(1/n) * Sum[ y*log(p) + (1-y)*log(1-p) ]
```

---

# SECTION E — Real-World Domain Cheat Sheet

| Domain | Task | Most Important Metric | Reason |
|---|---|---|---|
| Cancer/Disease Detection | Classification | **Recall** | Missing a real case (FN) is fatal |
| Spam Filter | Classification | **Precision** | Deleting real emails (FP) destroys trust |
| Fraud Detection | Classification | **Recall + F1** | Missing fraud (FN) means financial loss |
| House Price Prediction | Regression | **RMSE + R²** | Large errors (outliers) matter a lot |
| Delivery Time Prediction | Regression | **MAE** | All errors roughly equal importance |
| Credit Risk Scoring | Classification | **AUC + Log Loss** | Need calibrated probabilities for risk bands |
| Weather Forecasting | Regression | **RMSE** | Extreme weather errors are costly |
| Resume Screening | Classification | **Precision** | False positives (unqualified candidates) waste recruiter time |
| Fire/Flood Alarm | Classification | **Recall** | Missing a real event (FN) is catastrophic |
| Student Score Prediction | Regression | **MAE + R²** | Interpretable, understand explanatory power |

---

> **The one-line philosophy behind all evaluation metrics:**
> *"A metric is just a formalised answer to the question — in your specific domain, what does it mean to be wrong, and how much does each type of wrong cost you?"*