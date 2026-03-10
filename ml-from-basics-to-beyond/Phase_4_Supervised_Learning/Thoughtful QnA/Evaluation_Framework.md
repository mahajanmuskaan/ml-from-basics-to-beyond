# Evaluation Framework — LR, KNN & SVM Q&A Notes
> How confusion matrices and classification metrics apply uniformly across all three algorithms

---

## The Convergence Point

**Q: LR, KNN, and SVM all work very differently internally — how can the same evaluation framework apply to all three?**

The key insight is: the evaluation framework doesn't care how you made the decision — it only cares about the **final output**: a predicted class label.

No matter how different the internal machinery is, all three algorithms ultimately reduce to:

> *Given a new data point x, output a class label $\hat{y} \in \{0, 1, 2, \ldots\}$*

| Algorithm | Internal Mechanism | Final Output |
|---|---|---|
| Logistic Regression | Computes $P(y=1 \mid x) = 0.73$ → since $0.73 > 0.5$ threshold | $\hat{y} = 1$ |
| KNN | 5 neighbors: 4 are class 1, 1 is class 0 → majority vote | $\hat{y} = 1$ |
| SVM | Computes $w^Tx + b = +2.3$ → positive side of margin | $\hat{y} = 1$ |

All three said $\hat{y} = 1$. The confusion matrix doesn't know or care how they got there. It only sees: *"you predicted 1, the truth was 1 — that's a True Positive."*

---

## The Confusion Matrix

**Q: How is the confusion matrix built, and is the process different for each algorithm?**

The confusion matrix is purely a **post-prediction bookkeeping table**. You run your model on a held-out test set of $n$ samples, collect all $(y_i, \hat{y}_i)$ pairs, and tally them.

For binary classification (Class 0 = Negative, Class 1 = Positive):

```
                    PREDICTED
                  Class 0    Class 1
ACTUAL  Class 0  |  TN    |   FP   |
        Class 1  |  FN    |   TP   |
```

| Cell | Meaning | Status |
|---|---|---|
| **TP** (True Positive) | Model said 1, truth was 1 | ✅ |
| **TN** (True Negative) | Model said 0, truth was 0 | ✅ |
| **FP** (False Positive) | Model said 1, truth was 0 | ❌ Type I Error |
| **FN** (False Negative) | Model said 0, truth was 1 | ❌ Type II Error |

This construction is **identical for LR, KNN, and SVM**. The matrix mechanism doesn't change — only the $\hat{y}$ values feeding into it change based on which model you used.

---

## How Each Algorithm Produces ŷ

**Q: If the confusion matrix is the same, where does the internal difference between the three algorithms actually matter?**

It matters in **how each algorithm produces the final $\hat{y}$** — the decision step:

**Logistic Regression**

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1 \mid x) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

A threshold $\tau$ (default 0.5) converts probability → label. Changing $\tau$ from 0.5 → 0.3 directly changes your TP/FP counts → changes your confusion matrix. This is the basis of the **ROC curve** — you sweep $\tau$ and trace how the confusion matrix evolves.

**KNN**

$$\hat{y} = \text{mode}\{y_i : x_i \in \mathcal{N}_K(x)\}$$

The implicit threshold is $K/2$ — if more than half the neighbors belong to class 1, predict 1. Changing K changes the neighborhood → changes $\hat{y}$ → changes the confusion matrix.

**SVM**

$$\hat{y} = \text{sign}(w^Tx + b)$$

The decision is made at the hyperplane itself (score = 0). There is no natural probability — the threshold is always 0 on the decision score. For ROC curves with SVM, you use the raw score value to sweep thresholds — this is why SVM ROC curves are slightly less interpretable than LR's.

---

## Classification Metrics

**Q: How are standard evaluation metrics computed, and do they differ across the three algorithms?**

Once you have TP, TN, FP, FN — regardless of which model filled them in — all standard metrics are computed **identically**:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(of all predicted positives, how many were correct?)}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{(of all actual positives, how many did we catch?)}$$

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Specificity} = \frac{TN}{TN + FP}$$

These are **model-agnostic formulas**. Whether your $\hat{y}$ came from a sigmoid, a majority vote, or a hyperplane — it doesn't matter at all.

---

## The Deeper Conceptual Point

**Q: Why is it theoretically valid to compare LR, KNN, and SVM on the same confusion matrix — aren't they doing fundamentally different things?**

This is a beautiful abstraction in ML theory. The **evaluation layer** operates on the output space (label space), while the algorithms operate in the **hypothesis space** (how they search for a decision function). These two spaces are **decoupled by design**.

The implicit contract every classifier must fulfill is:

$$f : \mathcal{X} \rightarrow \mathcal{Y} \quad \text{— map from feature space to label space}$$

As long as a model fulfills this contract, any evaluation metric applies uniformly. This is precisely why scikit-learn has a single `classification_report()` that works identically for all three — it only ever sees the $\hat{y}$ array, never the model's internals.

---

**Q: Is there any evaluation scenario where the internal difference between the algorithms does surface?**

Yes — **probabilistic metrics** like log-loss (cross-entropy) and calibration curves require a confidence score, not just a label. Here the internal difference matters:

| Algorithm | Probability Output | Implication |
|---|---|---|
| Logistic Regression | Natively outputs calibrated probabilities | Works directly with log-loss, calibration curves, ROC-AUC |
| KNN | Can use neighbor vote fraction as a proxy probability | Less well-calibrated |
| SVM | Outputs a raw decision score — not a probability | Requires **Platt Scaling** as an additional calibration step to produce reliable probability estimates |

LR has an inherent advantage for any metric that requires confidence scores rather than just class labels.

---

## Summary

**Q: One-line summary of the key insight?**

> The evaluation framework evaluates **what the model decides**, not **how it thinks** — so the confusion matrix and all metrics derived from it are completely model-agnostic.

---