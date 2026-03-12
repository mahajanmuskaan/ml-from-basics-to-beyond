# SVM — Complete Mathematical Walkthrough
> One dataset, solved end to end: Linear (Primal) → Dual → Soft Margin → Polynomial Kernel → RBF Kernel

---

## Table of Contents
1. [The Dataset](#the-dataset)
2. [Stage 1 — Hard Margin Linear SVM (Primal)](#stage-1--hard-margin-linear-svm-primal)
3. [Stage 2 — Dual Formulation: The Bridge to Kernels](#stage-2--dual-formulation-the-bridge-to-kernels)
4. [Stage 3 — Soft Margin](#stage-3--soft-margin)
5. [Stage 4 — Polynomial Kernel](#stage-4--polynomial-kernel)
6. [Stage 5 — RBF Kernel](#stage-5--rbf-kernel)
7. [Complete Summary](#complete-summary)
8. [The Mathematical Flow](#the-mathematical-flow)

---

## The Dataset

4 patients, 2 features — Sugar ($x_1$) and Inflammation ($x_2$):

| Patient | $x_1$ (Sugar) | $x_2$ (Inflammation) | Label $y$ |
|---|---|---|---|
| P1 | 1 | 2 | −1 (Healthy) |
| P2 | 2 | 1 | −1 (Healthy) |
| P3 | 4 | 5 | +1 (Diabetic) |
| P4 | 5 | 4 | +1 (Diabetic) |

**New patient to classify:** $P_{\text{new}} = (3, 4)$

```
x2
|
5|         × P3
4|            × P4      ← Diabetic group
3|
2| ○ P1
1|    ○ P2              ← Healthy group
+------------------→ x1
  1  2  3  4  5
```

---

## Stage 1 — Hard Margin Linear SVM (Primal)

### 1.1 — The Optimization Problem

Margin width $= \frac{2}{||\vec{w}||}$. Maximizing this = minimizing $||\vec{w}||$ = minimizing $\frac{1}{2}||\vec{w}||^2$.

**Primal Problem:**

$$\min_{\vec{w},b} \ \frac{1}{2}||\vec{w}||^2 = \min_{\vec{w},b} \ \frac{1}{2}(w_1^2 + w_2^2)$$

**Subject to — every point correctly classified outside the margin:**

$$y^{(i)}(\vec{w}\cdot\vec{x}^{(i)} + b) \geq 1 \quad \forall i$$

Writing out each constraint explicitly:

$$\text{P1: } (-1)(w_1 \cdot 1 + w_2 \cdot 2 + b) \geq 1 \quad \Rightarrow \quad w_1 + 2w_2 + b \leq -1$$
$$\text{P2: } (-1)(w_1 \cdot 2 + w_2 \cdot 1 + b) \geq 1 \quad \Rightarrow \quad 2w_1 + w_2 + b \leq -1$$
$$\text{P3: } (+1)(w_1 \cdot 4 + w_2 \cdot 5 + b) \geq 1 \quad \Rightarrow \quad 4w_1 + 5w_2 + b \geq 1$$
$$\text{P4: } (+1)(w_1 \cdot 5 + w_2 \cdot 4 + b) \geq 1 \quad \Rightarrow \quad 5w_1 + 4w_2 + b \geq 1$$

---

### 1.2 — Finding $\vec{w}^*$ and $b^*$

**Key Insight:** Support vectors satisfy the constraint with **equality**:

$$y^{(i)}(\vec{w}\cdot\vec{x}^{(i)} + b) = 1 \quad \text{(exactly, not just } \geq\text{)}$$

**Symmetry Argument — Why $w_1 = w_2$:**

- P1 and P2: $x_1 + x_2 = 3$ — both healthy points lie on the same diagonal
- P3 and P4: $x_1 + x_2 = 9$ — both diabetic points lie on the same diagonal

The two groups are symmetric about the line $x_1 = x_2$, so $w_1 = w_2 = a$, giving $\vec{w} = [a, a]$.

**Setting up the system — using equality at support vectors:**

From P1 $(x_1=1, x_2=2, y=-1)$:

$$(-1)(a \cdot 1 + a \cdot 2 + b) = 1 \quad \Rightarrow \quad 3a + b = -1 \quad \cdots (i)$$

From P3 $(x_1=4, x_2=5, y=+1)$:

$$(+1)(a \cdot 4 + a \cdot 5 + b) = 1 \quad \Rightarrow \quad 9a + b = 1 \quad \cdots (ii)$$

**Solving:**

Subtract $(i)$ from $(ii)$: $6a = 2 \Rightarrow \boxed{a = \frac{1}{3}}$

Substitute into $(i)$: $1 + b = -1 \Rightarrow \boxed{b = -2}$

$$\boxed{\vec{w}^* = \left[\frac{1}{3},\ \frac{1}{3}\right], \quad b^* = -2}$$

**Decision boundary:** $\frac{1}{3}x_1 + \frac{1}{3}x_2 - 2 = 0 \quad \Rightarrow \quad x_1 + x_2 = 6$

---

### 1.3 — Verify Constraints and Identify Support Vectors

| Patient | Calculation | Value | Status |
|---|---|---|---|
| P1 $(1,2,y=-1)$ | $(-1)(\frac{1}{3}+\frac{2}{3}-2) = (-1)(-1)$ | $= 1$ | ✓ **Support Vector** |
| P2 $(2,1,y=-1)$ | $(-1)(\frac{2}{3}+\frac{1}{3}-2) = (-1)(-1)$ | $= 1$ | ✓ **Support Vector** |
| P3 $(4,5,y=+1)$ | $(+1)(\frac{4}{3}+\frac{5}{3}-2) = 3-2$ | $= 1$ | ✓ **Support Vector** |
| P4 $(5,4,y=+1)$ | $(+1)(\frac{5}{3}+\frac{4}{3}-2) = 3-2$ | $= 1$ | ✓ **Support Vector** |

All 4 points are support vectors — the data is perfectly symmetric, so all 4 sit exactly on the margin edges.

---

### 1.4 — Compute Margin Width

$$||\vec{w}^*|| = \sqrt{\left(\frac{1}{3}\right)^2 + \left(\frac{1}{3}\right)^2} = \sqrt{\frac{2}{9}} = \frac{\sqrt{2}}{3}$$

$$\boxed{\text{Margin Width} = \frac{2}{||\vec{w}^*||} = \frac{2}{\frac{\sqrt{2}}{3}} = \frac{6}{\sqrt{2}} = 3\sqrt{2} \approx 4.243}$$

---

### 1.5 — The Three Boundaries

| Boundary | Equation | Line |
|---|---|---|
| Healthy margin | $\vec{w}\cdot\vec{x}+b = -1$ | $x_1+x_2 = 3$ |
| **Decision boundary** | $\vec{w}\cdot\vec{x}+b = 0$ | $x_1+x_2 = 6$ |
| Diabetic margin | $\vec{w}\cdot\vec{x}+b = +1$ | $x_1+x_2 = 9$ |

```
x2
|
9|. . . . . . . .← Diabetic margin (x1+x2=9)
7|         × P3(4,5)
6|             × P4(5,4)
4|= = = = = = =  ← DECISION BOUNDARY (x1+x2=6)
2|○ P1(1,2)
1|   ○ P2(2,1)
 |. . . . . . . .← Healthy margin (x1+x2=3)
 +-------------------→ x1

←——————3√2 ≈ 4.24——————→
           MARGIN
```

---

### 1.6 — Predict New Patient $P_{\text{new}} = (3, 4)$

$$f(3,4) = \frac{1}{3}(3) + \frac{1}{3}(4) - 2 = 1 + \frac{4}{3} - 2 = \frac{7}{3} - 2 = +\frac{1}{3} > 0$$

$$\boxed{\text{Prediction: Diabetic (+1)}} \quad \checkmark$$

Check: $x_1 + x_2 = 7 > 6$ → above the decision boundary → Diabetic ✓

---

## Stage 2 — Dual Formulation: The Bridge to Kernels

### 2.1 — Why We Need the Dual

The primal works fine for linear SVM. But to use kernels, we need the decision function expressed as **dot products** $\vec{x}^{(i)}\cdot\vec{x}^{(j)}$ between training points. The dual formulation achieves exactly this.

---

### 2.2 — The Lagrangian

Introduce a **Lagrange multiplier** $\alpha_i \geq 0$ for each constraint:

$$\mathcal{L}(\vec{w}, b, \vec{\alpha}) = \frac{1}{2}||\vec{w}||^2 - \sum_{i=1}^{m}\alpha_i\left[y^{(i)}(\vec{w}\cdot\vec{x}^{(i)}+b) - 1\right]$$

**KKT Conditions — Setting Gradients to Zero:**

With respect to $\vec{w}$:
$$\frac{\partial \mathcal{L}}{\partial \vec{w}} = 0 \quad \Rightarrow \quad \boxed{\vec{w} = \sum_{i=1}^{m}\alpha_i y^{(i)}\vec{x}^{(i)}}$$

This is critical: **$\vec{w}$ is a weighted sum of training points**, weighted by $\alpha_i y^{(i)}$.

With respect to $b$:
$$\frac{\partial \mathcal{L}}{\partial b} = 0 \quad \Rightarrow \quad \sum_{i=1}^{m}\alpha_i y^{(i)} = 0$$

**KKT Complementary Slackness:**
$$\alpha_i\left[y^{(i)}(\vec{w}\cdot\vec{x}^{(i)}+b) - 1\right] = 0 \quad \forall i$$

This means either:
- $\alpha_i = 0$ → point is **not** a support vector
- $y^{(i)}(\vec{w}\cdot\vec{x}^{(i)}+b) = 1$ → point **is** a support vector (sits on margin edge)

---

### 2.3 — The Dual Problem

Substituting $\vec{w} = \sum_i \alpha_i y^{(i)}\vec{x}^{(i)}$ back into the Lagrangian gives:

$$\boxed{\max_{\vec{\alpha}} \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_j y^{(i)}y^{(j)}(\vec{x}^{(i)}\cdot\vec{x}^{(j)})}$$

$$\text{subject to: } \alpha_i \geq 0, \quad \sum_{i=1}^{m}\alpha_i y^{(i)} = 0$$

The dot products $\vec{x}^{(i)}\cdot\vec{x}^{(j)}$ are exactly where **the kernel trick will enter later**.

---

### 2.4 — Solving for $\alpha_i$ Values

By symmetry: $\alpha_1 = \alpha_2 = \alpha_3 = \alpha_4 = \alpha$

**Verify constraint:** $\alpha(-1) + \alpha(-1) + \alpha(+1) + \alpha(+1) = 0$ ✓

**Compute all dot products $\vec{x}^{(i)}\cdot\vec{x}^{(j)}$:**

| Pair | Calculation | Value |
|---|---|---|
| $P1 \cdot P1$ | $1(1)+2(2)$ | $5$ |
| $P1 \cdot P2$ | $1(2)+2(1)$ | $4$ |
| $P1 \cdot P3$ | $1(4)+2(5)$ | $14$ |
| $P1 \cdot P4$ | $1(5)+2(4)$ | $13$ |
| $P2 \cdot P2$ | $2(2)+1(1)$ | $5$ |
| $P2 \cdot P3$ | $2(4)+1(5)$ | $13$ |
| $P2 \cdot P4$ | $2(5)+1(4)$ | $14$ |
| $P3 \cdot P3$ | $4(4)+5(5)$ | $41$ |
| $P3 \cdot P4$ | $4(5)+5(4)$ | $40$ |
| $P4 \cdot P4$ | $5(5)+4(4)$ | $41$ |

**$y^{(i)}y^{(j)}(\vec{x}^{(i)}\cdot\vec{x}^{(j)})$ Matrix:**

| | P1 ($y=-1$) | P2 ($y=-1$) | P3 ($y=+1$) | P4 ($y=+1$) |
|---|---|---|---|---|
| **P1** | $+5$ | $+4$ | $-14$ | $-13$ |
| **P2** | $+4$ | $+5$ | $-13$ | $-14$ |
| **P3** | $-14$ | $-13$ | $+41$ | $+40$ |
| **P4** | $-13$ | $-14$ | $+40$ | $+41$ |

**Sum of all entries:**

$$(-18) + (-18) + (54) + (54) = 72$$

**Dual objective with $\alpha_1=\alpha_2=\alpha_3=\alpha_4=\alpha$:**

$$W(\alpha) = 4\alpha - \frac{1}{2}(72)\alpha^2 = 4\alpha - 36\alpha^2$$

**Maximize — take derivative and set to zero:**

$$\frac{dW}{d\alpha} = 4 - 72\alpha = 0 \quad \Rightarrow \quad \boxed{\alpha = \frac{1}{18}}$$

**Verify by recovering $\vec{w}^*$:**

$$\vec{w}^* = \frac{1}{18}\left[(-1)(1,2) + (-1)(2,1) + (+1)(4,5) + (+1)(5,4)\right] = \frac{1}{18}[6, 6] = \left[\frac{1}{3}, \frac{1}{3}\right] \checkmark$$

---

### 2.5 — Decision Function Through the Dual

$$f(\vec{x}) = \sum_{i=1}^{m}\alpha_i y^{(i)}(\vec{x}^{(i)}\cdot\vec{x}) + b$$

**Predict $P_{\text{new}} = (3, 4)$:**

| Training Point | Dot Product with $(3,4)$ |
|---|---|
| P1: $(1,2)$ | $1(3)+2(4) = 11$ |
| P2: $(2,1)$ | $2(3)+1(4) = 10$ |
| P3: $(4,5)$ | $4(3)+5(4) = 32$ |
| P4: $(5,4)$ | $5(3)+4(4) = 31$ |

$$f(3,4) = \frac{1}{18}\left[(-1)(11) + (-1)(10) + (+1)(32) + (+1)(31)\right] + (-2)$$
$$= \frac{1}{18}(42) - 2 = \frac{7}{3} - 2 = +\frac{1}{3} > 0$$

$$\boxed{\text{Prediction: Diabetic (+1)}} \checkmark$$

Same result as the primal. The dual expresses the **identical decision function through dot products** — exactly what we need to plug in kernel functions.

---

## Stage 3 — Soft Margin

### 3.1 — Adding an Outlier

Add a mislabeled patient: **P5 = (3, 2), $y = +1$ (Diabetic)**

```
x2
|
5|         × P3
4|            × P4
3|
2| ○ P1    ★ P5 ← Labeled Diabetic, but lands in healthy zone!
1|    ○ P2
+------------------→ x1
  1  2  3  4  5

★ P5: x1+x2=5 < 6 → wrong side of the decision boundary
```

---

### 3.2 — Computing the Slack Variable $\xi_5$

Check P5 against the original boundary:

$$f(P5) = \frac{1}{3}(3) + \frac{1}{3}(2) - 2 = 1 + \frac{2}{3} - 2 = -\frac{1}{3}$$

$$y^{(5)} \cdot f(P5) = (+1)\left(-\frac{1}{3}\right) = -\frac{1}{3} < 1 \quad \text{→ margin violated}$$

$$\xi_5 = \max\left(0,\ 1 - y^{(5)}(\vec{w}\cdot\vec{x}^{(5)}+b)\right) = \max\left(0,\ 1 + \frac{1}{3}\right) = \frac{4}{3}$$

Since $\xi_5 = \frac{4}{3} > 1$ → P5 is **misclassified**.

| $\xi$ range | Meaning | P5 Status |
|---|---|---|
| $\xi = 0$ | Safely outside margin | — |
| $0 < \xi < 1$ | Inside margin, correct side | — |
| $\xi = 1$ | On the decision boundary | — |
| $\xi > 1$ | **Misclassified** | ← P5 here ($\xi = 4/3$) |

---

### 3.3 — The Soft Margin Objective with $C$

$$\min_{\vec{w},b,\xi} \ \frac{1}{2}||\vec{w}||^2 + C\sum_{i=1}^{m}\xi_i = \frac{1}{2}||\vec{w}||^2 + C \cdot \frac{4}{3}$$

**The two terms fight each other:**

| $C$ | Penalty for $\xi_5$ | Effect | Risk |
|---|---|---|---|
| Large ($C=100$) | $100 \times \frac{4}{3} = 133.3$ | Boundary shifts to classify P5 correctly, margin narrows | Overfitting |
| Small ($C=0.01$) | $0.01 \times \frac{4}{3} = 0.013$ | Boundary unchanged, P5 remains misclassified | Underfitting |

```
Large C: boundary shifts to accommodate P5
  Healthy margin: x1+x2 ≈ 2
  Decision boundary: x1+x2 ≈ 5
  → Narrower margin

Small C: boundary stays put, P5 misclassified
  Healthy margin: x1+x2 = 3
  Decision boundary: x1+x2 = 6
  → Wide margin maintained
```

---

## Stage 4 — Polynomial Kernel

### 4.1 — The Formula

$$K(\vec{x}^{(i)},\ \vec{x}^{(j)}) = (\vec{x}^{(i)}\cdot\vec{x}^{(j)} + 1)^2 \quad (d=2,\ c=1)$$

In the dual problem, **replace every dot product** $\vec{x}^{(i)}\cdot\vec{x}^{(j)}$ with $K(\vec{x}^{(i)}, \vec{x}^{(j)})$.

---

### 4.2 — Compute the Full Kernel Matrix

| Pair | Dot Product | $+1$ | $K = (\cdot)^2$ |
|---|---|---|---|
| $K(P1,P1)$ | $5$ | $6$ | $36$ |
| $K(P1,P2)$ | $4$ | $5$ | $25$ |
| $K(P1,P3)$ | $14$ | $15$ | $225$ |
| $K(P1,P4)$ | $13$ | $14$ | $196$ |
| $K(P2,P2)$ | $5$ | $6$ | $36$ |
| $K(P2,P3)$ | $13$ | $14$ | $196$ |
| $K(P2,P4)$ | $14$ | $15$ | $225$ |
| $K(P3,P3)$ | $41$ | $42$ | $1764$ |
| $K(P3,P4)$ | $40$ | $41$ | $1681$ |
| $K(P4,P4)$ | $41$ | $42$ | $1764$ |

---

### 4.3 — $y^{(i)}y^{(j)}K_{ij}$ Matrix

| | P1 ($y=-1$) | P2 ($y=-1$) | P3 ($y=+1$) | P4 ($y=+1$) |
|---|---|---|---|---|
| **P1** | $+36$ | $+25$ | $-225$ | $-196$ |
| **P2** | $+25$ | $+36$ | $-196$ | $-225$ |
| **P3** | $-225$ | $-196$ | $+1764$ | $+1681$ |
| **P4** | $-196$ | $-225$ | $+1681$ | $+1764$ |

**Sum of all entries:** $(-360) + (-360) + (3024) + (3024) = 5328$

---

### 4.4 — Solve Dual for $\alpha$

$$W(\alpha) = 4\alpha - \frac{5328}{2}\alpha^2 = 4\alpha - 2664\alpha^2$$

$$\frac{dW}{d\alpha} = 4 - 5328\alpha = 0 \quad \Rightarrow \quad \boxed{\alpha = \frac{1}{1332} \approx 0.000751}$$

---

### 4.5 — Compute $b_{\text{poly}}$ Using Support Vector P1

$$\sum_j \alpha_j y^{(j)} K(\vec{x}^{(j)}, P1) = \frac{1}{1332}[(-1)(36)+(-1)(25)+(+1)(225)+(+1)(196)] = \frac{360}{1332} = 0.270$$

From $(-1)(0.270 + b) = 1$: $\quad \boxed{b_{\text{poly}} = -1.270}$

**Verify using P3:** $\frac{1}{1332}[-225-196+1764+1681] = \frac{3024}{1332} = 2.270$

From $(+1)(2.270 + b) = 1$: $b = -1.270$ ✓

---

### 4.6 — Predict $P_{\text{new}} = (3, 4)$ with Polynomial Kernel

| Training Point | Dot Product with $(3,4)$ | $+1$ | $K = (\cdot)^2$ |
|---|---|---|---|
| P1: $(1,2)$ | $11$ | $12$ | $144$ |
| P2: $(2,1)$ | $10$ | $11$ | $121$ |
| P3: $(4,5)$ | $32$ | $33$ | $1089$ |
| P4: $(5,4)$ | $31$ | $32$ | $1024$ |

$$f(3,4) = \frac{1}{1332}\left[(-1)(144)+(-1)(121)+(+1)(1089)+(+1)(1024)\right] + (-1.270)$$
$$= \frac{1848}{1332} - 1.270 = 1.387 - 1.270 = \mathbf{+0.117} > 0$$

$$\boxed{\text{Polynomial Kernel Prediction: Diabetic (+1)}} \checkmark$$

---

## Stage 5 — RBF Kernel

### 5.1 — The Formula

$$K(\vec{x}^{(i)},\ \vec{x}^{(j)}) = \exp\left(-\gamma||\vec{x}^{(i)} - \vec{x}^{(j)}||^2\right), \quad \gamma = 0.5$$

Output always between 0 and 1 — a **similarity score**:
- Points **close** → $K$ near 1
- Points **far** → $K$ near 0

---

### 5.2 — Compute All Squared Distances

| Pair | Calculation | $d^2$ |
|---|---|---|
| $P1-P1$ | $0$ | $0$ |
| $P1-P2$ | $(1-2)^2+(2-1)^2$ | $2$ |
| $P1-P3$ | $(1-4)^2+(2-5)^2$ | $18$ |
| $P1-P4$ | $(1-5)^2+(2-4)^2$ | $20$ |
| $P2-P2$ | $0$ | $0$ |
| $P2-P3$ | $(2-4)^2+(1-5)^2$ | $20$ |
| $P2-P4$ | $(2-5)^2+(1-4)^2$ | $18$ |
| $P3-P3$ | $0$ | $0$ |
| $P3-P4$ | $(4-5)^2+(5-4)^2$ | $2$ |
| $P4-P4$ | $0$ | $0$ |

---

### 5.3 — Full RBF Kernel Matrix $K_{ij} = e^{-0.5 \times d^2}$

| Pair | $d^2$ | $K = e^{-0.5 d^2}$ |
|---|---|---|
| $K(P1,P1)$ | $0$ | $1.000$ |
| $K(P1,P2)$ | $2$ | $e^{-1} = 0.368$ |
| $K(P1,P3)$ | $18$ | $e^{-9} \approx 0.000123$ |
| $K(P1,P4)$ | $20$ | $e^{-10} \approx 0.0000454$ |
| $K(P2,P2)$ | $0$ | $1.000$ |
| $K(P2,P3)$ | $20$ | $e^{-10} \approx 0.0000454$ |
| $K(P2,P4)$ | $18$ | $e^{-9} \approx 0.000123$ |
| $K(P3,P3)$ | $0$ | $1.000$ |
| $K(P3,P4)$ | $2$ | $e^{-1} = 0.368$ |
| $K(P4,P4)$ | $0$ | $1.000$ |

> **Key observation:** Cross-class kernel values ($K(P1,P3)$, $K(P1,P4)$, etc.) are nearly **zero** — healthy and diabetic patients are far apart, so their RBF similarity is negligible.

---

### 5.4 — $y^{(i)}y^{(j)}K_{ij}$ Matrix

| | P1 ($y=-1$) | P2 ($y=-1$) | P3 ($y=+1$) | P4 ($y=+1$) |
|---|---|---|---|---|
| **P1** | $+1.000$ | $+0.368$ | $-0.000123$ | $-0.0000454$ |
| **P2** | $+0.368$ | $+1.000$ | $-0.0000454$ | $-0.000123$ |
| **P3** | $-0.000123$ | $-0.0000454$ | $+1.000$ | $+0.368$ |
| **P4** | $-0.0000454$ | $-0.000123$ | $+0.368$ | $+1.000$ |

**Sum of all entries** (cross-class terms ≈ 0):

$$\approx 1.368 + 1.368 + 1.368 + 1.368 = 5.472$$

---

### 5.5 — Solve Dual for $\alpha$ (RBF)

$$W(\alpha) = 4\alpha - \frac{5.472}{2}\alpha^2 = 4\alpha - 2.736\alpha^2$$

$$\frac{dW}{d\alpha} = 4 - 5.472\alpha = 0 \quad \Rightarrow \quad \boxed{\alpha \approx 0.731}$$

Note the contrast across kernel types:
- Linear: $\alpha = 1/18 \approx 0.056$
- Polynomial: $\alpha \approx 0.00075$
- **RBF: $\alpha \approx 0.731$** — more weight on each support vector due to near-zero cross-class similarities

---

### 5.6 — Compute $b_{\text{RBF}}$ Using P3

$$\sum_j \alpha_j y^{(j)} K(\vec{x}^{(j)}, P3) = 0.731[-0.000123 - 0.0000454 + 1.000 + 0.368] = 0.731 \times 1.3678 \approx 1.000$$

From $(+1)(1.000 + b_{\text{RBF}}) = 1$: $\quad \boxed{b_{\text{RBF}} = 0}$

**Verify using P1:**
$$0.731[(-1)(1.000)+(-1)(0.368)+(+1)(0.000123)+(+1)(0.0000454)] = 0.731 \times (-1.368) \approx -1.000$$

$(-1)(-1.000 + 0) = 1$ ✓

---

### 5.7 — Predict $P_{\text{new}} = (3, 4)$ with RBF Kernel

**Squared distances from $(3,4)$ to each training point:**

| Training Point | Calculation | $d^2$ |
|---|---|---|
| P1: $(1,2)$ | $(3-1)^2+(4-2)^2$ | $8$ |
| P2: $(2,1)$ | $(3-2)^2+(4-1)^2$ | $10$ |
| P3: $(4,5)$ | $(3-4)^2+(4-5)^2$ | $2$ |
| P4: $(5,4)$ | $(3-5)^2+(4-4)^2$ | $4$ |

**RBF kernel values with $P_{\text{new}}$:**

| Training Point | $d^2$ | $K = e^{-0.5 d^2}$ | Interpretation |
|---|---|---|---|
| P1 (Healthy) | $8$ | $e^{-4} = 0.0183$ | Very low — far away |
| P2 (Healthy) | $10$ | $e^{-5} = 0.0067$ | Very low — far away |
| P3 (Diabetic) | $2$ | $e^{-1} = 0.368$ | High — close! |
| P4 (Diabetic) | $4$ | $e^{-2} = 0.135$ | Moderate — reasonably close |

**Decision function:**

$$f(3,4) = 0.731\left[(-1)(0.0183) + (-1)(0.0067) + (+1)(0.368) + (+1)(0.135)\right] + 0$$
$$= 0.731\left[-0.0183 - 0.0067 + 0.368 + 0.135\right] = 0.731 \times 0.478 = \mathbf{+0.350} > 0$$

$$\boxed{\text{RBF Kernel Prediction: Diabetic (+1)}} \checkmark$$

**Why the RBF made this decision — contribution breakdown:**

```
Patient  | Kernel Value | Label | Contribution (α × y × K)
---------|--------------|-------|---------------------------
P1       |   0.0183     |  −1   |  0.731×(−1)×0.0183 = −0.013  ← tiny negative
P2       |   0.0067     |  −1   |  0.731×(−1)×0.0067 = −0.005  ← tiny negative
P3       |   0.368      |  +1   |  0.731×(+1)×0.368  = +0.269  ← LARGE positive
P4       |   0.135      |  +1   |  0.731×(+1)×0.135  = +0.099  ← moderate positive
                                   ─────────────────────────────
                           Total:  +0.350 → Diabetic (+1)
```

> P3 and P4 are **close** to the new patient → high similarity → their diabetic votes dominate.
> P1 and P2 are **far** from the new patient → near-zero similarity → their healthy votes are negligible.
>
> **Core mechanism of RBF: nearby training points vote loudly, faraway ones whisper.**

---

## Complete Summary

| Stage | Method | $\vec{w}$ or $\alpha$ | $b$ | $f(3,4)$ | Prediction |
|---|---|---|---|---|---|
| 1 | Linear (Primal) | $\vec{w}=[1/3,\ 1/3]$ | $-2$ | $+1/3$ | Diabetic ✓ |
| 2 | Linear (Dual) | $\alpha=1/18$ each | $-2$ | $+1/3$ | Diabetic ✓ |
| 4 | Polynomial ($d=2$) | $\alpha=1/1332$ each | $-1.270$ | $+0.117$ | Diabetic ✓ |
| 5 | RBF ($\gamma=0.5$) | $\alpha=0.731$ each | $0$ | $+0.350$ | Diabetic ✓ |

All four methods agree — the new patient is **Diabetic**.

---

## The Mathematical Flow

```
PRIMAL                              DUAL
────────────────────────────────────────────────────────────
Minimize (1/2)||w||²                Maximize Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
Subject to: yᵢ(w·xᵢ+b) ≥ 1        Subject to: αᵢ≥0, Σαᵢyᵢ=0
                                              ↓
                                    w = Σαᵢyᵢxᵢ  ← KKT condition
                                              ↓
                          Decision: f(x) = Σαᵢyᵢ(xᵢ·x) + b
                                              ↓
                          ┌───────────────────┘
                          │  Replace (xᵢ·x) with K(xᵢ,x)
                          ↓
LINEAR KERNEL          POLYNOMIAL KERNEL        RBF KERNEL
K(xᵢ,xⱼ)=xᵢ·xⱼ       K=(xᵢ·xⱼ+1)²            K=e^(-γ||xᵢ-xⱼ||²)
Straight boundary      Curved boundary           Infinite-dim boundary
α=1/18 ≈ 0.056         α=1/1332 ≈ 0.00075        α=0.731
b=-2                   b=-1.270                  b=0
f(3,4)=+0.333          f(3,4)=+0.117             f(3,4)=+0.350
```

> The entire SVM — from margin width to RBF — is built on one idea: **express everything through dot products, then replace those dot products with kernel functions.** The dual formulation is the mathematical bridge that makes this substitution possible.

---