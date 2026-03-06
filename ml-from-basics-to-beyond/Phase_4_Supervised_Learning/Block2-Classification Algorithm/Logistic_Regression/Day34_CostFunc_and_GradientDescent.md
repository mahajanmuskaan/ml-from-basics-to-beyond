# 📘 Day 34 – Logistic Regression  
## Cost Function & Gradient Descent  

---

### 1️⃣ Decision Boundary (Quick Recap)

For logistic regression, the model predicts:

\[
\hat{y} = g(z) = \frac{1}{1 + e^{-z}}
\]

Where:  
\[
z = w^T x + b
\]

The **decision boundary** is defined as:

\[
w^T x + b = 0
\]

#### Example (Polynomial Feature Case)

If the decision boundary is:

\[
x_1^2 + x_2^2 - 1 = 0
\]

This represents a **circle of radius 1**.  
- Inside circle → Class 1  
- Outside circle → Class 0  

---

### 2️⃣ Why Not Use Squared Error?

In Linear Regression we use:

\[
J(w,b) = \frac{1}{2m} \sum ( \hat{y}^{(i)} - y^{(i)} )^2
\]

But in Logistic Regression:
- The hypothesis uses the **sigmoid (non-linear)** function.  
- Squared error makes the cost function **non-convex**.  
- Gradient Descent may **get stuck in local minima**.  

👉 Hence, we use a different cost function.  

---

### 3️⃣ Logistic Loss (Single Training Example)

#### Case 1: When \( y = 1 \)

\[
L(\hat{y}, y) = -\log(\hat{y})
\]

- If \( \hat{y} = 1 \) → Loss = 0 ✅  
- If \( \hat{y} = 0 \) → Loss → ∞ ❌  

#### Case 2: When \( y = 0 \)

\[
L(\hat{y}, y) = -\log(1 - \hat{y})
\]

- If \( \hat{y} = 0 \) → Loss = 0 ✅  
- If \( \hat{y} = 1 \) → Loss → ∞ ❌  

👉 The model is **heavily penalized for confident wrong predictions**.  

---

### 4️⃣ Simplified Logistic Loss Formula

We combine both cases into one equation:

\[
L(\hat{y}, y) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
\]

Where:

\[
\hat{y} = g(w^T x + b)
\]

---

### 5️⃣ Full Cost Function (Binary Cross Entropy)

For \( m \) training examples:

\[
J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
\]

This is called **Binary Cross Entropy Loss**.

**Properties:**
- Convex function  
- Has one global minimum  
- Gradient Descent works reliably  

---

### 6️⃣ Gradient Descent for Logistic Regression

We update parameters to minimize the cost:

\[
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]

Where \( \alpha \) = **learning rate**.  

---

#### 6.1 Derivatives

After applying the chain rule:

\[
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
\]
\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
\]

---

#### 6.2 Final Update Rules

\[
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
\]
\[
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
\]

🔥 **Important Observation:**  
The gradient update formula looks **very similar to linear regression**,  
but the **prediction mechanism is different**:

\[
\hat{y} = \frac{1}{1 + e^{-(w^T x + b)}}
\]


