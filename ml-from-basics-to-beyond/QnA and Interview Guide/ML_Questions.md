# 🎓 Top ML Interview Questions & Answers
---

## 📌 SECTION 1: Foundations of AI & ML

---

### Q1. What is the difference between AI, ML, and Deep Learning?

**Answer:**

- **AI (Artificial Intelligence)** is the broad concept of making machines smart enough to mimic human behavior — like thinking, reasoning, and problem-solving.
- **ML (Machine Learning)** is a *subset* of AI where machines learn from data without being explicitly programmed.
- **Deep Learning** is a *subset* of ML that uses multi-layered neural networks to learn complex patterns automatically.

Think of it as **AI ⊃ ML ⊃ Deep Learning** — each is a more specific version of the previous.

**Real-World Example:**
When you use Gmail, AI handles the whole email system. ML helps it detect spam by learning from labeled emails. Deep Learning powers smart replies by understanding the meaning of full sentences.

---

### Q2. What are the main types of Machine Learning? Give real-world examples of each.

**Answer:**

| Type | How it works | Example |
|------|-------------|---------|
| **Supervised Learning** | Learns from labeled data (input → output pairs) | Predicting house prices |
| **Unsupervised Learning** | Finds patterns in unlabeled data | Customer segmentation |
| **Reinforcement Learning** | An agent learns by getting rewards/penalties | Game-playing AI (Chess, Go) |
| **Semi-supervised Learning** | Mix of labeled + unlabeled data | Medical image classification |
| **Self-supervised Learning** | Creates its own labels from data | GPT-style language models |

**Real-World Example:**
Netflix uses **supervised learning** to predict ratings, **unsupervised learning** to group users with similar tastes, and **reinforcement learning** to optimize which thumbnail to show.

---

### Q3. Can you describe the end-to-end ML pipeline at a high level?

**Answer:**

An ML pipeline has these key stages:

1. **Problem Definition** — What are we trying to predict or solve?
2. **Data Collection** — Gather relevant data from sources
3. **Data Preprocessing** — Clean, handle missing values, encode categories
4. **Exploratory Data Analysis (EDA)** — Understand patterns and distributions
5. **Feature Engineering** — Create/select meaningful input features
6. **Model Selection & Training** — Choose algorithm and fit to data
7. **Model Evaluation** — Measure performance on test/validation data
8. **Hyperparameter Tuning** — Optimize model settings
9. **Deployment** — Serve the model in production
10. **Monitoring** — Track performance over time, retrain if needed

**Real-World Example:**
A bank building a loan default predictor: collects customer data → cleans it → engineers features like debt-to-income ratio → trains a Random Forest → evaluates using AUC → deploys via an API → monitors for performance drift monthly.

---

### Q4. What is the difference between supervised and unsupervised learning?

**Answer:**

- **Supervised Learning:** The training data has *labels* (correct answers). The model learns to map inputs to outputs. Goal: predict or classify new data.
- **Unsupervised Learning:** The training data has *no labels*. The model finds hidden patterns or groupings on its own. Goal: discover structure in data.

**Real-World Example:**
In an email system — if you tag emails as "spam" or "not spam" and train a model on that, it's **supervised**. If you just give the model all emails and ask it to find natural groupings (promotional, work, personal), that's **unsupervised** (clustering).

---

### Q5. When would you choose reinforcement learning over supervised learning?

**Answer:**

Choose **Reinforcement Learning (RL)** when:
- There is no labeled dataset of correct actions
- The system must **learn by interacting** with an environment
- Actions have **delayed consequences** (long-term reward matters)
- The task is **sequential decision-making**

Choose **Supervised Learning** when you have labeled examples of input-output pairs.

**Real-World Example:**
Training a robot to walk — you can't label every joint movement as "correct." Instead, the robot tries movements, gets a reward when it moves forward without falling, and gradually learns the right strategy. This is RL. Supervised learning can't easily handle this because there's no "correct answer" dataset of walking movements.

---

## 📌 SECTION 2: Math & Statistics

---

### Q6. What is NumPy broadcasting and why is it useful?

**Answer:**

Broadcasting is NumPy's ability to perform operations on arrays of **different shapes** without copying data. It automatically "stretches" smaller arrays to match the shape of larger ones — in a memory-efficient way.

**Rules:** NumPy compares dimensions from the right. A dimension of 1 or a missing dimension is "broadcastable."

**Real-World Example:**
You have a dataset of 1000 images, each 28×28 pixels (shape: 1000×28×28). To normalize by subtracting the mean of each pixel (shape: 28×28), you just write:

```python
normalized = images - mean_image  # broadcasting handles the 1000 automatically
```

Without broadcasting, you'd need to loop over all 1000 images — much slower.

---

### Q7. What is the difference between variance and standard deviation?

**Answer:**

- **Variance** measures how spread out values are from the mean — it's the *average of squared differences* from the mean. Unit: squared units (e.g., cm²).
- **Standard Deviation (SD)** is just the *square root of variance* — it's in the **same units as the data**, making it more interpretable.

**Formula:**
- Variance: σ² = Σ(xᵢ - μ)² / N
- SD: σ = √(Variance)

**Real-World Example:**
A teacher checks student scores. Mean = 70. If scores are spread widely (40–100), variance and SD will be high. If scores are clustered (65–75), both will be low. The teacher uses **SD** (not variance) to communicate spread because "scores vary by ±10 points" is more meaningful than "variance is 100 square-points."

---

### Q8. What does correlation tell us, and what are its limitations?

**Answer:**

**Correlation** measures the *strength and direction* of a linear relationship between two variables. It ranges from **-1 to +1**:
- +1: Perfect positive relationship
- 0: No linear relationship
- -1: Perfect negative relationship

**Limitations:**
- It measures only **linear** relationships — misses non-linear patterns
- **Correlation ≠ Causation** — two variables can be correlated by coincidence
- Sensitive to **outliers**
- Doesn't tell us about the **scale** of the relationship

**Real-World Example:**
Ice cream sales and drowning rates are positively correlated — but ice cream doesn't cause drowning. Both go up in summer (a hidden third variable: hot weather). This is the classic "spurious correlation" warning.

---

## 📌 SECTION 3: Python & Data Tools

---

### Q9. Why is Python the dominant language for ML?

**Answer:**

Python dominates ML because of:
1. **Simple, readable syntax** — easy to prototype quickly
2. **Rich ecosystem** — NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
3. **Large community** — massive support, tutorials, and Stack Overflow answers
4. **Interoperability** — works well with C/C++ for performance-critical code
5. **Jupyter Notebooks** — ideal for exploratory analysis and teaching

**Real-World Example:**
A data scientist at Google can go from raw data exploration in Pandas → model building in Scikit-learn → deep learning in TensorFlow → deployment with Flask — all in Python, without switching languages. This seamless workflow is hard to match in R or Java.

---

### Q10. What is a DataFrame and how is it used in data analysis?

**Answer:**

A **DataFrame** (from Pandas library) is a 2D table-like data structure with **labeled rows and columns** — think of it as a Python version of an Excel spreadsheet or SQL table. It's the core object for data analysis in Python.

Common operations: filtering rows, selecting columns, groupby aggregation, merging tables, handling missing values.

**Real-World Example:**
A hospital loads patient records into a DataFrame — each row is a patient, columns are age, diagnosis, test results. They can then filter: `df[df['age'] > 60]` to select elderly patients, or `df.groupby('diagnosis')['test_score'].mean()` to find average scores per condition — all in one line.

---

## 📌 SECTION 4: Data Preprocessing

---

### Q11. How do you handle missing values in a dataset?

**Answer:**

**Strategies:**

| Approach | When to use |
|----------|------------|
| **Drop rows/columns** | If very few rows are missing OR a column has >70% missing |
| **Mean/Median imputation** | Numerical data, no major skew (use median if skewed) |
| **Mode imputation** | Categorical data |
| **Forward/Backward fill** | Time-series data |
| **Model-based imputation (KNN, MICE)** | When missingness has complex patterns |
| **Flag as a new category** | When "missing" itself is informative |

**Real-World Example:**
In a credit scoring dataset, income is missing for 10% of users. Using **median imputation** is safer than mean (because income is right-skewed — a few billionaires pull the mean up). But if we notice that income is missing mainly for self-employed users, creating an "income_missing" flag column captures that signal too.

---

### Q12. What is the difference between normalization and standardization? When do you use each?

**Answer:**

- **Normalization (Min-Max Scaling):** Scales values to a fixed range, typically [0, 1].
  - Formula: x' = (x - min) / (max - min)
  - Use when: data has no extreme outliers and algorithm is sensitive to ranges (KNN, Neural Networks)

- **Standardization (Z-score Scaling):** Centers data at mean 0 with SD of 1.
  - Formula: x' = (x - μ) / σ
  - Use when: data has outliers or algorithm assumes normal distribution (SVM, Linear Regression, PCA)

**Real-World Example:**
Building a model to predict house prices using area (100–5000 sqft) and number of rooms (1–10). Without scaling, gradient descent will update weights unevenly. **Standardization** is preferred here because housing data often has outliers (mansions), and algorithms like Linear Regression handle standardized features better.

---

### Q13. What is data leakage and why is it dangerous?

**Answer:**

**Data leakage** happens when information from outside the training data — specifically from the future or from the test set — accidentally gets included during training, making the model seem much better than it really is.

**Types:**
- **Target leakage:** Using a feature that is influenced by the target (e.g., using "treatment given" to predict "disease")
- **Train-test contamination:** Fitting a scaler or imputer on the full dataset before splitting

**Why dangerous:** The model shows artificially high accuracy during evaluation but **fails completely in production** — a costly and embarrassing failure.

**Real-World Example:**
A fraud detection model includes "was_account_frozen" as a feature to predict fraud. But accounts are frozen *because* of fraud — so this feature is a result of the target, not a predictor. The model gets 99% accuracy in testing but is useless in real-time detection where this info doesn't exist yet.

---

### Q14. What is one-hot encoding and when would you use label encoding instead?

**Answer:**

**One-Hot Encoding:** Converts each category into a separate binary (0/1) column.
- Example: Color {Red, Blue, Green} → three columns: is_Red, is_Blue, is_Green

**Label Encoding:** Converts each category into an integer.
- Example: Color {Red, Blue, Green} → {0, 1, 2}

**Use One-Hot when:** Categories have no natural order (nominal data) — e.g., City, Color, Gender. Using label encoding here implies Red < Blue < Green, which is wrong.

**Use Label Encoding when:** Categories have a natural order (ordinal data) — e.g., Size {Small=0, Medium=1, Large=2}, or with tree-based models that handle integers well.

**Real-World Example:**
Encoding customer "Education Level" (High School, Bachelor's, Master's, PhD) — use **label encoding** (0,1,2,3) because there's a clear order. But encoding "Favorite Sport" (Cricket, Football, Tennis) — use **one-hot encoding** because no sport is inherently "greater" than another.

---

### Q15. Why is feature engineering often more valuable than model selection?

**Answer:**

Feature engineering — transforming raw data into meaningful inputs — often has a **bigger impact on model performance** than switching between algorithms, because:
- A smart feature teaches the model domain knowledge it can't learn on its own
- Better features allow even simple models to perform well
- Raw data rarely contains exactly what the model needs

**Real-World Example:**
Predicting taxi trip duration: the raw features are pickup time, dropoff location, etc. A simple linear model with engineered features like "is_rush_hour," "day_of_week," and "distance_haversine" massively outperforms a complex neural network trained on raw latitude/longitude coordinates — because the engineer encoded the real-world knowledge.

---

### Q16. What are outliers and how do you detect/handle them?

**Answer:**

**Outliers** are data points that deviate significantly from the rest of the data. They can be:
- **Errors** (data entry mistakes)
- **Genuine extreme values** (billionaires in an income dataset)

**Detection Methods:**
- Z-score: flag points > 3 standard deviations from mean
- IQR Rule: flag points below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
- Box plots, scatter plots
- Isolation Forest (ML-based)

**Handling Options:**
- Remove if they're errors
- Cap/clip to a max value (Winsorization)
- Use robust algorithms (Tree-based models, Huber regression)
- Keep if genuine and important

**Real-World Example:**
In a salary dataset, a CEO's salary of $50M skews the mean enormously. A box plot reveals it as an outlier. Rather than removing it (it's real data!), you **log-transform** the salary column — this compresses the scale and makes the distribution more manageable without discarding valid data.

---

## 📌 SECTION 5: Core ML Concepts

---

### Q17. Explain the bias-variance tradeoff in simple terms.

**Answer:**

- **Bias** = error from wrong assumptions in the model. A high-bias model is too simple and **underfits** — it misses patterns in training data.
- **Variance** = error from the model being too sensitive to small fluctuations in training data. A high-variance model **overfits** — it memorizes training data but fails on new data.

**The Tradeoff:** Reducing bias increases variance and vice versa. The goal is to find the **sweet spot** — a model complex enough to learn real patterns, but not so complex it memorizes noise.

**Real-World Example:**
Predicting student exam scores:
- A model that just predicts everyone gets 70% → **high bias** (underfitting)
- A model that memorizes every student's past score but can't generalize → **high variance** (overfitting)
- A model using study hours, attendance, and past grades → **balanced bias-variance**

---

### Q18. What is the difference between overfitting and underfitting?

**Answer:**

| | Overfitting | Underfitting |
|--|-------------|--------------|
| **Definition** | Model learns training data *too well*, including noise | Model is too simple to capture patterns |
| **Training Error** | Very low | High |
| **Test Error** | Very high | High |
| **Cause** | Too complex model, too little data | Too simple model, too few features |
| **Fix** | Regularization, more data, pruning | More features, more complex model |

**Real-World Example:**
A spam classifier that memorizes every exact word in training spam emails will fail on new spam that uses slightly different wording — **overfitting**. A classifier that just checks if the email has the word "free" is too simple — **underfitting**. The right model learns patterns like sender domain + multiple spam keywords together.

---

### Q19. What is gradient descent and how does it work?

**Answer:**

**Gradient Descent** is an optimization algorithm used to minimize a loss function (error) by iteratively adjusting model parameters in the direction of steepest descent (negative gradient).

**Steps:**
1. Start with random weights
2. Calculate the loss (prediction error)
3. Compute the gradient (slope of loss w.r.t. weights)
4. Update weights: w = w - α × gradient
5. Repeat until loss is minimized

**Types:**
- **Batch GD:** Uses all data per update (slow but stable)
- **Stochastic GD (SGD):** Uses one sample per update (fast but noisy)
- **Mini-batch GD:** Uses small batches (best of both)

**Real-World Example:**
Imagine you're blindfolded on a hilly terrain trying to reach the lowest valley. At each step, you feel the slope under your feet and take a step downhill. Gradient descent does exactly this in the mathematical "loss landscape" — each step updates weights to reduce error, eventually reaching the minimum.

---

### Q20. What does regularization do and why is it needed?

**Answer:**

**Regularization** adds a **penalty term** to the loss function to discourage the model from learning overly complex patterns (large weights). It prevents overfitting by keeping the model simpler.

**Why needed:** Without regularization, models — especially complex ones — tend to assign very large weights to features, perfectly fitting training data but failing on unseen data.

**Real-World Example:**
A model predicting house prices starts assigning absurdly large weights to minor features like "house painted beige" because they happened to correlate with price in the training set. Regularization penalizes these large weights, forcing the model to focus only on truly important features like square footage and location.

---

### Q21. What is the difference between L1 and L2 regularization?

**Answer:**

| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| **Penalty** | Sum of absolute values of weights | Sum of squared weights |
| **Effect on weights** | Drives some weights to exactly **zero** | Shrinks all weights but rarely to zero |
| **Feature Selection** | Yes — acts as automatic feature selection | No |
| **Best when** | Many irrelevant features | All features likely relevant |

**Real-World Example:**
Predicting customer churn using 100 features — only 10 are truly relevant. **L1 (Lasso)** automatically zeroes out the 90 irrelevant features, giving a clean, interpretable model. **L2 (Ridge)** keeps all 100 features but with small weights — better when you believe all features contribute a little.

---

### Q22. How does learning rate affect gradient descent convergence?

**Answer:**

The **learning rate (α)** controls how big each step is during gradient descent.

- **Too high:** Takes large steps → overshoots the minimum → loss oscillates or diverges (never converges)
- **Too low:** Takes tiny steps → very slow convergence → might get stuck in local minima
- **Just right:** Steady convergence to the minimum

**Common techniques:** Learning rate schedules (reduce over time), Adam optimizer (adaptive learning rates per parameter).

**Real-World Example:**
Training a neural network to recognize handwritten digits. With learning rate = 10, weights fluctuate wildly and the model never learns. With learning rate = 0.000001, training takes weeks. With learning rate = 0.001 (typical default for Adam), the model converges in hours with good accuracy — the sweet spot.

---

## 📌 SECTION 6: Regression

---

### Q23. Why do we use RMSE instead of MAE in many scenarios?

**Answer:**

- **MAE (Mean Absolute Error):** Average of absolute differences. Treats all errors equally.
- **RMSE (Root Mean Squared Error):** Square root of average squared errors. **Penalizes large errors more heavily.**

We prefer RMSE when **large errors are especially costly** — because squaring amplifies big mistakes, making RMSE more sensitive to outlier predictions.

However, MAE is more robust and preferred when outliers are present in data.

**Real-World Example:**
Predicting electricity demand for a power grid: being off by 100 MW once is far more dangerous than being off by 10 MW ten times. **RMSE** captures this — the large single error gets penalized quadratically, encouraging the model to avoid rare but catastrophic misses.

---

### Q24. What assumptions does linear regression make?

**Answer:**

1. **Linearity:** Relationship between features and target is linear
2. **Independence:** Observations are independent of each other
3. **Homoscedasticity:** Constant variance of residuals (errors don't increase with prediction value)
4. **Normality of residuals:** Errors should be approximately normally distributed
5. **No multicollinearity:** Features should not be highly correlated with each other

**Real-World Example:**
Modeling salary from years of experience: if senior employees show much larger salary variation than juniors (heteroscedasticity), linear regression assumptions are violated. You'd need to log-transform salary or use a different model. Always check residual plots to verify these assumptions.

---

### Q25. What happens to gradient descent if you don't scale features?

**Answer:**

If features are on very different scales, the loss function becomes an **elongated ellipse** instead of a circle. Gradient descent takes inefficient zigzag steps along the elongated dimensions, converging very slowly or not at all.

Scaling features makes the loss surface more **circular**, so gradient descent can take direct, efficient steps toward the minimum.

**Real-World Example:**
Predicting house prices using area (100–10,000 sqft) and number of bedrooms (1–10). Area has 1000× the scale. Without scaling, gradient descent makes tiny updates to the area weight and huge updates to the bedroom weight, zigzagging without converging. After standardization, both features contribute equally and training converges ~10× faster.

---

### Q26. Explain R-squared. Can it ever be negative?

**Answer:**

**R² (coefficient of determination)** measures how much of the variance in the target variable is *explained* by the model.

- **R² = 1:** Perfect fit — model explains all variance
- **R² = 0:** Model does no better than predicting the mean every time
- **R² < 0:** Yes, it can be negative! This means the model is **worse than just predicting the mean** — a very bad model.

**Formula:** R² = 1 - (SS_residual / SS_total)

**Real-World Example:**
Predicting exam scores — if a model gives R² = 0.85, it explains 85% of the variance in scores (good!). If a poorly chosen model gives R² = -0.2, it means even a naive baseline (always predict class mean) would outperform it. This often signals something went wrong — wrong features, data leakage removed, or severe overfitting on training set.

---

### Q27. What is polynomial regression and when would you use it?

**Answer:**

**Polynomial regression** extends linear regression by adding polynomial (squared, cubed, etc.) terms of features, allowing it to fit **non-linear** relationships while still being a linear model in terms of coefficients.

Example: y = w₀ + w₁x + w₂x² + w₃x³

Use it when the relationship between input and output is curved, not straight.

**Real-World Example:**
Modeling the stopping distance of a car vs. speed: at low speeds it increases slowly, but at high speeds it increases very rapidly (non-linear). A straight line fits poorly. A **polynomial regression** with a squared speed term (distance ∝ speed²) fits perfectly — matching the physics of kinetic energy.

---

### Q28. What is multicollinearity and how does it affect linear regression?

**Answer:**

**Multicollinearity** occurs when two or more features are highly correlated with each other, making it hard for the model to distinguish their individual effects on the target.

**Effects:**
- Unstable, unreliable coefficient estimates
- Coefficients can flip signs or have huge standard errors
- Model predictions are still okay, but **interpretation is meaningless**

**Detection:** Variance Inflation Factor (VIF) — VIF > 10 indicates severe multicollinearity.

**Real-World Example:**
Predicting body fat percentage using both weight (kg) and BMI. Since BMI is calculated from weight and height, they're highly correlated. Including both confuses the model — the weight coefficient might appear negative when it should be positive. **Fix:** drop one feature, use PCA, or apply Ridge regression.

---

### Q29. Why can't we use linear regression for classification problems?

**Answer:**

Linear regression predicts **continuous values** (like 3.7 or -1.2), but classification needs **discrete class labels** (0 or 1). Problems include:

1. Output can go beyond [0,1] — meaningless for probabilities
2. It assumes a linear relationship between features and target — bad for classification boundaries
3. Sensitive to outliers — one extreme value can shift the decision boundary badly

**Real-World Example:**
Using linear regression to classify email as spam (1) or not spam (0): the model might predict -0.3 for some emails (less than 0) or 1.8 for others (more than 1) — these are invalid probabilities. **Logistic regression** solves this by squashing outputs to [0,1] using the sigmoid function.

---

## 📌 SECTION 7: Classification Metrics

---

### Q30. Explain precision and recall. When would you prioritize each?

**Answer:**

- **Precision** = Of all the *predicted positives*, how many are actually positive?
  - Formula: TP / (TP + FP)
  - Optimize when **false positives are costly**

- **Recall (Sensitivity)** = Of all the *actual positives*, how many did we correctly find?
  - Formula: TP / (TP + FN)
  - Optimize when **false negatives are costly**

**Real-World Example:**
- **Cancer screening:** Prioritize **Recall** — missing a real cancer case (false negative) is deadly. It's okay to flag some healthy patients for further tests.
- **Email spam filter:** Prioritize **Precision** — wrongly moving a legitimate email to spam (false positive) is annoying and costly. Better to let a few spams through.

---

### Q31. What is the ROC curve and what does AUC represent?

**Answer:**

The **ROC (Receiver Operating Characteristic) curve** plots **True Positive Rate (Recall)** vs. **False Positive Rate** at various classification thresholds, showing the tradeoff between sensitivity and specificity.

**AUC (Area Under the Curve)** summarizes the curve in a single number:
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.5:** Random classifier (useless)
- **AUC = 0.85:** Model is right 85% of the time when comparing a random positive and negative pair

AUC is **threshold-independent** — great for comparing models regardless of chosen cutoff.

**Real-World Example:**
Two models for predicting loan defaults: Model A has AUC = 0.92, Model B has AUC = 0.75. Even without picking a threshold, Model A is clearly better at *ranking* borrowers by default risk — it will consistently put actual defaulters higher in the risk ranking than non-defaulters.

---

### Q32. When does accuracy become a misleading metric?

**Answer:**

Accuracy is misleading on **imbalanced datasets** — where one class is far more common than the other. A model can achieve high accuracy by always predicting the majority class.

**Real-World Example:**
Detecting credit card fraud: 99.9% of transactions are legitimate, 0.1% are fraud. A model that predicts "not fraud" for *every* transaction gets **99.9% accuracy** but detects **zero frauds** — completely useless! Better metrics: Precision, Recall, F1-Score, or AUC-ROC.

---

## 📌 SECTION 8: Classification Algorithms

---

### Q33. What is the curse of dimensionality and how does it affect KNN?

**Answer:**

The **curse of dimensionality** refers to the exponential increase in data volume needed as dimensions (features) increase. In high dimensions:
- Data becomes increasingly sparse
- Distances between points become almost equal — making "nearest neighbor" meaningless
- Every point looks equally close to every other point

**Effect on KNN:** KNN relies entirely on distance between points. In high dimensions, distances lose meaning, so KNN performs poorly and becomes very slow.

**Real-World Example:**
In a 2D space (height, weight), finding the 5 nearest neighbors to a patient is meaningful. But with 500 genomic features, all patients start looking "equally distant" — KNN can't distinguish relevant neighbors. Solution: use PCA first to reduce dimensions.

---

### Q34. Explain the kernel trick in SVMs.

**Answer:**

SVMs find the best **hyperplane** to separate classes. But if data isn't linearly separable in its original space, the kernel trick maps data to a **higher-dimensional space** where it *is* linearly separable — without actually computing those higher-dimensional coordinates (computationally expensive).

**Common Kernels:** Linear, Polynomial, RBF (Radial Basis Function), Sigmoid

The trick: compute similarities in high-dimensional space **implicitly** using kernel functions — mathematically elegant and efficient.

**Real-World Example:**
Points on a 1D line: negatives in the middle, positives on both sides — not linearly separable. Map each point x to (x, x²) in 2D: now positives and negatives are easily separated by a horizontal line. The kernel function does this mapping mathematically without explicitly computing all those coordinates.

---

### Q35. What is the difference between hard-margin and soft-margin SVM?

**Answer:**

- **Hard-Margin SVM:** Requires *zero* training errors — all points must be correctly classified and outside the margin. Only works when data is **perfectly linearly separable**. Very sensitive to outliers.

- **Soft-Margin SVM:** Allows some misclassifications using a **slack variable (ξ)** and a penalty parameter **C**. More robust and works on real-world, noisy data.
  - High C → low tolerance for errors → smaller margin (may overfit)
  - Low C → more tolerance → larger margin (may underfit)

**Real-World Example:**
Classifying defective vs. good products using sensor data. One defective product has anomalous readings that look like a good product (outlier). Hard-margin SVM fails because no clean boundary exists. Soft-margin SVM allows this one misclassification and finds a robust boundary for all others.

---

### Q36. How does logistic regression use gradient descent?

**Answer:**

Logistic regression uses the **sigmoid function** to output probabilities, and **binary cross-entropy** as its loss function. Since this loss is convex, gradient descent can find the global minimum.

**Steps:**
1. Initialize weights randomly
2. Compute predictions using sigmoid: σ(wᵀx)
3. Calculate cross-entropy loss
4. Compute gradient of loss w.r.t. weights
5. Update: w = w - α × gradient
6. Repeat until convergence

**Real-World Example:**
Predicting if a student passes an exam based on study hours. Logistic regression initializes random weights, predicts probability of passing, compares with actual pass/fail outcomes, then adjusts weights via gradient descent — gradually getting better at estimating the study hours threshold that separates passing from failing students.

---

### Q37. What is entropy and how is it used in decision trees?

**Answer:**

**Entropy** measures the **impurity or disorder** in a dataset. Higher entropy = more mixed classes = more uncertain.

- **Formula:** H = -Σ p(c) log₂(p(c))
- Pure node (all one class): H = 0
- Maximum uncertainty (50/50 split): H = 1

Decision trees use entropy to decide **which feature to split on**: choose the feature that **reduces entropy the most** after the split (i.e., maximizes Information Gain).

**Real-World Example:**
A dataset has 50% spam and 50% non-spam → entropy = 1 (maximum). After splitting on "contains 'lottery'": one branch has 90% spam (low entropy), another has 10% spam (low entropy). This split dramatically reduces entropy — the tree will choose this feature first.

---

### Q38. What is the difference between Gini impurity and Information Gain?

**Answer:**

Both are criteria for splitting in decision trees — they measure how "pure" a split is.

| | Gini Impurity | Information Gain |
|--|--------------|-----------------|
| **Based on** | Probability of wrong classification | Entropy reduction |
| **Range** | 0 to 0.5 (binary) | 0 to 1 |
| **Speed** | Faster (no log) | Slightly slower |
| **Default in** | Scikit-learn (CART) | ID3, C4.5 algorithms |

Both usually produce similar trees. Gini is slightly faster; Information Gain is more theoretically grounded.

**Real-World Example:**
Splitting customers into "buy" vs. "not buy" groups. Both Gini and Information Gain will likely pick "income > 50K" as the best split if it cleanly separates the classes. The difference is mainly computational — Gini skips the log calculation and runs slightly faster for large datasets.

---

### Q39. Why does Random Forest reduce overfitting compared to a single tree?

**Answer:**

A single decision tree can grow very deep and memorize training data (overfit). Random Forest reduces overfitting through two mechanisms:

1. **Bagging (Bootstrap Aggregating):** Trains each tree on a random *subset* of training data (with replacement) — different trees see different data, so errors are uncorrelated.
2. **Random Feature Subsets:** At each split, only a random subset of features is considered — prevents all trees from learning the same dominant features.

The **ensemble average (voting)** of many diverse trees cancels out individual errors.

**Real-World Example:**
A single tree predicting stock movement might overfit to specific historical patterns. A Random Forest with 500 trees, each seeing different random data and features, creates diverse "opinions." When averaging their predictions, individual errors cancel out — like asking 500 independent stock analysts and averaging their views rather than trusting just one.

---

### Q40. What is the difference between Bagging and Boosting?

**Answer:**

| | Bagging | Boosting |
|--|---------|---------|
| **Training** | Trees trained **independently in parallel** | Trees trained **sequentially** |
| **Goal** | Reduce **variance** (overfitting) | Reduce **bias** (underfitting) |
| **Weights** | All samples weighted equally | Misclassified samples get higher weights |
| **Example** | Random Forest | XGBoost, AdaBoost, Gradient Boosting |
| **Risk** | Less likely to overfit | More prone to overfitting (if too many rounds) |

**Real-World Example:**
Bagging is like hiring 100 independent doctors and averaging their diagnoses. Boosting is like having each subsequent doctor focus specifically on the cases the previous doctors got wrong — specialists on the hard cases. Boosting often achieves higher accuracy but can overfit if not regularized.

---

### Q41. Why is XGBoost so popular in ML competitions?

**Answer:**

XGBoost (Extreme Gradient Boosting) wins competitions because it combines:

1. **Speed:** Parallel and distributed tree building + optimized data structures
2. **Regularization:** Built-in L1 & L2 regularization to prevent overfitting
3. **Handling missing values:** Automatically learns the best direction for missing data
4. **Tree pruning:** Uses "max_depth" and prunes trees bottom-up (more efficient)
5. **Cross-validation built-in:** Stops at optimal number of rounds
6. **Flexibility:** Works for regression, classification, ranking

**Real-World Example:**
In Kaggle competitions on structured/tabular data (like predicting house prices, customer churn, or loan defaults), XGBoost or LightGBM appear in ~70% of winning solutions. A typical data scientist can achieve top 10% results by using XGBoost with minimal tuning — it's that powerful out of the box.

---

### Q42. What is the 'naive' assumption in Naive Bayes?

**Answer:**

The "naive" assumption is that **all features are conditionally independent** given the class label. This means knowing the value of one feature gives you no information about another feature — a strong and often unrealistic assumption.

Despite this simplification (hence "naive"), the algorithm often works surprisingly well in practice.

**Real-World Example:**
Spam detection: Naive Bayes assumes that seeing "free" in an email is completely independent of seeing "lottery" in the same email. In reality, these words often appear together in spam. But even with this wrong assumption, Naive Bayes correctly classifies spam because the **combined probability signal** is still strong enough to separate spam from legitimate mail.

---

### Q43. When would you choose a tree-based model over a neural network?

**Answer:**

**Choose tree-based models (Random Forest, XGBoost) when:**
- Data is **tabular/structured**
- Dataset is **small to medium** (<100K rows typically)
- **Interpretability** is important (feature importance)
- Fast training and deployment needed
- Missing values and mixed data types present
- Limited compute resources

**Choose Neural Networks when:**
- Data is **unstructured** (images, text, audio)
- **Very large datasets** available
- Complex patterns (like spatial or temporal relationships) must be learned
- State-of-the-art performance is needed and compute is available

**Real-World Example:**
A bank predicting loan default on 50K customer records → **XGBoost** (fast, interpretable, great on tabular data). An autonomous car recognizing pedestrians in camera images → **Convolutional Neural Network** (only deep learning can learn spatial pixel patterns effectively).

---

## 📌 SECTION 9: Model Evaluation & Tuning

---

### Q44. What is K-Fold cross-validation and why is it better than a simple train-test split?

**Answer:**

**K-Fold CV** splits data into K equal parts ("folds"). The model trains on K-1 folds and tests on the remaining fold — this repeats K times, each time using a different fold as the test set. Final performance = average across all K runs.

**Why better than simple split:**
- Uses **all data for both training and testing** (nothing wasted)
- Gives a **more reliable estimate** of true performance
- Reduces variance in performance estimate
- Catches overfitting more reliably

**Real-World Example:**
Evaluating a cancer diagnosis model with only 500 patient records. A single 80/20 split wastes 100 records for testing and the result depends heavily on *which* 100 were chosen. **5-fold CV** tests on all 500 patients (100 at a time), giving a robust estimate — crucial when data is scarce, as in medical applications.

---

### Q45. What is the difference between hyperparameters and model parameters?

**Answer:**

| | Parameters | Hyperparameters |
|--|-----------|----------------|
| **Definition** | Values learned **from data** during training | Values set **before** training by the user |
| **Examples** | Weights, biases in neural networks; coefficients in regression | Learning rate, number of trees, depth of tree, C in SVM |
| **How set** | Optimization (gradient descent, etc.) | Manual tuning, Grid Search, Random Search, Bayesian Optimization |

**Real-World Example:**
Training a Random Forest to predict customer churn: the **number of trees** (e.g., 100 or 500) and **max depth** (e.g., 5 or 10) are **hyperparameters** — you choose them before training. The **split thresholds** and **feature importance weights** that the algorithm learns from data are **parameters**.

---

### Q46. What is Grid Search vs Random Search?

**Answer:**

Both are methods for **hyperparameter tuning** (finding the best settings for your model).

- **Grid Search:** Tries *every combination* of hyperparameter values in a specified grid. Exhaustive but very slow.
- **Random Search:** Samples *random combinations* from the hyperparameter space. Faster and often finds good results because most hyperparameters don't matter equally.

**Best practice:** Use Random Search first to narrow the range, then Grid Search in the promising region.

**Real-World Example:**
Tuning an SVM with C ∈ {0.1, 1, 10, 100} and gamma ∈ {0.001, 0.01, 0.1, 1} — Grid Search runs 16 combinations. With 5 hyperparameters and 5 values each, Grid Search = 3,125 combinations. **Random Search** testing 50 random combinations runs 62× faster and typically finds 90% of the optimal performance.

---

### Q47. How do you handle imbalanced datasets?

**Answer:**

**Strategies:**

1. **Resampling:**
   - Oversample minority class (SMOTE)
   - Undersample majority class
   - Combination of both

2. **Algorithm-level:**
   - Use `class_weight='balanced'` in Scikit-learn
   - Use cost-sensitive learning

3. **Evaluation:**
   - Use F1-score, AUC-ROC instead of accuracy
   - Use Precision-Recall curve

4. **Generate synthetic data (SMOTE)**

**Real-World Example:**
Detecting rare machine failures in a factory: 99% of readings are normal, 1% indicate failure. Training on raw data gives a model that always predicts "normal." Solution: undersample "normal" readings + oversample failures with SMOTE → balanced dataset → model that actually detects failures before costly breakdowns.

---

### Q48. What is SMOTE and when should you use it?

**Answer:**

**SMOTE (Synthetic Minority Over-sampling Technique)** creates *synthetic* (artificial) examples of the minority class by interpolating between existing minority examples — rather than simply duplicating them.

**How it works:** For each minority sample, find its K nearest minority neighbors and create new synthetic points along the lines connecting them.

**Use SMOTE when:** You have moderate imbalance AND enough minority samples to interpolate from. Don't use if minority class is extremely small (e.g., <5 samples) or if features are categorical.

**Real-World Example:**
Fraud detection dataset: 100,000 legitimate transactions and 500 frauds. Simply duplicating 500 frauds 200× overfits to those exact 500 cases. SMOTE creates 99,500 new *plausible* fraud examples by interpolating between existing frauds — the model learns the *region* of fraud, not just those specific 500 points.

---

### Q49. What is transfer learning and why is it powerful?

**Answer:**

**Transfer learning** uses a model **pre-trained on a large dataset** (like ImageNet) and *fine-tunes* it on a smaller, task-specific dataset. Instead of training from scratch, you start from a model that already understands low-level patterns.

**Why powerful:**
- Dramatically reduces training time and data requirements
- Works well even with small datasets (few thousand images vs. millions needed from scratch)
- Pre-trained models encode rich general knowledge

**Real-World Example:**
A hospital wants to detect pneumonia from chest X-rays but has only 5,000 labeled images. Training a CNN from scratch would require millions. Instead, they take **ResNet50 pre-trained on ImageNet** (general visual features), replace the final layer with a binary classifier, and fine-tune on 5,000 X-rays. Result: 92% accuracy — in hours, not weeks.

---

### Q50. How do learning curves help diagnose model problems?

**Answer:**

**Learning curves** plot training and validation error (or accuracy) as a function of **training set size**.

**Reading the curves:**

| Pattern | Problem | Fix |
|---------|---------|-----|
| Both errors high and close | Underfitting (high bias) | More complex model, better features |
| Large gap: low train error, high val error | Overfitting (high variance) | More data, regularization, dropout |
| Both converge nicely at low error | Good fit | You're done! |

**Real-World Example:**
Training an image classifier — learning curve shows training accuracy = 99%, validation accuracy = 70% with a large gap → classic **overfitting**. The fix: add dropout layers, data augmentation (flips, rotations), or collect more labeled images. Replot after fixes to confirm the gap narrows.

---

### Q51. What happens if features are highly correlated?

**Answer:**

**In Linear/Logistic Regression:** Multicollinearity makes coefficient estimates **unstable and unreliable** — small changes in data cause large swings in coefficients. Standard errors inflate, making significance tests meaningless.

**In Tree-based models:** Less problematic — trees naturally select features at each split, but highly correlated features reduce each other's apparent importance scores.

**In Neural Networks:** Generally handles correlation well through learned representations.

**Real-World Example:**
In a healthcare model, "systolic blood pressure" and "diastolic blood pressure" are highly correlated. Including both in linear regression causes the model to assign arbitrary large positive/negative weights to each (they cancel out). Fix: use only one, create a derived feature (pulse pressure = systolic − diastolic), or apply PCA.

---

## 📌 SECTION 10: Clustering & Unsupervised Learning

---

### Q52. How does K-Means clustering work? What is the Elbow Method?

**Answer:**

**K-Means Algorithm:**
1. Choose K (number of clusters)
2. Randomly initialize K centroids
3. Assign each point to the nearest centroid
4. Recalculate centroids as the mean of assigned points
5. Repeat steps 3-4 until centroids stop changing (convergence)

**Elbow Method:** Run K-Means for K = 1, 2, 3, ..., n and plot **Within-Cluster Sum of Squares (WCSS)** vs. K. The "elbow" — where adding more clusters gives diminishing returns — is the optimal K.

**Real-World Example:**
A supermarket segments customers for targeted marketing. Running K-Means for K=1 to 10 and plotting WCSS — the curve bends sharply at K=4. The 4 clusters reveal: budget shoppers, premium buyers, bulk buyers, and seasonal shoppers — each targeted with different promotions.

---

### Q53. What are the limitations of K-Means?

**Answer:**

1. **Requires K to be specified** in advance — unknown in practice
2. **Assumes spherical clusters** — fails on elongated or irregular shapes
3. **Sensitive to outliers** — one outlier can distort a centroid dramatically
4. **Sensitive to initialization** — different random starts give different results (use K-Means++)
5. **Doesn't handle varying cluster densities** well
6. **Only works with numerical data**

**Real-World Example:**
Using K-Means to cluster city neighborhoods by shape — real neighborhoods are irregular polygons, not circles. K-Means forces circular boundaries and misassigns many areas. **DBSCAN** (density-based clustering) handles arbitrary shapes much better in this geographic use case.

---

### Q54. What is a dendrogram and how do you read it?

**Answer:**

A **dendrogram** is a tree-like diagram produced by **hierarchical clustering**. It shows how data points are merged (or split) at each step.

**How to read it:**
- **Leaves (bottom):** Individual data points
- **Height of merge:** The distance at which two clusters were joined — higher = more different
- **Cut the tree** horizontally at a height to get your desired number of clusters — everything below the cut in the same branch belongs to one cluster

**Real-World Example:**
Clustering species by genetic similarity — a dendrogram shows humans and chimps merging at a low height (very similar), while birds and mammals merge at a much higher height (more different). Cutting the tree at height 50 might give you 3 clusters: mammals, birds, reptiles — a natural biological grouping.

---

### Q55. Explain PCA. What does it mean for a component to explain variance?

**Answer:**

**PCA (Principal Component Analysis)** finds new directions (**principal components**) in the data that capture the **maximum variance** — effectively summarizing the data with fewer dimensions while retaining most information.

**What "explain variance" means:** The first PC points in the direction of greatest spread in data. The second PC is perpendicular to the first and captures the next most spread. The "% variance explained" tells you how much information each component retains.

If PC1 explains 80% and PC2 explains 15% → together they capture 95% of all information in 2D.

**Real-World Example:**
A dataset of 1000 face images, each with 10,000 pixels (dimensions). PCA finds that just 50 principal components (called "eigenfaces") explain 95% of variance. Now faces can be represented as 50 numbers instead of 10,000 — 200× compression — enabling fast face recognition without sacrificing much accuracy.

---

### Q56. How is PCA different from feature selection?

**Answer:**

| | PCA | Feature Selection |
|--|-----|------------------|
| **What it does** | Creates *new* features (combinations of originals) | *Selects* existing features, discards others |
| **Interpretability** | Low — new features are abstract | High — original feature names retained |
| **Information** | Retains maximum variance | Retains original feature meaning |
| **When to use** | High-dimensional data, computational bottleneck | When interpretability matters, when specific features are irrelevant |

**Real-World Example:**
Medical diagnosis using 200 lab tests: **Feature selection** might pick the 20 most predictive tests (doctors understand what they mean). **PCA** creates 20 abstract "composite health indices" — harder to interpret but might capture subtle combined signals. For regulatory/audit purposes, feature selection wins; for raw predictive power, PCA might win.

---

### Q57. How would you detect anomalies using a Gaussian distribution model?

**Answer:**

**Steps:**
1. Assume features follow a Gaussian (normal) distribution
2. Estimate μ (mean) and σ² (variance) for each feature from training data
3. For a new data point, compute the probability p(x) using the Gaussian density function
4. If p(x) < threshold ε → flag as anomaly

Points with very low probability are in the "tail" of the distribution — statistically unlikely, hence anomalous.

**Real-World Example:**
Monitoring aircraft engine sensors: temperature, pressure, vibration. In normal operation, each sensor follows a Gaussian distribution. An anomaly detection system computes the probability of each new sensor reading. When an engine starts overheating (temperature deviates far from mean), p(x) drops below ε → alert triggered before engine failure.

---

## 📌 SECTION 11: Neural Networks & Deep Learning

---

### Q58. What is backpropagation and how does it use the chain rule?

**Answer:**

**Backpropagation** is the algorithm used to train neural networks by computing gradients of the loss function with respect to every weight, so gradient descent can update them.

**How it works:**
1. **Forward pass:** Compute predictions and loss
2. **Backward pass:** Propagate error backward through the network layer by layer
3. **Chain rule:** Since the network is a *composition of functions*, the gradient w.r.t. early layers = product of gradients through all subsequent layers (chain rule of calculus)

dL/dw₁ = (dL/dout) × (dout/dhidden) × (dhidden/dw₁)

**Real-World Example:**
In handwriting recognition, the loss is computed at the output layer. Backpropagation sends this error signal backward: "how much did the last hidden layer contribute to this error? The layer before that? The weights in layer 1?" — all computed via the chain rule. This tells us exactly how to nudge every single weight to reduce the error.

---

### Q59. Why is ReLU preferred over Sigmoid in hidden layers?

**Answer:**

**ReLU (Rectified Linear Unit):** f(x) = max(0, x)

**Reasons ReLU is preferred:**

1. **No vanishing gradient:** Sigmoid squashes outputs to (0,1), making gradients very small for extreme values → deep networks learn very slowly. ReLU has gradient = 1 for positive values.
2. **Faster computation:** Just a max operation vs. sigmoid's exponential computation
3. **Sparse activation:** Outputs exactly 0 for negative inputs → some neurons "turn off" → more efficient representations
4. **Empirically works better** in most deep learning tasks

**Real-World Example:**
Training a 50-layer deep network for image recognition with Sigmoid: gradients shrink by ~0.25× at each layer → after 50 layers, gradients near zero → early layers learn nothing (vanishing gradient problem). With **ReLU**, gradients flow freely → all 50 layers learn effectively → model achieves state-of-the-art accuracy.

---

### Q60. What is the vanishing gradient problem?

**Answer:**

During backpropagation, gradients are multiplied across layers. If each layer's gradient is a small number (<1) — as with Sigmoid which outputs gradients between 0 and 0.25 — the gradient **exponentially shrinks** as it travels backward through many layers.

By the time it reaches the early layers, the gradient is essentially zero → weights in early layers don't update → the network fails to learn.

**Solutions:** ReLU activation, batch normalization, residual connections (ResNets), LSTM/GRU for sequences, careful weight initialization (Xavier, He).

**Real-World Example:**
Early attempts to train deep sentiment analysis networks with Sigmoid activations failed — the first few embedding layers barely changed during training. Switching to ReLU and adding batch normalization allowed the model to effectively train 20+ layers, dramatically improving accuracy by learning richer text representations.

---

## 📌 SECTION 12: Reinforcement Learning

---

### Q61. What is the Bellman equation in RL?

**Answer:**

The **Bellman equation** expresses the value of a state as the **immediate reward** plus the discounted value of all future states reachable from it. It's the fundamental recursive relationship in RL.

**Formula:** V(s) = max_a [R(s,a) + γ × V(s')]

Where:
- V(s) = value of state s
- R(s,a) = immediate reward for taking action a in state s
- γ = discount factor (0 to 1, how much we value future rewards)
- s' = next state after action a

**Real-World Example:**
A robot navigating a warehouse: reaching the charging station gives reward +10. The Bellman equation says the value of any position = "how good is it to be here, considering I can earn rewards in the future?" Positions near the charger have high value because future rewards are close; positions far away have lower value. The robot learns to navigate by maximizing V(s).

---

### Q62. What is the exploration-exploitation tradeoff?

**Answer:**

In reinforcement learning, an agent must balance:

- **Exploration:** Trying new, unknown actions to discover potentially better rewards
- **Exploitation:** Using the best known action to maximize immediate reward

Too much exploration → never capitalizes on good strategies learned so far.
Too much exploitation → gets stuck in suboptimal strategies, never discovers better ones.

**Common strategy:** ε-greedy — with probability ε, explore randomly; with probability 1-ε, exploit best known action. ε decreases over time.

**Real-World Example:**
A recommendation system learning what movies users like: pure exploitation shows the same popular movies over and over (local optimum). Pure exploration shows random movies (bad user experience). A balanced approach occasionally recommends something new (exploration) while mostly showing movies the user is predicted to enjoy (exploitation) — gradually learning individual preferences.

---

## 📌 SECTION 13: Explainability & Recommender Systems

---

### Q63. What is SHAP and how does it explain individual predictions?

**Answer:**

**SHAP (SHapley Additive exPlanations)** uses game theory (Shapley values) to explain how much each feature contributed to a *specific* prediction. It answers: "For this particular patient/customer/transaction — which features pushed the prediction up, and which pushed it down?"

**Properties:**
- **Local:** Explains individual predictions, not just global feature importance
- **Fair:** Credit is distributed fairly using coalition game theory
- **Model-agnostic:** Works with any ML model

**Real-World Example:**
A bank's XGBoost model rejects a loan application. Regulatory compliance requires an explanation. SHAP output: "Loan rejected because: high debt-to-income ratio (−0.35 impact), short credit history (−0.20), but partially offset by high income (+0.15) and no previous defaults (+0.10)." The applicant gets a specific, fair, legally defensible explanation — impossible with just looking at feature importance globally.

---

### Q64. What is the difference between collaborative and content-based filtering?

**Answer:**

Both are recommendation system approaches:

**Collaborative Filtering:**
- Recommends based on **what similar users liked**
- Doesn't need to know *what* the item is — just who liked it
- Types: User-based CF, Item-based CF, Matrix Factorization
- **Problem:** Cold start — can't recommend to new users with no history

**Content-Based Filtering:**
- Recommends based on **the features of items** a user liked before
- Builds a profile of user preferences from item attributes
- Works for new users if they rate a few items, but needs good item features

**Hybrid:** Most modern systems (Netflix, Spotify) combine both.

**Real-World Example:**
- **Collaborative (Netflix):** "Users who watched 'Inception' also loved 'Interstellar' → recommend Interstellar to you."
- **Content-Based (Spotify):** "You like fast-tempo electronic music → here's another fast-tempo electronic track" — based on the song's audio features (BPM, genre, energy), regardless of what other users think.

---

## 📌 QUICK REVISION CHEAT SHEET

| Concept | One-Line Summary |
|---------|-----------------|
| AI > ML > DL | Nested subsets from broad to specific |
| Bias-Variance | Simple model underfits; complex model overfits |
| Overfitting | Great on train, poor on test |
| Gradient Descent | Step downhill on the loss surface iteratively |
| L1 vs L2 | L1 = zero weights (selection); L2 = small weights |
| Precision vs Recall | Precision = quality; Recall = coverage |
| Random Forest | Many trees + averaging = reduced variance |
| Bagging vs Boosting | Parallel (variance fix) vs Sequential (bias fix) |
| PCA | Compress dimensions while retaining most variance |
| SHAP | Fair, individual prediction explanation via game theory |
| Transfer Learning | Reuse pre-trained model knowledge on new task |
| SMOTE | Synthesize minority class samples to fix imbalance |
| Exploration-Exploitation | Try new things vs. use what works |
| Collaborative Filtering | Recommend based on similar users |
| Content-Based Filtering | Recommend based on item features |

---