# 🤖 Machine Learning — From Basics to Beyond

> A complete, structured documentation of my end-to-end Machine Learning learning journey —
> from Python foundations and mathematical intuition, all the way through supervised learning,
> unsupervised learning, and explainable AI.

---

## 🏁 All Phases Completed ✅

| Phase | Topic | Status |
|---|---|---|
| **Phase 0** | Conceptual Orientation | ✅ Complete |
| **Phase 1** | Python & Mathematical Foundations | ✅ Complete |
| **Phase 2** | Data Preprocessing & Preparation | ✅ Complete |
| **Phase 3** | Machine Learning Fundamentals | ✅ Complete |
| **Phase 4** | Supervised Learning | ✅ Complete |
| **Phase 5** | Unsupervised Learning | ✅ Complete |

---

## 📖 What This Repository Contains

This repository is a **structured learning log** — not just code, but a complete record of concepts, intuitions, formulae, and interview-ready explanations for every topic in the ML roadmap below.

Each phase builds on the previous one. A beginner can follow this roadmap sequentially — no prior ML knowledge required.

> 💭 **[Q&A and Interview Guide](https://github.com/mahajanmuskaan/ml-from-basics-to-beyond/tree/main/ml-from-basics-to-beyond/QnA%20and%20Interview%20Guide)** — A dedicated space for quick interview revision and concept summaries across all phases.

---

## 🗺️ Complete Learning Roadmap

---

### 📌 Phase 0 — Conceptual Orientation

> *Before writing a single line of code, understand what ML actually is and where it fits in the world.*

| # | Topic |
|---|---|
| 1 | AI vs ML vs Deep Learning — understanding the hierarchy |
| 2 | Types of ML: Supervised, Unsupervised, Semi-Supervised, Reinforcement Learning |
| 3 | Real-life applications — healthcare, finance, NLP, computer vision |
| 4 | The ML Workflow — end-to-end overview: data → model → deployment |

---

### 📌 Phase 1 — Python & Mathematical Foundations

> *The tools and language of ML. Python is the ecosystem; math is the logic underneath.*

#### 🐍 Python for ML

| # | Topic |
|---|---|
| 1 | Why Python for ML? — ecosystem overview |
| 2 | Python basics: variables, loops, conditionals, functions |
| 3 | Data structures: lists, tuples, dictionaries, sets |
| 4 | NumPy — arrays, vectorised operations, broadcasting |
| 5 | Pandas — DataFrames, CSV handling, filtering, groupby |
| 6 | Data visualisation: Matplotlib & Seaborn basics |

#### 📐 Math Foundations for ML

| # | Topic |
|---|---|
| 1 | What is data? — vectors, matrices, scalars |
| 2 | Descriptive Statistics: mean, median, mode |
| 3 | Variance & standard deviation — spread of data |
| 4 | Correlation — understanding feature relationships |
| 5 | Basic Probability (intuition only): events, conditional probability |
| 6 | Why math matters in ML — demystifying the fear |

---

### 📌 Phase 2 — Data Preprocessing & Preparation

> *In real-world ML, 60–80% of the work happens here. Clean data beats clever algorithms.*

#### 📊 Dataset Understanding

| # | Topic |
|---|---|
| 1 | What is a dataset? Rows (samples) vs Columns (features) |
| 2 | Structured vs Unstructured data |
| 3 | Types of features: numerical, categorical, ordinal, binary |

#### 🧹 Data Cleaning

| # | Topic |
|---|---|
| 1 | Handling missing values: Mean / Median / Mode imputation |
| 2 | Forward fill and Backward fill |
| 3 | Dropping rows or columns — when and why |

#### 🔍 Outlier Detection & Handling

| # | Topic |
|---|---|
| 1 | IQR method and Z-score method |
| 2 | Winsorisation and capping |

#### 🔤 Encoding Categorical Data

| # | Topic |
|---|---|
| 1 | One-Hot Encoding — for nominal categories |
| 2 | Label Encoding — for tree-based models |
| 3 | Ordinal Encoding — for categories with a natural order |

#### ⚙️ Feature Engineering & Scaling

| # | Topic |
|---|---|
| 1 | Feature Engineering — creating new features from existing ones |
| 2 | Normalisation (Min-Max Scaling) — scales features to [0, 1] |
| 3 | Standardisation (Z-score) — transforms to mean=0, std=1 |
| 4 | Polynomial features — fitting non-linear relationships with linear models |

#### ✂️ Dataset Splitting & Leakage

| # | Topic |
|---|---|
| 1 | Train-Test Split — purpose and ratio choices |
| 2 | Train / Validation / Test Split — the 3-way split |
| 3 | Data leakage — what it is and why it destroys model validity |
| 4 | Why data quality matters more than model sophistication |

---

### 📌 Phase 3 — Machine Learning Fundamentals

> *The core ideas that power every algorithm. Understand these deeply and everything else makes sense.*

| # | Topic |
|---|---|
| 1 | What is a model? — mathematical function mapping inputs to outputs |
| 2 | Training vs Testing — the learning process vs evaluation |
| 3 | Overfitting — model learns noise, poor generalisation |
| 4 | Underfitting — model too simple, poor even on training data |
| 5 | Bias–Variance Tradeoff — the fundamental tension in ML |
| 6 | Cost Functions — the target that training minimises (MSE, Log Loss) |
| 7 | Gradient Descent — how models learn iteratively by minimising cost |
| 8 | Learning Rate — controls step size during gradient descent |
| 9 | Regularisation — penalising complexity to prevent overfitting |
| | &nbsp;&nbsp;&nbsp;→ L2 (Ridge) — shrinks weights, does not zero them out |
| | &nbsp;&nbsp;&nbsp;→ L1 (Lasso) — can zero out weights, acts as feature selection |

---

### 📌 Phase 4 — Supervised Learning

> *The most widely used category of ML. Algorithms that learn from labelled examples.*

---

#### 📈 Block 1 — Regression Algorithms

**1. Linear Regression**

| Sub-topic |
|---|
| Intuition — fitting a line to minimise prediction error |
| Cost Function — Mean Squared Error (MSE) |
| Gradient Descent — conceptual walkthrough and basic derivation |
| Assumptions of Linear Regression |
| Multiple Feature Linear Regression — one hyperplane for multiple features |
| Gradient Descent for multiple features |
| Feature Scaling — essential for faster convergence |
| Polynomial Regression — non-linear patterns with linear machinery |

**2. Regression Evaluation Metrics**

| Metric | What It Measures |
|---|---|
| MAE (Mean Absolute Error) | Average absolute difference — robust to outliers |
| MSE (Mean Squared Error) | Penalises large errors more heavily |
| RMSE (Root Mean Squared Error) | Same unit as the target variable — interpretable |
| R² (R-Squared) | Proportion of variance explained by the model |

---

#### 🎯 Block 2 — Classification Algorithms

**3. Logistic Regression**

| Sub-topic |
|---|
| Why not use linear regression for classification? |
| The Sigmoid function — maps any value to (0, 1) probability |
| Decision boundary — linear and non-linear |
| Log Loss (Binary Cross-Entropy) — the cost function |
| Gradient Descent for classification |
| Regularisation in Logistic Regression — preventing overfitting |

**4. Classification Evaluation Metrics**

| Metric | What It Measures |
|---|---|
| Confusion Matrix | TP, TN, FP, FN — the four outcomes |
| Accuracy | Overall correctness — misleading on imbalanced datasets |
| Precision | Of all predicted positives, how many are correct? |
| Recall | Of all actual positives, how many did we catch? |
| F1-Score | Harmonic mean of Precision and Recall — balances the tradeoff |
| ROC-AUC | Model's ability to discriminate at all possible thresholds |

**5. K-Nearest Neighbors (KNN)**

| Sub-topic |
|---|
| Intuition — classify by majority vote of K nearest neighbours |
| Distance metrics — Euclidean, Manhattan, Minkowski |
| Choosing K — too small = noise-sensitive; too large = over-smoothed |
| Curse of Dimensionality — why KNN breaks in high dimensions |
| Lazy learner — no training phase; all computation at inference time |

**6. Support Vector Machines (SVM)**

| Sub-topic |
|---|
| Margin intuition — maximise the gap between classes |
| Support vectors — the critical data points defining the margin |
| Hard margin vs Soft margin (C parameter) |
| Kernel trick — mapping to higher dimensions (RBF, Polynomial) |
| When to use SVMs vs neural networks |

---

#### 🌲 Block 3 — Tree-Based Models

**7. Naive Bayes**

| Sub-topic |
|---|
| Conditional probability — P(class \| features) |
| The 'naive' assumption — features are conditionally independent |
| Why it works despite the naive assumption |
| Gaussian, Multinomial, and Bernoulli variants |
| Laplace Smoothing — preventing zero probabilities |

**8. Decision Trees**

| Sub-topic |
|---|
| How trees split — choosing the best feature at each node |
| Entropy and Information Gain — measuring purity after splits |
| Gini Impurity — an alternative splitting criterion |
| Variance Reduction — for regression trees |
| Overfitting in trees — why deep trees memorise training data |

**9. Random Forests**

| Sub-topic |
|---|
| Bagging (Bootstrap Aggregating) — training on random subsets |
| Random feature selection at each split — increases tree diversity |
| Why Random Forest reduces overfitting vs a single tree |
| Out-of-Bag (OOB) error — free internal validation |
| Feature importance from Random Forest |
| Parallelisable — trees trained independently |

**10. Ensemble Methods & XGBoost**

| Sub-topic |
|---|
| Ensemble Learning — combining weak learners for a stronger model |
| Bagging (Random Forest) vs Boosting (XGBoost) — the core difference |
| Gradient Boosting — sequential trees, each correcting the previous |
| XGBoost — regularised gradient boosting with second-order gradients |
| LightGBM — faster gradient boosting with leaf-wise growth |
| CatBoost — native categorical feature handling |
| When to use Trees vs Neural Networks |
| &nbsp;&nbsp;&nbsp;→ Trees / XGBoost: structured / tabular data, smaller datasets |
| &nbsp;&nbsp;&nbsp;→ Neural Networks: unstructured data — images, text, audio |

---

#### ✅ Block 4 — Model Validation & Improvement

**Model Validation Techniques**

| # | Topic |
|---|---|
| 1 | Cross-Validation — why simple train-test split is often not enough |
| 2 | K-Fold Cross-Validation — rotating K subsets for robust evaluation |
| 3 | Stratified K-Fold — preserving class balance in each fold |
| 4 | Learning Curves — diagnosing overfitting vs underfitting visually |
| 5 | Error Analysis — systematic approach to model improvement |

**Hyperparameter Tuning**

| # | Topic |
|---|---|
| 1 | Hyperparameters vs Parameters — what is the difference? |
| 2 | Grid Search — exhaustive search over a parameter grid |
| 3 | Random Search — random sampling, often faster and equally effective |
| 4 | Bayesian Optimisation — intelligent search guided by prior results |
| 5 | Learning Rate Scheduling and Decay |

---

#### 🔍 Block 5 — Explainable AI (XAI)

| # | Topic |
|---|---|
| 1 | Why Explainability matters — trust, regulation, and debugging |
| 2 | SHAP (SHapley Additive exPlanations) — per-prediction feature attribution |
| 3 | LIME (Local Interpretable Model-agnostic Explanations) — local surrogate models |
| 4 | PDP (Partial Dependence Plots) — average effect of a feature on predictions |
| 5 | Feature Importance from tree-based models |
| 6 | Black-box vs White-box models |

---

### 📌 Phase 5 — Unsupervised Learning

> *Learning from unlabelled data — discovering hidden patterns and structure.*

#### 🔵 K-Means Clustering

| # | Topic |
|---|---|
| 1 | How centroids initialise and move iteratively |
| 2 | The Elbow Method — choosing the optimal number of clusters K |
| 3 | Limitations: sensitive to initialisation and outliers |

#### 🚨 Anomaly Detection

| # | Topic |
|---|---|
| 1 | Gaussian distribution to model 'normal' behaviour |
| 2 | Setting the epsilon threshold for flagging anomalies |
| 3 | Applications: fraud detection, manufacturing defects, network intrusion |

---

## 📚 Courses & References

### 🎓 Primary Course

| Course | Purpose |
|---|---|
| [Andrew Ng — Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) *(Coursera)* | Core reference for Supervised Learning, Unsupervised Learning, Neural Networks, Recommender Systems, and Reinforcement Learning |

### 📺 Topic-Specific References

| Reference | Used For |
|---|---|
| [Data Cleaning Full Course — WsCube Tech](https://www.youtube.com/watch?v=ITy8R4278sk) | Phase 2: Complete guide to Data Handling, Cleaning & Preprocessing |
| YouTube (topic-specific videos) | Additional clarity and alternate explanations on individual concepts |

### 🛠️ Tools & Workflow

| Tool | Purpose |
|---|---|
| [Claude (Anthropic)](https://claude.ai) | Concept clarification, note structuring, and generating study material |
| [ChatGPT (OpenAI)](https://chatgpt.com) | Concept clarification and alternate explanations |

---

## 💡 Who This Is For

This roadmap is designed for **absolute beginners** who want a structured, no-gap path into Machine Learning:

- **Students** starting their ML journey with no prior background
- **Developers** transitioning into data science or AI roles
- **Researchers** wanting a solid classical ML foundation before deep learning
- **Interview candidates** looking for a comprehensive, revision-ready reference


---

*Built with curiosity, structured with intention — one concept at a time.*
