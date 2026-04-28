# Course 3 Week 2 — Recommender Systems
*Reference: Andrew Ng's ML Specialization, Course 3 — Unsupervised Learning, Recommenders, Reinforcement Learning*

---

## The Big Picture — What is a Recommender System?

The fundamental problem:

- You have many **users**, many **items** (movies, products, songs), and a **sparse matrix of ratings** (most users rated very few items).
- You want to predict what rating a user would give an item they haven't seen, and recommend items they would likely enjoy.

**Movie Ratings Matrix** (`?` = not yet rated):

|              | Titanic | Avatar | LOTR | Inception | Interstellar |
|--------------|---------|--------|------|-----------|--------------|
| Alice        | 5       | 4      | ?    | 3         | ?            |
| Bob          | ?       | 5      | 4    | ?         | 5            |
| Carol        | 3       | ?      | 5    | 4         | ?            |
| Dave         | ?       | 2      | ?    | 5         | 4            |
| Eve          | 4       | ?      | 3    | ?         | ?            |

Week 2 covers two fundamentally different approaches:

| Approach | Description |
|---|---|
| **Collaborative Filtering** | "Users who liked similar things to you also liked X" — learn from the pattern of ratings |
| **Content-Based Filtering** | "Based on features of items you liked and who you are" — match user features to item features explicitly |

---

# PART 1 — Collaborative Filtering

## 1.1 Core Intuition

Collaborative Filtering is built on one powerful insight: you don't need to know anything about the *content* of items or users — just the **pattern of ratings**. Users reveal their preferences through their ratings, and similar users tend to rate similar items similarly.

The word "collaborative" means users **collaborate** — even unknowingly — to help each other get better recommendations.

```
Alice rated: Action movies high, Romance movies low
Bob  rated: Action movies high, Romance movies low
Carol rated: Action movies high, ???? (unrated)

→ Carol is similar to Alice and Bob
→ Carol probably also rates Romance movies low
→ Don't recommend Romance to Carol
```

No information about what the movies contain. Just the **pattern of ratings**.

---

## 1.2 Setup and Notation

| Symbol | Meaning |
|--------|---------|
| $n_u$ | Number of users |
| $n_m$ | Number of movies (items) |
| $r(i, j)$ | 1 if user $j$ has rated movie $i$, 0 otherwise |
| $y^{(i,j)}$ | Rating given by user $j$ to movie $i$ (only defined if $r(i,j)=1$) |
| $\vec{w}^{(j)},\ b^{(j)}$ | Parameters for user $j$ |
| $\vec{x}^{(i)}$ | Feature vector for movie $i$ |
| $n$ | Number of features per movie |
| $m^{(j)}$ | Number of movies rated by user $j$ |

---

## 1.3 Starting Point: Assuming We Have Movie Features

### The Dataset

5 movies, 4 users, rating scale 0–5:

| Movie | Alice | Bob | Carol | Dave | $x_1$ (Romance) | $x_2$ (Action) |
|-------|-------|-----|-------|------|-----------------|----------------|
| M1: Love Story         | 5 | 5 | 0 | 0 | 1.0  | 0.0 |
| M2: Romance Pt2        | 5 | ? | ? | 0 | 0.9  | 0.1 |
| M3: Cute Puppies       | ? | 4 | 0 | ? | 0.99 | 0.0 |
| M4: Nonstop Car Chases | 0 | 0 | 5 | 4 | 0.1  | 1.0 |
| M5: Swords & Karate    | 0 | 0 | 5 | ? | 0.0  | 0.9 |

Movie feature vectors:

$$\vec{x}^{(1)} = \begin{bmatrix}1.0 \\ 0.0\end{bmatrix}, \quad \vec{x}^{(2)} = \begin{bmatrix}0.9 \\ 0.1\end{bmatrix}, \quad \vec{x}^{(4)} = \begin{bmatrix}0.1 \\ 1.0\end{bmatrix}$$

### Prediction Model

For user $j$ rating movie $i$:

$$\boxed{\hat{y}^{(i,j)} = \vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)}}$$

Each user has their own parameter vector $\vec{w}^{(j)} \in \mathbb{R}^n$ and bias $b^{(j)}$.

**Example — Alice's parameters** (loves romance, hates action):

$$\vec{w}^{(\text{Alice})} = \begin{bmatrix}5 \\ 0\end{bmatrix}, \quad b^{(\text{Alice})} = 0$$

**Predict Alice's rating for M3** (Cute Puppies, $\vec{x}^{(3)} = [0.99,\ 0]^\top$):

$$\hat{y}^{(3,\text{Alice})} = \begin{bmatrix}5 \\ 0\end{bmatrix} \cdot \begin{bmatrix}0.99 \\ 0\end{bmatrix} + 0 = 4.95$$

Alice would likely give Cute Puppies nearly 5 stars. ✓

---

## 1.4 Cost Function for One User

Minimize the mean squared error over all movies user $j$ has rated:

$$J(\vec{w}^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:\,r(i,j)=1} \left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2$$

The notation $\sum_{i:\,r(i,j)=1}$ means **sum only over movies user $j$ has actually rated**.

With **regularization** to prevent overfitting:

$$J(\vec{w}^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:\,r(i,j)=1} \!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2m^{(j)}} \sum_{k=1}^{n} \left(w_k^{(j)}\right)^2$$

---

## 1.5 Cost Function for All Users

Extend to all $n_u$ users simultaneously:

$$\boxed{J = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:\,r(i,j)=1} \left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} \left(w_k^{(j)}\right)^2}$$

The $\frac{1}{m^{(j)}}$ is dropped for simplicity — it doesn't change the optimal parameters.

---

## 1.6 The Real Problem: We Don't Know the Movie Features

In practice, nobody manually labels movies with "romance = 0.9, action = 0.1" scores. Can we **learn** these features from ratings data?

### Flipping the Problem

If we knew user parameters $\vec{w}^{(j)}, b^{(j)}$, we could learn movie features $\vec{x}^{(i)}$ by minimizing:

$$J(\vec{x}^{(i)}) = \frac{1}{2} \sum_{j:\,r(i,j)=1} \left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} \left(x_k^{(i)}\right)^2$$

Across all movies:

$$J(\vec{x}^{(1)},\ldots,\vec{x}^{(n_m)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:\,r(i,j)=1} \left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} \left(x_k^{(i)}\right)^2$$

---

## 1.7 Collaborative Filtering: Learn Both Simultaneously

> *Key insight: We don't need to know features OR user parameters in advance. We can learn both at the same time from just the ratings.*

### The Combined Cost Function

$$\boxed{J = \frac{1}{2}\sum_{(i,j):\,r(i,j)=1} \!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}\!\left(w_k^{(j)}\right)^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}\!\left(x_k^{(i)}\right)^2}$$

Minimize jointly over $\vec{w}^{(1)}, \ldots, \vec{w}^{(n_u)},\ b^{(1)}, \ldots, b^{(n_u)},\ \vec{x}^{(1)}, \ldots, \vec{x}^{(n_m)}$.

Everything is learned from the **ratings matrix alone** — no content information needed.

---

## 1.8 Gradient Descent for Collaborative Filtering

Update all parameters simultaneously:

$$w_k^{(j)} := w_k^{(j)} - \alpha \frac{\partial J}{\partial w_k^{(j)}}$$

$$b^{(j)} := b^{(j)} - \alpha \frac{\partial J}{\partial b^{(j)}}$$

$$x_k^{(i)} := x_k^{(i)} - \alpha \frac{\partial J}{\partial x_k^{(i)}}$$

The partial derivatives:

$$\frac{\partial J}{\partial w_k^{(j)}} = \sum_{i:\,r(i,j)=1}\!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)x_k^{(i)} + \lambda w_k^{(j)}$$

$$\frac{\partial J}{\partial b^{(j)}} = \sum_{i:\,r(i,j)=1}\!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)$$

$$\frac{\partial J}{\partial x_k^{(i)}} = \sum_{j:\,r(i,j)=1}\!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} - y^{(i,j)}\right)w_k^{(j)} + \lambda x_k^{(i)}$$

---

## 1.9 Binary Labels: Likes and Dislikes

Many platforms use **binary feedback** — click/no-click, purchase/no-purchase, watched-50%/not, like/no-like.

### Adapting the Model

Apply the **sigmoid function** to get a probability:

$$\boxed{f\!\left(\vec{x}^{(i)}, \vec{w}^{(j)}, b^{(j)}\right) = g\!\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)}\right) = \frac{1}{1+e^{-\left(\vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)}\right)}}}$$

This predicts: *"What is the probability user $j$ engages with movie $i$?"*

### Loss Function for Binary Labels

Replace MSE with **Binary Cross-Entropy Loss**:

$$\mathcal{L}\!\left(f, y^{(i,j)}\right) = -y^{(i,j)}\log\!\left(f\right) - \left(1-y^{(i,j)}\right)\log\!\left(1 - f\right)$$

The full cost function:

$$J = \sum_{(i,j):\,r(i,j)=1} \mathcal{L}\!\left(f\!\left(\vec{x}^{(i)}, \vec{w}^{(j)}, b^{(j)}\right),\ y^{(i,j)}\right)$$

---

## 1.10 Mean Normalization: A Critical Practical Improvement

### The Problem With New Users

If Eve has rated no movies, her gradient is:

$$\frac{\partial J}{\partial w_k^{(\text{Eve})}} = \underbrace{\sum_{i:\,r(i,\text{Eve})=1}(\cdots)x_k^{(i)}}_{\text{empty sum} = 0} + \lambda \cdot 0 = 0$$

No gradient → no learning → $\vec{w}^{(\text{Eve})}$ stays at zero → all predictions ≈ 0.

### Mean Normalization Fixes This

**Step 1 — Compute mean rating for each movie:**

$$\mu_i = \frac{\displaystyle\sum_{j:\,r(i,j)=1} y^{(i,j)}}{\displaystyle\sum_j r(i,j)}$$

**Step 2 — Subtract the mean from every rating:**

$$y^{(i,j)}_{\text{norm}} = y^{(i,j)} - \mu_i$$

**Step 3 — Train the model on normalized ratings.**

**Step 4 — Add the mean back at prediction time:**

$$\boxed{\hat{y}^{(i,j)} = \vec{w}^{(j)} \cdot \vec{x}^{(i)} + b^{(j)} + \mu_i}$$

### Numerical Example

| Movie | Alice | Bob | Carol | Dave | Eve | $\mu_i$ |
|-------|-------|-----|-------|------|-----|---------|
| M1    | 5     | 5   | 0     | 0    | ?   | 2.5     |
| M2    | 5     | ?   | ?     | 0    | ?   | 2.5     |
| M3    | ?     | 4   | 0     | ?    | ?   | 2.0     |
| M4    | 0     | 0   | 5     | 4    | ?   | 2.25    |
| M5    | 0     | 0   | 5     | ?    | ?   | 1.67    |

**For Eve** (no ratings), parameters stay near zero:

$$\hat{y}^{(i,\text{Eve})} = \underbrace{\vec{w}^{(\text{Eve})} \cdot \vec{x}^{(i)} + b^{(\text{Eve})}}_{\approx\, 0} + \mu_i = \mu_i$$

Eve gets the **average rating** for each movie — a sensible default. As she rates movies, her parameters update and recommendations personalize.

---

## 1.11 Finding Related Items

Once movie features $\vec{x}^{(i)}$ are learned, find similar movies by computing:

$$\text{Distance}(i, k) = \left\| \vec{x}^{(i)} - \vec{x}^{(k)} \right\|^2 = \sum_{l=1}^{n} \left(x_l^{(i)} - x_l^{(k)}\right)^2$$

Movies $k$ with the **smallest distance** to movie $i$ are most similar. This is **item-item collaborative filtering**.

---

## 1.12 Limitations of Collaborative Filtering

### Cold Start Problem

- **New movie** with 0 ratings → no data to learn $\vec{x}^{(i)}$ → cannot estimate who would like it.
- **New user** with 0 ratings → no data to learn $\vec{w}^{(j)}$ → falls back to average (with mean normalization).

### Cannot Use Side Information Efficiently

Collaborative filtering only uses the ratings matrix — it ignores movie metadata (genre, director, cast) and user metadata (age, location, viewing history). This motivates **Content-Based Filtering**.

---

# PART 2 — Content-Based Filtering

## 2.1 Core Intuition

Instead of relying only on patterns in ratings, explicitly use **features of users** and **features of movies**. Match users to movies by comparing their feature vectors directly.

| | Collaborative Filtering | Content-Based Filtering |
|---|---|---|
| Flow | Rating pattern → learn hidden features → match | Known features → directly match |

---

## 2.2 Feature Vectors for Users and Movies

### User Feature Vector $\vec{x}_u^{(j)}$

$$\vec{x}_u^{(j)} = \begin{bmatrix} \text{age} \\ \text{gender (encoded)} \\ \text{country} \\ \text{avg rating given} \\ \text{top genres} \\ \vdots \end{bmatrix}$$

### Movie Feature Vector $\vec{x}_m^{(i)}$

$$\vec{x}_m^{(i)} = \begin{bmatrix} \text{year} \\ \text{genre (one-hot)} \\ \text{avg rating received} \\ \text{box office} \\ \text{director popularity} \\ \vdots \end{bmatrix}$$

> **Key point:** $\vec{x}_u^{(j)}$ and $\vec{x}_m^{(i)}$ can have **different sizes** — user features and movie features don't need to match in dimension.

---

## 2.3 The Content-Based Filtering Model

### Two-Network Architecture

```
User Features x_u^(j)          Movie Features x_m^(i)
      [28, 1, USA, ...]              [2010, Action, 7.8, ...]
            ↓                               ↓
    ┌──────────────┐              ┌──────────────────┐
    │  User Network│              │  Movie Network   │
    │  (Dense      │              │  (Dense layers)  │
    │   layers)    │              │                  │
    └──────────────┘              └──────────────────┘
            ↓                               ↓
    User Embedding v_u^(j)        Movie Embedding v_m^(i)
    [0.3, 0.8, 0.1, ...]          [0.2, 0.9, 0.1, ...]
         (32-dimensional)              (32-dimensional)
                    ↓                 ↓
                    v_u^(j) · v_m^(i)
                         ↓
                  Predicted Rating
```

### Prediction

$$\boxed{\hat{y}^{(i,j)} = \vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}}$$

- $\vec{v}_u^{(j)} \in \mathbb{R}^d$ — **user embedding** (compressed user representation)
- $\vec{v}_m^{(i)} \in \mathbb{R}^d$ — **movie embedding** (compressed movie representation)

If they are aligned (both love action movies) → high dot product → high predicted rating.

For **binary labels**, wrap in sigmoid:

$$\hat{y}^{(i,j)} = g\!\left(\vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}\right)$$

---

## 2.4 Neural Network Architecture

### User Network

| Layer | Units | Activation |
|-------|-------|------------|
| Input | size of $\vec{x}_u$ | — |
| Dense | 256 | ReLU |
| Dense | 128 | ReLU |
| Dense (output) | 32 | Linear |

Output: $\vec{v}_u^{(j)} \in \mathbb{R}^{32}$

### Movie Network

| Layer | Units | Activation |
|-------|-------|------------|
| Input | size of $\vec{x}_m$ | — |
| Dense | 256 | ReLU |
| Dense | 128 | ReLU |
| Dense (output) | 32 | Linear |

Output: $\vec{v}_m^{(i)} \in \mathbb{R}^{32}$

> **Only constraint:** the output embedding dimension must be the **same** ($d = 32$) so the dot product is defined.

### Cost Function

$$\boxed{J = \frac{1}{2}\sum_{(i,j):\,r(i,j)=1}\!\left(\vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)} - y^{(i,j)}\right)^2 + \text{regularization}}$$

Both networks are trained **jointly** via backpropagation — gradients flow through the dot product into both networks simultaneously.

---

## 2.5 Why Embeddings Are Powerful

After training, embedding vectors encode meaningful semantic information without anyone labeling what dimensions mean:

```
v_m(Inception)    ≈ v_m(Interstellar)   → similar embeddings → similar movies
v_m(Inception)    ≠ v_m(Titanic)        → different embeddings → different type
```

Finding similar movies after training:

$$\text{Similar movies to } i: \quad \underset{k}{\operatorname{argmin}}\ \left\| \vec{v}_m^{(i)} - \vec{v}_m^{(k)} \right\|^2$$

---

## 2.6 Collaborative Filtering vs Content-Based Filtering

| Dimension | Collaborative Filtering | Content-Based Filtering |
|-----------|------------------------|------------------------|
| Input needed | Ratings matrix only | User features + Movie features + Ratings |
| Cold start — new movie | Struggles | Can handle (has movie features) |
| Cold start — new user | Struggles | Can handle (has user features) |
| Side information | Ignores | Directly uses |
| Complexity | Simpler | More complex (needs feature engineering) |
| Scalability | Slower at prediction | Can precompute movie embeddings |
| Discovery | Better at surprising recommendations | Tends to recommend similar to past |

---

## 2.7 Scaling Up: Efficient Retrieval and Ranking

Computing $\vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}$ for **every** movie across **every** user request is too slow at millions of items.

### Two-Stage System

**Stage 1 — Retrieval (fast, approximate):** find ~100–1000 plausible candidates quickly.
- Precompute all $\vec{v}_m^{(i)}$ offline.
- At request time: compute $\vec{v}_u^{(j)}$, then run **Approximate Nearest Neighbor (ANN)** search to find ~100 close movies in milliseconds.

**Stage 2 — Ranking (slower, precise):** rank the candidates accurately.
- Run full model on ~100 candidates.
- Apply business logic: diversity, recency, freshness.
- Return top 10–20 recommendations.

```
All movies          Retrieve         Rank
(millions)    →    ~100–1000    →   Top 10–20
              (ANN search)     (full model + business rules)
```

**Retrieval** ensures speed. **Ranking** ensures accuracy.

---

# PART 3 — Principal Component Analysis (PCA)

## 3.1 Why PCA Appears Here

Recommender systems learn high-dimensional embedding vectors — visualizing and understanding them requires **dimensionality reduction**. PCA is the standard tool for this and also stands alone as a fundamental unsupervised learning technique.

---

## 3.2 Core Intuition

Many real-world features are **correlated** (e.g., car length ↔ wheelbase, engine size ↔ horsepower). The data doesn't really live in its nominal high-dimensional space — a few underlying factors explain most of the variation.

> *PCA finds new axes that capture the maximum variance in your data. Project the data onto these axes to get a lower-dimensional representation.*

---

## 3.3 The Simple 2D → 1D Example

Data varies mostly along one diagonal direction → project everything onto that direction → 1D representation captures most of the information.

```
PC1 = direction of maximum variance
PC2 = perpendicular to PC1 (minimum remaining variance)

Dropping PC2 loses minimal information when variance along PC2 is small.
```

---

## 3.4 The PCA Algorithm

### Step 1 — Preprocessing (Always Required)

1. **Mean normalize:** $x_j := x_j - \mu_j$
2. **Feature scale:** $x_j := \dfrac{x_j}{\sigma_j}$ (if features have very different scales)

### Step 2 — Find Principal Components

PCA finds unit vectors $\vec{u}^{(1)}, \vec{u}^{(2)}, \ldots$ such that:
- $\vec{u}^{(1)}$ is the direction of **maximum variance**
- $\vec{u}^{(2)}$ is the direction of **maximum remaining variance** perpendicular to $\vec{u}^{(1)}$
- and so on…

These are found by computing eigenvectors of the **covariance matrix**:

$$\Sigma = \frac{1}{m} X^T X$$

where $X$ is the $m \times n$ data matrix (rows = examples, columns = features).

### Step 3 — Project Data

Project each example onto the first $K$ principal components:

$$z_k^{(i)} = \vec{x}^{(i)} \cdot \vec{u}^{(k)}, \quad k = 1, 2, \ldots, K$$

The new representation:

$$\vec{z}^{(i)} = \begin{bmatrix} z_1^{(i)} \\ z_2^{(i)} \\ \vdots \\ z_K^{(i)} \end{bmatrix} \in \mathbb{R}^K \quad (K \ll n)$$

### Step 4 — Reconstruct (Approximate)

$$\hat{\vec{x}}^{(i)} = z_1^{(i)} \vec{u}^{(1)} + z_2^{(i)} \vec{u}^{(2)} + \cdots + z_K^{(i)} \vec{u}^{(K)}$$

This is an **approximation** — information from lower-variance components is lost.

---

## 3.5 Numerical Example: 2D to 1D PCA

**Data** (4 examples, 2 features, already mean-normalized):

| Example | $x_1$ | $x_2$ |
|---------|-------|-------|
| E1      | 2     | 1     |
| E2      | 3     | 2     |
| E3      | −2    | −1    |
| E4      | −3    | −2    |

**Step 1 — Covariance Matrix:**

$$\Sigma = \frac{1}{4}\begin{bmatrix}2 & 3 & -2 & -3 \\ 1 & 2 & -1 & -2\end{bmatrix}\begin{bmatrix}2 & 1 \\ 3 & 2 \\ -2 & -1 \\ -3 & -2\end{bmatrix} = \begin{bmatrix}6.5 & 4 \\ 4 & 2.5\end{bmatrix}$$

**Step 2 — First Eigenvector (Principal Component):**

$$\vec{u}^{(1)} = \begin{bmatrix}0.832 \\ 0.555\end{bmatrix}, \quad \lambda_1 = 8.96$$

**Step 3 — Project All Points:**

$$z^{(1)} = 2(0.832) + 1(0.555) = 2.219$$

$$z^{(2)} = 3(0.832) + 2(0.555) = 3.606$$

$$z^{(3)} = -2(0.832) + (-1)(0.555) = -2.219$$

$$z^{(4)} = -3(0.832) + (-2)(0.555) = -3.606$$

1D projection: $[2.219,\ 3.606,\ -2.219,\ -3.606]$

**Step 4 — Reconstruct E1:**

$$\hat{\vec{x}}^{(1)} = 2.219 \times \begin{bmatrix}0.832 \\ 0.555\end{bmatrix} = \begin{bmatrix}1.846 \\ 1.232\end{bmatrix}$$

Original: $(2,\ 1)$. Reconstructed: $(1.846,\ 1.232)$. Very close — minimal information lost.

---

## 3.6 Choosing How Many Principal Components ($K$)

### Explained Variance Ratio

$$\text{Explained Variance Ratio of PC}_k = \frac{\lambda_k}{\displaystyle\sum_{j=1}^{n} \lambda_j}$$

### Andrew Ng's Rule of Thumb

> *Choose the smallest $K$ such that at least 95% of the variance is retained.*

$$\frac{\displaystyle\sum_{k=1}^{K} \lambda_k}{\displaystyle\sum_{k=1}^{n} \lambda_k} \geq 0.95$$

**For our example:** $\lambda_1 = 8.96$, $\lambda_2 = 0.04$, total = 9.00.

$$\text{Variance retained by PC1} = \frac{8.96}{9.00} = 99.6\%$$

One principal component retains 99.6% of variance — the data was essentially 1D.

### Scree Plot

Plot cumulative explained variance against $K$ and pick the "elbow" or where you cross the 95% threshold.

---

## 3.7 Correct Use of PCA

### Good Uses

| Use Case | Description |
|----------|-------------|
| **Visualization** | $n=50$ features → PCA → 2D/3D plots; discover clusters and outliers |
| **Speed up ML** | $n=1000$ features → 100 features → faster training |
| **Storage compression** | Store PCA coefficients instead of raw high-dimensional data |

### Bad Uses

| Misuse | Why It's Wrong |
|--------|----------------|
| **Fixing overfitting** | Use regularization instead. PCA discards variance indiscriminately — may remove useful signal. |
| **Applying by default** | First try without PCA. Add it only if training is too slow or memory is an issue. |

### PCA vs Feature Selection

| | PCA | Feature Selection |
|---|---|---|
| Output | New abstract features (linear combinations) | Subset of original features |
| Interpretability | Hard — new features have no direct meaning | Easy — original features kept |
| Handles correlation | Yes, explicitly | No |
| Information preserved | Quantified exactly | Not guaranteed |

---

# PART 4 — Key Takeaways

### Collaborative Filtering
- Learns $\vec{w}^{(j)},\ b^{(j)}$ and $\vec{x}^{(i)}$ **jointly** from ratings alone
- Combined cost function minimizes squared prediction error across all users and movies simultaneously
- **Mean normalization** is critical — enables sensible recommendations for new users
- Related items: find smallest $\left\|\vec{x}^{(i)} - \vec{x}^{(k)}\right\|^2$

### Content-Based Filtering
- Uses explicit $\vec{x}_u^{(j)}$ (user features) and $\vec{x}_m^{(i)}$ (movie features)
- Two neural networks learn embeddings $\vec{v}_u^{(j)}$ and $\vec{v}_m^{(i)}$ of the **same dimension**
- Prediction: $\hat{y}^{(i,j)} = \vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}$
- Large-scale systems use **Retrieval → Ranking** pipeline

### PCA
- Projects $n$-dimensional data onto $K$ principal components ($K \ll n$)
- Each PC = direction of maximum remaining variance
- Choose $K$ to retain $\geq 95\%$ of explained variance
- Use for **visualization and compression** — not for fixing overfitting

---