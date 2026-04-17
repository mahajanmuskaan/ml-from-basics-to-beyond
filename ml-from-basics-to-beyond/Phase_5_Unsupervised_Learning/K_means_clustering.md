# Part 1 — K-Means Clustering

## 1.1 — Core Intuition

Imagine you run an online clothing store. You have data on 1000 customers — their age and annual spending. You want to group similar customers together so you can send targeted marketing to each group.

Nobody has told you how many groups exist or which customers belong together. You just have the raw data:

```
Spending
  |
  |        * * *
  |      * * * * *
  |    * *
  |
  |                * * *
  |              * * * * *
  |
  |    * *
  |  * * * *
  |    * *
  +----------------------→ Age

Three natural clusters visible — but the algorithm must discover them
```

K-Means finds these natural groupings automatically by answering one question repeatedly:

> *"Which points belong together, and what is the center of each group?"*

---

## 1.2 — The Story: Grouping Customers

You have 10 customers with two features — Age and Spending Score:

| Customer | Age $x_1$ | Spending $x_2$ |
|----------|-----------|----------------|
| C1       | 2         | 8              |
| C2       | 3         | 9              |
| C3       | 4         | 8              |
| C4       | 8         | 2              |
| C5       | 9         | 3              |
| C6       | 9         | 2              |
| C7       | 15        | 8              |
| C8       | 16        | 9              |
| C9       | 15        | 7              |
| C10      | 5         | 3              |

**Goal:** Find $K=3$ natural groups.

```
Spending
  |
9 |  ○C2           ★C8
8 |○C1  ○C3       ★C7  ★C9
7 |
6 |
5 |
4 |
3 |        □C10    □C5
2 |           □C4   □C6
  +--------------------→ Age
  2  3  4  5  6  7  8  9  10  15  16
```

Three visible groups:
- **○** = Young high spenders (left top)
- **□** = Middle low spenders (middle/right bottom)
- **★** = Old high spenders (right top)

---

## 1.3 — The K-Means Algorithm: Step by Step

### Step 0 — Initialize: Place K Centroids Randomly

A **centroid** is the center point of a cluster. Start by randomly placing $K=3$ centroids anywhere in the feature space.

$$\mu_1 = (3, 7), \quad \mu_2 = (8, 5), \quad \mu_3 = (14, 6)$$

These are just random starting positions — they have no special meaning yet.

```
Spending
  |
9 |  ○C2           ★C8
8 |○C1  ○C3       ★C7  ★C9
7 |  ×μ1
6 |                   ×μ3
5 |        ×μ2
4 |
3 |        □C10   □C5
2 |           □C4  □C6
  +--------------------→ Age

× = centroid positions (random start)
```

---

### Step 1 — Assignment: Assign Each Point to Its Nearest Centroid

For every customer, compute the distance to each centroid. Assign the customer to the closest one.

Using Euclidean distance:

$$d = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}$$

**Customer C1 = (2, 8):**

$$d(C1, \mu_1) = \sqrt{(2-3)^2 + (8-7)^2} = \sqrt{1+1} = 1.41$$
$$d(C1, \mu_2) = \sqrt{(2-8)^2 + (8-5)^2} = \sqrt{36+9} = 6.71$$
$$d(C1, \mu_3) = \sqrt{(2-14)^2 + (8-6)^2} = \sqrt{144+4} = 12.17$$

Nearest: $\mu_1$ → **Assign C1 to Cluster 1**

**Customer C4 = (8, 2):**

$$d(C4, \mu_1) = \sqrt{(8-3)^2 + (2-7)^2} = \sqrt{25+25} = 7.07$$
$$d(C4, \mu_2) = \sqrt{(8-8)^2 + (2-5)^2} = \sqrt{0+9} = 3.00$$
$$d(C4, \mu_3) = \sqrt{(8-14)^2 + (2-6)^2} = \sqrt{36+16} = 7.21$$

Nearest: $\mu_2$ → **Assign C4 to Cluster 2**

**Customer C7 = (15, 8):**

$$d(C7, \mu_1) = \sqrt{(15-3)^2 + (8-7)^2} = \sqrt{144+1} = 12.04$$
$$d(C7, \mu_2) = \sqrt{(15-8)^2 + (8-5)^2} = \sqrt{49+9} = 7.62$$
$$d(C7, \mu_3) = \sqrt{(15-14)^2 + (8-6)^2} = \sqrt{1+4} = 2.24$$

Nearest: $\mu_3$ → **Assign C7 to Cluster 3**

Doing this for all 10 customers:

| Customer | Position | Nearest Centroid       | Cluster |
|----------|----------|------------------------|---------|
| C1       | (2, 8)   | $\mu_1$ (1.41)         | 1       |
| C2       | (3, 9)   | $\mu_1$ (2.24)         | 1       |
| C3       | (4, 8)   | $\mu_1$ (2.24)         | 1       |
| C4       | (8, 2)   | $\mu_2$ (3.00)         | 2       |
| C5       | (9, 3)   | $\mu_2$ (2.83)         | 2       |
| C6       | (9, 2)   | $\mu_2$ (3.16)         | 2       |
| C7       | (15, 8)  | $\mu_3$ (2.24)         | 3       |
| C8       | (16, 9)  | $\mu_3$ (3.16)         | 3       |
| C9       | (15, 7)  | $\mu_3$ (1.41)         | 3       |
| C10      | (5, 3)   | $\mu_2$ (2.83)         | 2       |

---

### Step 2 — Update: Move Each Centroid to the Mean of Its Assigned Points

Now recompute each centroid as the average position of all points assigned to it:

**Cluster 1** (C1, C2, C3):

$$\mu_1^{\text{new}} = \left(\frac{2+3+4}{3}, \frac{8+9+8}{3}\right) = (3,\ 8.33)$$

**Cluster 2** (C4, C5, C6, C10):

$$\mu_2^{\text{new}} = \left(\frac{8+9+9+5}{4}, \frac{2+3+2+3}{4}\right) = (7.75,\ 2.5)$$

**Cluster 3** (C7, C8, C9):

$$\mu_3^{\text{new}} = \left(\frac{15+16+15}{3}, \frac{8+9+7}{3}\right) = (15.33,\ 8)$$

The centroids have moved from their random starting positions to better represent their clusters.

```
Spending
  |
9 |  ○C2                    ★C8
8 |○C1  ○C3  ×μ1(new)  ★C7 ×μ3(new) ★C9
7 |
6 |
5 |
4 |
3 |      ×μ2(new) □C10 □C5
2 |              □C4  □C6
  +--------------------→ Age

Centroids moved closer to their actual cluster centers!
```

---

### Step 3 — Repeat Until Convergence

Go back to Step 1 — reassign all points to their nearest (updated) centroid. Then update centroids again. Repeat.

In our clean example, the clusters are well-separated so assignments don't change after one update.

**Convergence:** When no customer changes cluster AND centroids stop moving — the algorithm has converged.

**Final Result:**

- **Cluster 1** (Young High Spenders): C1, C2, C3 → Center: (3, 8.33)
- **Cluster 2** (Middle Low Spenders): C4, C5, C6, C10 → Center: (7.75, 2.5)
- **Cluster 3** (Old High Spenders): C7, C8, C9 → Center: (15.33, 8)

---

## 1.4 — The Cost Function (Distortion Function)

K-Means optimizes a cost function called the **Distortion Function** — the average squared distance from each point to its assigned centroid:

$$\boxed{J = \frac{1}{m}\sum_{i=1}^{m}||\vec{x}^{(i)} - \mu_{c^{(i)}}||^2}$$

Where:
- $\vec{x}^{(i)}$ = the $i$-th data point
- $\mu_{c^{(i)}}$ = centroid of the cluster assigned to point $i$
- $||\cdot||^2$ = squared distance

**What this measures:** How tightly packed the clusters are. Smaller $J$ means points are closer to their cluster centers = tighter, better clusters.

### K-Means Minimizes J in Two Steps

| Step            | What It Does to J                                                                |
|-----------------|----------------------------------------------------------------------------------|
| Assignment step | Minimizes $J$ over cluster assignments (fix centroids, optimize assignments)     |
| Update step     | Minimizes $J$ over centroid positions (fix assignments, optimize centroids)      |

Both steps reduce or maintain $J$ — it never increases. This **guarantees K-Means always converges**.

---

## 1.5 — The Random Initialization Problem

K-Means starts with random centroids. Different starting positions can lead to different final clusters — sometimes bad ones.

```
Bad initialization:          Good initialization:
  ×μ1  ×μ2  ×μ3              ×μ1     ×μ2    ×μ3

All 3 centroids start        Centroids spread across
in the same cluster          different regions
        ↓                            ↓
Poor final clustering        Good final clustering

   ○                           ○     ★     □
   ○○  ★□  □★                  ○○    ★★    □□
   ○   ★   □                         (correct)
   (wrong — merged)
```

This is called getting stuck in a **local minimum** — not the globally best clustering.

### Andrew Ng's Solution: Multiple Random Initializations

Run K-Means 50–100 times with different random starting positions. Keep the run that gives the **lowest final cost $J$**.

```
Run 1:   Random init → converges → J = 45.2
Run 2:   Random init → converges → J = 38.7  ← Best!
Run 3:   Random init → converges → J = 52.1
...
Run 100: Random init → converges → J = 41.3

Final answer: Use the clustering from Run 2 (J = 38.7)
```

This simple strategy reliably finds good clusterings for most practical problems.

### K-Means++ Initialization (Better Alternative)

Instead of purely random initialization, K-Means++ spreads out the initial centroids deliberately:

1. **Step 1:** Pick first centroid randomly from data points
2. **Step 2:** For each remaining centroid:
   - Compute distance from each point to nearest existing centroid
   - Choose next centroid with probability proportional to distance²
   - (Points far from existing centroids are more likely chosen)
3. **Step 3:** Repeat until $K$ centroids are placed

This dramatically reduces the chance of bad initializations because centroids start spread across the data.

---

## 1.6 — Choosing K: The Elbow Method

How do you know the right number of clusters? Plot $J$ (distortion) against different values of $K$:

```
J (Distortion)
  |
  |*  ← K=1: all points in one cluster, high distortion
  | *
  |  *   ← Elbow at K=3
  |   * * * * * * ─── ← diminishing returns after K=3
  +-------------------→ K
  1  2  3  4  5  6  7
```

The "elbow" — where adding more clusters stops meaningfully reducing distortion — suggests the optimal $K$.

> **Important caveat (Andrew Ng):** *"The elbow method doesn't always give a clear answer. Sometimes the curve is smooth with no obvious elbow. In those cases, choose K based on the downstream purpose."*

### Choosing K Based on Purpose

Sometimes the right $K$ comes from what you'll do with the clusters.

**T-shirt sizing example:** You're grouping customers by height and weight to decide how many T-shirt sizes to make.

| K    | Sizes            | Trade-off                              |
|------|------------------|----------------------------------------|
| K=3  | S, M, L          | Cheaper to manufacture, some poor fit  |
| K=5  | XS, S, M, L, XL  | More expensive, most customers fit well |
| K=10 | 10 sizes         | Very expensive, near-perfect fit       |

The distortion curve can't tell you which tradeoff is right — **you decide based on business context**.

---

## 1.7 — What K-Means Is Good For

| Application            | What Gets Clustered        | Clusters Represent               |
|------------------------|----------------------------|----------------------------------|
| Customer segmentation  | Customers by behavior      | Market segments for targeting    |
| Image compression      | Pixel colors               | Color palette (reduce colors)    |
| Document grouping      | Documents by words         | Topics                           |
| Genomics               | Genes by expression        | Gene function groups             |