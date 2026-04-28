# Course 3 Week 3 — Reinforcement Learning
*Reference: Andrew Ng's ML Specialization, Course 3 — Unsupervised Learning, Recommenders, Reinforcement Learning*

---

## The Big Picture — What is Reinforcement Learning?

Reinforcement Learning (RL) is a fundamentally different paradigm from supervised and unsupervised learning.

| Paradigm | Input | Output | Signal |
|---|---|---|---|
| **Supervised** | Labeled $(x, y)$ pairs | Prediction $\hat{y}$ | Explicit correct answer |
| **Unsupervised** | Unlabeled $x$ | Structure/clusters | None |
| **Reinforcement** | State $s$, reward $r$ | Action $a$ | Scalar reward signal |

> *"In reinforcement learning, you don't tell the agent what the right answer is. You just reward it for good outcomes and let it figure out how to get there."*

**Core idea:** An **agent** interacts with an **environment**, observes the current **state**, takes an **action**, receives a **reward**, and transitions to a new state. The goal is to learn a **policy** — a mapping from states to actions — that maximises cumulative reward over time.

**Famous examples:**
- Playing chess or Go (state = board position, action = move, reward = win/lose)
- Controlling a robot (state = sensor readings, action = joint torques, reward = distance walked)
- Helicopter control (state = orientation + velocity, action = rotor inputs, reward = flight stability)
- The Mars Rover (state = position + power level, action = movement direction, reward = science value collected)

---

# PART 1 — The Reinforcement Learning Framework

## 1.1 The Mars Rover Example

Andrew Ng introduces RL through a simple 1D Mars Rover problem. The rover can be in one of **6 positions** (states):

```
State:  1    2    3    4    5    6
       [L1] [  ] [  ] [  ] [  ] [R6]

Reward:  100   0    0    0    0   40
```

- **State 1** has a reward of 100 (interesting rock formation on the left)
- **State 6** has a reward of 40 (less interesting site on the right)
- All other states have reward 0

At each step, the rover can take one of two actions: **Go Left** or **Go Right**.

The **terminal states** are State 1 (left end) and State 6 (right end). Once the rover reaches a terminal state, the episode ends.

**The question:** What sequence of actions should the rover take from each position?

---

## 1.2 Formalising RL: The Markov Decision Process (MDP)

Reinforcement learning problems are formalised as a **Markov Decision Process (MDP)**, defined by five components:

| Component | Symbol | Description |
|---|---|---|
| **States** | $\mathcal{S}$ | Set of all possible situations the agent can be in |
| **Actions** | $\mathcal{A}$ | Set of all possible actions the agent can take |
| **Transition probabilities** | $P(s' \mid s, a)$ | Probability of moving to state $s'$ after taking action $a$ in state $s$ |
| **Reward function** | $R(s)$ or $R(s, a)$ | Scalar reward received at each step |
| **Discount factor** | $\gamma \in [0, 1)$ | How much the agent values future rewards relative to immediate rewards |

The **Markov Property** means the future state depends only on the current state and action — not on the history of how you got there:

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

---

## 1.3 The Return: Discounted Cumulative Reward

The agent doesn't just care about the next reward — it cares about the **total reward accumulated over time**. But rewards far in the future are worth less than immediate rewards.

### Definition of Return

The **return** $G_t$ starting from time step $t$ is the discounted sum of all future rewards:

$$\boxed{G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}}$$

The discount factor $\gamma \in [0, 1)$ controls how myopic or far-sighted the agent is:
- $\gamma = 0$: the agent is fully **myopic** — only cares about immediate reward
- $\gamma \to 1$: the agent is fully **far-sighted** — treats all future rewards almost equally
- Typical values: $\gamma = 0.9$, $\gamma = 0.99$, $\gamma = 0.995$

### Why Discount?

Three reasons:

1. **Mathematical convenience:** ensures infinite sums converge for infinite-horizon problems
2. **Economic intuition:** a dollar today is worth more than a dollar tomorrow (time value of money)
3. **Uncertainty:** the further in the future, the less certain we are about receiving the reward

### Numerical Example

Starting at State 4, taking the sequence: Right → Right → Right (reaching terminal state 6).

Rewards collected: $R = 0$ (State 5), $R = 40$ (State 6), with $\gamma = 0.5$:

$$G_4 = 0 + 0.5 \times 0 + 0.5^2 \times 40 = 0 + 0 + 10 = 10$$

Starting at State 4, taking: Left → Left → Left → Left (reaching terminal state 1):

$$G_4 = 0 + 0.5 \times 0 + 0.5^2 \times 0 + 0.5^3 \times 100 = 12.5$$

Going left is better from State 4, even though State 6 is closer!

---

## 1.4 The Policy

A **policy** $\pi$ maps states to actions (or distributions over actions):

$$\pi: \mathcal{S} \to \mathcal{A}$$

$$\pi(s) = a \quad \text{(deterministic policy)}$$

$$\pi(a \mid s) = P(\text{take action } a \text{ in state } s) \quad \text{(stochastic policy)}$$

The goal of reinforcement learning is to find the **optimal policy** $\pi^*$ that maximises the expected return from every state.

For the Mars Rover with $\gamma = 0.5$:

| State | Optimal Action | Reasoning |
|-------|---------------|-----------|
| 1 | — | Terminal state |
| 2 | Left | Closer to the 100 reward |
| 3 | Left | Return from left > return from right |
| 4 | Left | As shown above: 12.5 > 10 |
| 5 | Right | Return from right > return from left |
| 6 | — | Terminal state |

---

# PART 2 — Value Functions

## 2.1 The State-Value Function $V^\pi(s)$

The **state-value function** $V^\pi(s)$ is the **expected return** starting from state $s$ and following policy $\pi$ thereafter:

$$\boxed{V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\ \Bigg|\ s_t = s\right]}$$

This answers: "How good is it to be in state $s$ when following policy $\pi$?"

The **optimal state-value function** $V^*(s)$ is the maximum over all policies:

$$V^*(s) = \max_\pi V^\pi(s)$$

---

## 2.2 The Action-Value Function (Q-Function) $Q^\pi(s, a)$

The **action-value function** (also called the **Q-function**) is the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$\boxed{Q^\pi(s, a) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s,\ a_t = a\right]}$$

This answers: "How good is it to take action $a$ in state $s$ when following policy $\pi$ afterward?"

The **optimal Q-function** $Q^*(s, a)$ is what we ultimately want to learn in Deep Q-Learning:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

**Relationship to $V^*$:**

$$V^*(s) = \max_a Q^*(s, a)$$

**Extracting the optimal policy from $Q^*$:**

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

If we know $Q^*(s, a)$ for all states and actions, we can immediately read off the optimal action in any state.

---

## 2.3 The Bellman Equation — The Central Equation of RL

The Bellman equation expresses a **recursive relationship** between the value of a state and the values of its successors.

### Bellman Equation for $V^*$

$$\boxed{V^*(s) = \max_a \left[R(s) + \gamma \sum_{s'} P(s' \mid s, a)\ V^*(s')\right]}$$

**Intuition:** The value of the current state equals the immediate reward plus the discounted value of the best next state.

In the deterministic case (where $P(s' \mid s, a) = 1$ for exactly one $s'$):

$$V^*(s) = R(s) + \gamma \max_a V^*(s')$$

### Bellman Equation for $Q^*$

$$\boxed{Q^*(s, a) = R(s) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')}$$

**Intuition:** The Q-value of $(s, a)$ equals the immediate reward plus the discounted maximum Q-value of the next state (under the best action).

### Bellman Equation for the Mars Rover

Using $\gamma = 0.5$ and deterministic transitions:

For State 2 (optimal action = Left, which leads to State 1):

$$Q^*(2, \text{Left}) = R(2) + \gamma \cdot V^*(1) = 0 + 0.5 \times 100 = 50$$

$$Q^*(2, \text{Right}) = R(2) + \gamma \cdot V^*(3) = 0 + 0.5 \times 25 = 12.5$$

$$V^*(2) = \max(50, 12.5) = 50 \quad \Rightarrow \quad \pi^*(2) = \text{Left}$$

---

## 2.4 Worked Example: Solving the Mars Rover MDP

**Setup:** 6 states, terminal states have $V^*(1) = 100$, $V^*(6) = 40$, $\gamma = 0.5$.

Using the Bellman equation $V^*(s) = R(s) + \gamma \max_a V^*(s')$:

$$V^*(5) = 0 + 0.5 \times \max(V^*(4),\ V^*(6)) = 0.5 \times \max(V^*(4),\ 40)$$

$$V^*(4) = 0 + 0.5 \times \max(V^*(3),\ V^*(5))$$

$$V^*(3) = 0 + 0.5 \times \max(V^*(2),\ V^*(4))$$

$$V^*(2) = 0 + 0.5 \times \max(V^*(1),\ V^*(3)) = 0.5 \times \max(100,\ V^*(3))$$

Solving by working outward from the terminal states:

| State | $V^*(s)$ | Optimal Action |
|-------|----------|----------------|
| 1 | 100 | Terminal |
| 2 | 50 | Left |
| 3 | 25 | Left |
| 4 | 12.5 | Left |
| 5 | 20 | Right |
| 6 | 40 | Terminal |

At State 5: $V^*(5) = 0.5 \times 40 = 20$ (go right). At State 4: $V^*(4) = 0.5 \times \max(25, 20) = 12.5$ (go left).

---

# PART 3 — Stochastic Environments

## 3.1 Randomness in RL

In real environments, actions don't always produce the intended outcome. The **Lunar Lander** might fire its thruster left but a gust of wind pushes it right. The Mars Rover might try to go left but slip and go right.

### Stochastic Transition Example

Suppose the rover's actions are noisy:
- "Go Left" → actually goes left with probability 0.9, right with probability 0.1
- "Go Right" → actually goes right with probability 0.9, left with probability 0.1

Now the Bellman equation becomes:

$$V^*(s) = R(s) + \gamma \max_a \sum_{s'} P(s' \mid s, a)\ V^*(s')$$

The return $G_t$ is now a **random variable**, and we optimise its **expected value**:

$$\pi^* = \arg\max_\pi\ \mathbb{E}\!\left[G_t\right]$$

---

## 3.2 Stochastic Rewards

The reward itself can also be random:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

Where $R_{t+k}$ is a random variable. The objective becomes:

$$\pi^* = \arg\max_\pi\ \mathbb{E}\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\right]$$

All RL algorithms implicitly target this expectation through sampling and averaging.

---

# PART 4 — Deep Q-Learning (DQN)

## 4.1 Why Deep Learning for RL?

In the Mars Rover, we had only 6 states. In real problems:

- **Chess:** $10^{43}$ states
- **Go:** $10^{170}$ states
- **Atari games:** raw pixel inputs ($84 \times 84 \times 4 = 28{,}224$ dimensional states)
- **Lunar Lander:** continuous state space (8 real-valued numbers)

We cannot build a lookup table for $Q^*(s, a)$. We need to **approximate** it with a function.

$$Q^*(s, a) \approx Q(s, a;\ \boldsymbol{\theta})$$

where $\boldsymbol{\theta}$ are the parameters of a **neural network**.

---

## 4.2 The Lunar Lander Environment

Andrew Ng uses the **Lunar Lander** as the running example for Deep Q-Learning.

**State space (8-dimensional):**
- $(x, y)$ — horizontal and vertical position
- $(\dot{x}, \dot{y})$ — horizontal and vertical velocity
- $(\theta, \dot{\theta})$ — angle and angular velocity
- $(l_L, l_R)$ — whether left/right legs are touching the ground (binary)

**Action space (4 discrete actions):**

| Action | Description |
|--------|-------------|
| 0 | Do nothing |
| 1 | Fire left thruster |
| 2 | Fire main engine (down) |
| 3 | Fire right thruster |

**Reward function:**
- Moving toward landing pad: positive reward
- Moving away: negative reward
- Crashing: −100
- Safe landing: +100 to +140
- Each leg on ground: +10
- Firing main engine: −0.3 per frame (encourages fuel efficiency)
- Firing side thrusters: −0.03 per frame

**Episode:** Starts with lander at top of screen, ends when lander crashes, lands, or flies out of bounds.

---

## 4.3 The DQN Architecture

For the Lunar Lander, the neural network takes a **state** as input and outputs a **Q-value for each action**:

```
Input: state s (8-dimensional vector)
       [x, y, ẋ, ẏ, θ, θ̇, l_L, l_R]
             ↓
    Dense layer: 64 units, ReLU
             ↓
    Dense layer: 64 units, ReLU
             ↓
Output: Q-values for all 4 actions
    [Q(s, 0),  Q(s, 1),  Q(s, 2),  Q(s, 3)]
    [Nothing, Left thr., Main eng., Right thr.]
```

To select an action in state $s$:

$$a^* = \arg\max_a Q(s, a;\ \boldsymbol{\theta})$$

**Why output all Q-values at once?** With one forward pass, we get Q-values for all actions. If we had a separate network per action, we'd need 4 forward passes per decision step.

---

## 4.4 Training the DQN: The Key Idea

The DQN is trained using the **Bellman equation as a training target**.

Given a transition $(s, a, r, s')$:

- **Current estimate:** $Q(s, a;\ \boldsymbol{\theta})$
- **Bellman target (label):** $y = r + \gamma \max_{a'} Q(s', a';\ \boldsymbol{\theta})$

The training objective is to minimise the **TD (Temporal Difference) error**:

$$\mathcal{L}(\boldsymbol{\theta}) = \left(Q(s, a;\ \boldsymbol{\theta}) - \underbrace{\left(r + \gamma \max_{a'} Q(s', a';\ \boldsymbol{\theta})\right)}_{\text{TD target}}\right)^2$$

This is exactly like supervised learning, except the **labels are themselves derived from the current network**.

---

## 4.5 The Full DQN Algorithm

```
Initialise:
    Q-network with random weights θ
    Replay buffer D (empty)
    
For each episode:
    Observe initial state s
    
    For each step t:
        1. SELECT action a using ε-greedy policy:
               With prob ε: take random action (explore)
               With prob (1-ε): take a* = argmax_a Q(s, a; θ) (exploit)
        
        2. EXECUTE action a in environment
           Observe reward r and next state s'
        
        3. STORE (s, a, r, s', done) in replay buffer D
        
        4. SAMPLE random minibatch of transitions from D
        
        5. COMPUTE targets:
               y = r                          (if terminal)
               y = r + γ * max_a' Q(s', a'; θ)  (if non-terminal)
        
        6. TRAIN: update θ by gradient descent on:
               L(θ) = (Q(s, a; θ) - y)²
        
        7. UPDATE s ← s'
        
        8. DECAY ε (gradually reduce exploration)
```

---

## 4.6 Critical Technique 1: Experience Replay

**The Problem Without Replay:**

If we train on transitions in the order they were collected:
- Consecutive transitions are **highly correlated** (State 5 is always followed by State 4 or 6)
- Neural networks assume i.i.d. (independent, identically distributed) training data
- Highly correlated updates cause **unstable learning** and **catastrophic forgetting**

**The Solution: Replay Buffer**

Store all recent transitions $(s, a, r, s')$ in a **circular buffer** of fixed size (e.g., 100,000 transitions).

At each training step, sample a **random minibatch** (e.g., 64 transitions) from the buffer and train on that.

**Benefits:**
- Breaks temporal correlations between transitions
- Each transition can be reused many times (data efficiency)
- More stable gradient estimates

```
Replay buffer (FIFO, capacity N):
┌──────────────────────────────────────────────────────┐
│ (s1,a1,r1,s1')  (s2,a2,r2,s2')  ...  (sN,aN,rN,sN') │
└──────────────────────────────────────────────────────┘
                   ↑ Random sample 64 transitions for each update
```

---

## 4.7 Critical Technique 2: Target Network

**The Problem:**

The training update is:

$$\mathcal{L}(\boldsymbol{\theta}) = \left(Q(s, a;\ \boldsymbol{\theta}) - \left(r + \gamma \max_{a'} Q(s', a';\ \boldsymbol{\theta})\right)\right)^2$$

Both the **prediction** AND the **target** depend on $\boldsymbol{\theta}$. When we update $\boldsymbol{\theta}$:
- The prediction changes
- But the target ALSO changes
- This is like chasing a moving target — leads to **oscillations and divergence**

**The Solution: Target Network**

Maintain two networks:
- **Online network** $Q(s, a;\ \boldsymbol{\theta})$ — trained every step
- **Target network** $\hat{Q}(s, a;\ \boldsymbol{\theta}^-)$ — frozen copy, updated only every $C$ steps

The training objective becomes:

$$\mathcal{L}(\boldsymbol{\theta}) = \left(Q(s, a;\ \boldsymbol{\theta}) - \underbrace{\left(r + \gamma \max_{a'} \hat{Q}(s', a';\ \boldsymbol{\theta}^-)\right)}_{\text{fixed target}}\right)^2$$

Every $C$ steps (e.g., $C = 100$ or $C = 10{,}000$):

$$\boldsymbol{\theta}^- \leftarrow \boldsymbol{\theta}$$

**Why it works:** The target is now stable for $C$ steps, making the optimisation problem much more like standard supervised learning.

---

## 4.8 Critical Technique 3: ε-Greedy Exploration

**The Exploration-Exploitation Dilemma:**

- **Exploit:** take the action the network currently thinks is best ($\arg\max_a Q(s, a)$)
- **Explore:** try random actions to discover potentially better strategies

If we only exploit, we may get stuck in a **local optimum** (the network thinks firing the right thruster is always best because it hasn't seen what happens if it fires the main engine).

**ε-Greedy Policy:**

$$a = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(s, a;\ \boldsymbol{\theta}) & \text{with probability } 1 - \varepsilon \end{cases}$$

**ε-Decay Schedule:**

Start with high exploration ($\varepsilon = 1.0$), gradually decay:

$$\varepsilon_{t+1} = \max(\varepsilon_{\min},\ \varepsilon_t \times \varepsilon_{\text{decay}})$$

Typical values: $\varepsilon_{\text{init}} = 1.0$, $\varepsilon_{\min} = 0.01$, decay per episode = 0.995.

```
Early training:   ε = 1.0  →  100% random actions (pure exploration)
Mid training:     ε = 0.5  →  50% random, 50% greedy
Late training:    ε = 0.01 →  1% random, 99% greedy (mostly exploitation)
```

---

## 4.9 Critical Technique 4: Soft Update of Target Network

Instead of a hard copy every $C$ steps, a **soft update** (also called **Polyak averaging**) blends the online network weights into the target network at every step:

$$\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1 - \tau) \boldsymbol{\theta}^-$$

With a small $\tau$ (e.g., $\tau = 0.001$ or $\tau = 0.005$).

This gives a smoother, more stable target that gradually tracks the online network without abrupt jumps.

**Comparison:**

| Method | Update frequency | Stability |
|--------|-----------------|-----------|
| Hard copy | Every $C$ steps | Target jumps every $C$ steps |
| Soft update | Every step | Target drifts slowly — smoother |

Andrew Ng uses **soft updates** in the Lunar Lander implementation.

---

## 4.10 The Complete DQN Training Loop

Putting it all together, here is the complete algorithm with all four key techniques:

```
INITIALISE:
    Online network Q(s, a; θ) with random θ
    Target network Q̂(s, a; θ⁻) with θ⁻ ← θ
    Replay buffer D with capacity N = 100,000
    ε = 1.0,   ε_min = 0.01,   ε_decay = 0.995

FOR episode = 1, 2, ..., MAX_EPISODES:
    s = env.reset()
    total_reward = 0

    FOR step = 1, 2, ..., MAX_STEPS:

        ── ACTION SELECTION (ε-greedy) ─────────────────────────────
        If random() < ε:
            a = random_action()                  ← explore
        Else:
            a = argmax_a Q(s, a; θ)              ← exploit

        ── ENVIRONMENT STEP ────────────────────────────────────────
        s', r, done = env.step(a)
        Store (s, a, r, s', done) in D
        total_reward += r

        ── LEARNING UPDATE ─────────────────────────────────────────
        If len(D) ≥ BATCH_SIZE:
            Sample minibatch {(sᵢ, aᵢ, rᵢ, sᵢ', doneᵢ)} from D

            For each transition i:
                If doneᵢ:    yᵢ = rᵢ
                Else:        yᵢ = rᵢ + γ * max_a' Q̂(sᵢ', a'; θ⁻)

            Gradient step to minimise Σᵢ (Q(sᵢ, aᵢ; θ) - yᵢ)²

        ── SOFT UPDATE OF TARGET NETWORK ───────────────────────────
        θ⁻ ← τ * θ + (1 - τ) * θ⁻

        s ← s'
        If done: break

    ── EPSILON DECAY ───────────────────────────────────────────────
    ε ← max(ε_min, ε * ε_decay)
```

---

# PART 5 — Practical Implementation Details

## 5.1 Hyperparameter Reference

| Hyperparameter | Typical Value | Role |
|----------------|--------------|------|
| $\gamma$ (discount factor) | 0.995 | How far-sighted the agent is |
| $\varepsilon_{\text{init}}$ | 1.0 | Initial exploration rate |
| $\varepsilon_{\min}$ | 0.01 | Minimum exploration rate |
| $\varepsilon_{\text{decay}}$ | 0.995 | Per-episode decay multiplier |
| Replay buffer size $N$ | 100,000 | Number of transitions stored |
| Minibatch size | 64 | Transitions sampled per update |
| Target update rate $\tau$ | 0.001 – 0.01 | Soft update mixing coefficient |
| Learning rate $\alpha$ | 0.001 – 0.0001 | Neural network learning rate |
| Network architecture | 64 → 64 (ReLU) | Hidden layer sizes |

---

## 5.2 The Q-Function as a Neural Network (TensorFlow/Keras)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# Hyperparameters
GAMMA = 0.995
ALPHA = 1e-3
TAU   = 1e-3
BATCH = 64
BUFFER_SIZE = int(1e5)

# Q-Network
def build_q_network(state_size, num_actions):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')  # linear output for Q-values
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA),
                  loss='mse')
    return model

# State size = 8 (Lunar Lander), Actions = 4
q_network     = build_q_network(8, 4)
target_network = build_q_network(8, 4)
target_network.set_weights(q_network.get_weights())

# Replay Buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)

def store_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def sample_batch():
    return random.sample(replay_buffer, BATCH)

def compute_targets(batch):
    states      = np.array([t[0] for t in batch])
    actions     = np.array([t[1] for t in batch])
    rewards     = np.array([t[2] for t in batch])
    next_states = np.array([t[3] for t in batch])
    dones       = np.array([t[4] for t in batch])

    # Target network predicts Q(s', a') for all a'
    next_q  = target_network.predict(next_states, verbose=0)
    max_q   = np.max(next_q, axis=1)

    # y = r  (if done)
    # y = r + γ * max_a' Q(s', a'; θ⁻)  (if not done)
    targets = rewards + GAMMA * max_q * (1 - dones)

    # Build full Q-value targets — only update the action taken
    current_q = q_network.predict(states, verbose=0)
    current_q[np.arange(BATCH), actions] = targets
    return states, current_q

def soft_update():
    online_weights = q_network.get_weights()
    target_weights = target_network.get_weights()
    new_weights = [TAU * ow + (1 - TAU) * tw
                   for ow, tw in zip(online_weights, target_weights)]
    target_network.set_weights(new_weights)

def select_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)  # explore
    q_values = q_network.predict(state[np.newaxis, :], verbose=0)
    return np.argmax(q_values[0])   # exploit
```

---

## 5.3 Training Loop

```python
import gym

env = gym.make('LunarLander-v2')

epsilon     = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
total_points_history = []

for episode in range(3000):
    state = env.reset()
    total_reward = 0

    for t in range(1000):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        store_transition(state, action, reward, next_state, done)

        if len(replay_buffer) >= BATCH:
            batch = sample_batch()
            X, y = compute_targets(batch)
            q_network.fit(X, y, verbose=0, epochs=1)
            soft_update()

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    total_points_history.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg = np.mean(total_points_history[-100:])
        print(f'Episode {episode+1:4d} | Avg reward (last 100): {avg:6.1f} | ε: {epsilon:.3f}')
```

---

# PART 6 — Key Concepts and Theory

## 6.1 The Relationship Between Q, V, and π

These three objects fully define an RL solution:

$$\boxed{Q^*(s, a) \xrightarrow{\displaystyle\max_a} V^*(s) \xrightarrow{\displaystyle\arg\max_a Q^*} \pi^*(s)}$$

| Object | What it gives you |
|--------|------------------|
| $Q^*(s, a)$ | Value of being in state $s$ and taking action $a$ |
| $V^*(s)$ | Value of being in state $s$ (assuming optimal actions thereafter) |
| $\pi^*(s)$ | The best action to take in state $s$ |

---

## 6.2 Why the Bellman Equation Is Self-Consistent

The Bellman optimality equation has a unique solution $Q^*$ satisfying:

$$Q^*(s, a) = R(s) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

This is a **fixed-point equation** — $Q^*$ is the fixed point of the Bellman operator $\mathcal{T}$:

$$(\mathcal{T}Q)(s, a) = R(s) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$$

The Bellman operator is a **contraction mapping** (with contraction factor $\gamma$), so by the Banach fixed-point theorem, repeated application of $\mathcal{T}$ from any starting $Q$ converges to $Q^*$:

$$Q^* = \lim_{n \to \infty} \mathcal{T}^n Q_0$$

This is the theoretical basis for why Q-learning works.

---

## 6.3 Convergence Guarantees

**Tabular Q-learning** (with a lookup table) is **guaranteed to converge** to $Q^*$ given:
- Every $(s, a)$ pair is visited infinitely often
- The learning rate satisfies $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$

**Deep Q-learning** (with neural network approximation) has **no convergence guarantee** in general. The three instabilities that DQN addresses are:

| Instability | Solution |
|-------------|----------|
| Correlated samples | Experience Replay |
| Non-stationary target | Target Network |
| Diverging Q-values | Gradient clipping + careful architecture |

In practice, DQN works extremely well despite the lack of formal guarantees.

---

## 6.4 The Discount Factor $\gamma$ in Practice

The discount factor has a crucial effect on agent behaviour:

$$\gamma = 0.9 \implies \text{Horizon} \approx \frac{1}{1 - 0.9} = 10 \text{ steps}$$

$$\gamma = 0.99 \implies \text{Horizon} \approx \frac{1}{1 - 0.99} = 100 \text{ steps}$$

$$\gamma = 0.995 \implies \text{Horizon} \approx \frac{1}{1 - 0.995} = 200 \text{ steps}$$

The "effective horizon" $\frac{1}{1-\gamma}$ is the number of steps over which rewards are meaningfully accumulated. For the Lunar Lander (episodes of ~200-1000 steps), $\gamma = 0.995$ is appropriate.

**Effect of $\gamma$ on policy:**
- Low $\gamma$ (0.5): prefer immediate rewards, impatient agent
- High $\gamma$ (0.999): plan far into the future, patient agent
- Too high $\gamma$ (→ 1): instability in infinite-horizon problems

---

# PART 7 — Putting It All Together

## 7.1 The Full RL Framework Diagram

```
                 ┌────────────────────────────────┐
                 │           AGENT                │
                 │                                │
                 │  ┌──────────┐  ┌───────────┐  │
    State s ────►│  │ Q-Network│  │  Policy   │  │
                 │  │ Q(s,a;θ) │─►│ π(s)=     │  │──► Action a
                 │  │          │  │ argmax Q  │  │
    Reward r ────►│  └──────────┘  └───────────┘  │
                 │       │ Training                │
                 │  ┌──────────┐  ┌───────────┐  │
                 │  │  Target  │  │  Replay   │  │
                 │  │ Network  │  │  Buffer   │  │
                 │  │  Q̂(s,a)  │  │           │  │
                 │  └──────────┘  └───────────┘  │
                 └────────────────────────────────┘
                              │     ▲
                           Action   State, Reward
                              │     │
                 ┌────────────▼─────┴─────────────┐
                 │         ENVIRONMENT             │
                 │   (Lunar Lander, MDP, etc.)     │
                 └────────────────────────────────┘
```

---

## 7.2 Limitations of RL and DQN

### Sample Inefficiency

RL agents typically require **millions of environment interactions** to learn. Human players learn Atari games in minutes; DQN needs 50 million frames. This makes RL expensive for real-world applications where environment steps are costly (robotics, healthcare).

### Reward Design is Hard

The agent will **exactly optimise whatever reward you specify**. If the reward is mis-specified, you get unexpected behaviour:

- A boat racing game agent discovered it could score more points by spinning in circles collecting bonuses than finishing the race
- A robot trained to run quickly learned to make itself tall and fall forward
- A cleaning robot trained to minimise messes might hide messes instead of cleaning them

This is the **reward hacking** or **specification gaming** problem.

### Only Discrete Actions (Basic DQN)

The basic DQN framework outputs Q-values for each discrete action. For **continuous action spaces** (e.g., joint torques $\in [-1, 1]$), DQN does not directly apply. Extensions include:

- **DDPG** (Deep Deterministic Policy Gradient) — outputs continuous actions directly
- **TD3** (Twin Delayed DDPG) — more stable version of DDPG
- **SAC** (Soft Actor-Critic) — state-of-the-art continuous control

### Instability and Hyperparameter Sensitivity

DQN is notoriously sensitive to hyperparameter choices. Small changes to $\gamma$, $\tau$, or network architecture can cause training to diverge. Getting DQN to work reliably requires careful tuning.

---

## 7.3 Extensions to DQN

Several important improvements to the basic DQN have been developed:

| Extension | Key Idea | Benefit |
|-----------|---------|---------|
| **Double DQN** | Use online network to select action, target network to evaluate it | Reduces overestimation of Q-values |
| **Dueling DQN** | Decompose $Q(s,a) = V(s) + A(s,a)$ | Better value estimation, especially when many actions have similar values |
| **Prioritised Experience Replay** | Sample transitions with high TD error more often | More efficient use of replay buffer |
| **Rainbow** | Combines all of the above + distributional RL + noisy networks | State-of-the-art Atari performance |
| **Policy Gradient (REINFORCE)** | Directly optimise the policy $\pi(a \mid s; \theta)$ | Handles continuous action spaces natively |
| **PPO** | Proximal Policy Optimisation | Most stable and widely used policy gradient method today |
| **Actor-Critic** | Combines value function and policy gradient | Lower variance than pure policy gradients |

---

# PART 8 — Key Takeaways

### Core Concepts

The RL framework consists of an **agent** that observes a **state** $s$, takes an **action** $a$, receives a **reward** $r$, and transitions to a new state $s'$. The goal is to learn a **policy** $\pi^*$ that maximises the expected **discounted return** $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.

### Bellman Equations

The two central equations of RL:

$$Q^*(s, a) = R(s) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

$$V^*(s) = \max_a \left[R(s) + \gamma \sum_{s'} P(s' \mid s, a)\ V^*(s')\right]$$

The optimal policy follows immediately: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

### Deep Q-Learning (DQN)

When state spaces are too large for lookup tables, approximate $Q^*(s, a)$ with a neural network $Q(s, a;\ \boldsymbol{\theta})$. Train it using Bellman targets. Four critical techniques make DQN stable:

1. **Experience Replay** — store transitions in a buffer, sample randomly to break correlations
2. **Target Network** — use a frozen copy of the network for Bellman targets, update slowly
3. **ε-Greedy Exploration** — balance exploration (random actions) vs exploitation (greedy actions), decay ε over time
4. **Soft Updates** — $\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1-\tau) \boldsymbol{\theta}^-$ for smooth target tracking

### Practical Notes

- Choose $\gamma$ based on the effective horizon you need: $\frac{1}{1-\gamma}$
- Large replay buffers and small $\tau$ make training more stable
- Start with high $\varepsilon$ and decay slowly — insufficient exploration is the most common failure mode
- Monitor the **average episode reward over the last 100 episodes** as the primary training signal

---
