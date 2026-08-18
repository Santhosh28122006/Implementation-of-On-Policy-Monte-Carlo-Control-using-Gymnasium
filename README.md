# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate complete episode
    episode = generate_episode(epsilon)

    # Store total reward
    total_reward = sum([step[2] for step in episode])
    episode_rewards.append(total_reward)

    # Calculate returns
    G = 0

    # Keep track of state-action pairs already updated
    visited = set()

    # Process episode backwards
    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # First-visit Monte Carlo
        if (state, action) not in visited:

            visited.add((state, action))

            # Incremental update of Q-value
            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Print progress
    if (episode_num + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-1000:])
        print(
            f"Episode {episode_num + 1}/{num_episodes}, "
            f"Epsilon: {epsilon:.4f}, "
            f"Average Reward: {avg_reward:.3f}"
        )

```

---

## Output


Final Q-table:


<img width="735" height="697" alt="Screenshot 2026-08-18 153731" src="https://github.com/user-attachments/assets/b8db39ca-1b2d-4f0a-8708-3bb78de835e3" />


<img width="905" height="482" alt="Screenshot 2026-08-18 153532" src="https://github.com/user-attachments/assets/ffe57fcd-3958-4aeb-945c-eaa66d47a8b6" />



<img width="992" height="700" alt="Screenshot 2026-08-18 153749" src="https://github.com/user-attachments/assets/af181ac5-a900-431a-b664-5814eb50ddb4" />


## Result
```text

The On-Policy Monte Carlo Control algorithm was successfully implemented using Gymnasium's FrozenLake-v1 environment. The agent learned an improved policy using Monte Carlo returns and an epsilon-greedy strategy.



```
---

## Inference
```text

The learned Q-table represents the estimated value of taking each action in every state. The state-value function is obtained by selecting the maximum Q-value for each state.

The epsilon-greedy policy allows the agent to explore different actions initially and gradually exploit the actions that provide higher estimated returns.

The learning curve shows the change in the average reward as the number of training episodes increases.



```





---

