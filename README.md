# Solving a Stochastic Grid-World Markov Decision Process Using Value Iteration and Policy Iteration

A compact Python implementation of two dynamic-programming methods for solving a stochastic grid-world Markov decision process (MDP):

- **Value Iteration**
- **Policy Iteration**

The program calculates the utility of each state and prints the resulting greedy policy.

## Why sequential decision problems matter

A **sequential decision problem** is one in which a decision made now affects what choices, rewards, and risks are available later. The objective is not simply to choose the action with the best immediate result; it is to choose actions that lead to the best *long-term* outcome despite uncertainty.

This grid world is a small example: moving toward the goal may be worthwhile, but a movement can drift unexpectedly toward the trap. The agent must therefore balance the immediate step cost, the chance of reaching the reward, and the risk of a future penalty.

Solving problems like this is important because many real systems operate through a sequence of connected decisions:

- **Robotics and navigation:** choosing safe routes while accounting for imperfect movement and obstacles.
- **Operations and logistics:** planning inventory, deliveries, and resources when demand and travel conditions are uncertain.
- **Finance:** weighing current returns against longer-term risk and future market outcomes.
- **Healthcare:** selecting treatments over time as a patient's condition and response evolve.
- **Reinforcement learning:** training an agent to act effectively from feedback rather than fixed instructions.

Value iteration and policy iteration provide principled ways to solve a Markov decision process (MDP). They evaluate both immediate rewards and expected future utility, producing a policy that tells an agent what action to take in each state. This makes them foundational techniques for planning under uncertainty.

## Grid-world configuration

The environment is a 3 × 4 grid:

```text
+-------+-------+-------+--------+
| (2,0) | (2,1) | (2,2) | Goal   |
+-------+-------+-------+--------+
| (1,0) | Block | (1,2) | Trap   |
+-------+-------+-------+--------+
| (0,0) | (0,1) | (0,2) | (0,3)  |
+-------+-------+-------+--------+
```

| Element | Details |
| --- | --- |
| Normal-state reward | `-0.04` |
| Goal at `(2, 3)` | terminal state with reward `+1.0` |
| Trap at `(1, 3)` | terminal state with reward `-1.0` |
| Blocked cell | `(1, 1)` |
| Discount factor | `γ = 1.0` |
| Convergence threshold | `ε = 1e-4` |

Coordinates use `(row, column)` with `(0, 0)` at the bottom-left. The displayed arrays are flipped vertically so the top row appears first.

## Movement model

The available actions are `UP`, `DOWN`, `LEFT`, and `RIGHT`.

An intended action succeeds with probability **0.8**. With probability **0.1** each, the agent drifts to the action on its left or right. If a move would leave the grid or enter the blocked cell, the agent stays in its current state.

## Requirements

- Python 3.8 or newer
- NumPy

Install the dependency:

```bash
python3 -m pip install numpy
```

## Run

```bash
python3 valuePolicyIter.py
```

By default, the script runs value iteration and prints a utility table followed by the extracted policy.

Example output:

```text
--- Final Utility Table ---
[[ 0.812  0.868  0.918  1.   ]
 [ 0.762  0.796  0.66  -1.   ]
 [ 0.705  0.655  0.611  0.388]]

--- Extracted Policy Layout ---
[['RIGHT' 'RIGHT' 'RIGHT' 'GOAL']
 ['UP' 'UP' 'UP' 'TRAP']
 ['UP' 'LEFT' 'LEFT' 'LEFT']]
```

## Choose an algorithm

The active algorithm is selected near the bottom of `valuePolicyIter.py`.

### Value iteration (default)

```python
U, policy = value_iteration(gamma, epsilon)
```

Value iteration repeatedly applies the Bellman optimality update until the largest utility change is smaller than `epsilon`.

### Policy iteration

Comment out the value-iteration line and enable this one:

```python
U, policy = policy_iteration(gamma, epsilon)
```

Policy iteration alternates between evaluating the current policy and improving it until no action changes.



## Customize the environment

Edit these values near the top of the script to experiment with other MDPs:

- `rows`, `cols` — grid size
- `R` — reward table
- `terminals` — terminal-state coordinates
- `gamma` — discount factor
- `epsilon` — stopping tolerance
- `get_next_state()` — boundaries and blocked-cell behavior
- `get_action_distribution()` / `expected_utility()` — transition dynamics

## Notes

- Terminal-state utilities stay fixed at their assigned rewards.
- The policy grid includes a label for every non-terminal coordinate, including the blocked coordinate. Since that cell cannot be entered, its displayed action is not part of the reachable environment.


