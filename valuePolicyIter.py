import numpy as np

# 1. Setup Grid Dimensions
rows, cols = 3, 4

# 2. Initialize Rewards
R = np.full((rows, cols), -0.04)
R[2, 3] = 1.0    # Goal
R[1, 3] = -1.0   # Penalty

terminals = [(2, 3), (1, 3)]
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

gamma = 1.0
epsilon = 1e-4


# 3. Define Movement Rules
def get_next_state(r, c, action):
    if action == "UP":
        next_r, next_c = r + 1, c
    elif action == "DOWN":
        next_r, next_c = r - 1, c
    elif action == "LEFT":
        next_r, next_c = r, c - 1
    elif action == "RIGHT":
        next_r, next_c = r, c + 1

    # Stay in place if moving outside the grid.
    if not (0 <= next_r < rows and 0 <= next_c < cols):
        return r, c

    # Obstacle at (2, 2), represented by array position (1, 1).
    if next_r == 1 and next_c == 1:
        return r, c

    return next_r, next_c


def get_action_distribution(action):
    """Return intended, left-drift, and right-drift actions."""
    if action == "UP":
        return "UP", "LEFT", "RIGHT"
    if action == "DOWN":
        return "DOWN", "RIGHT", "LEFT"
    if action == "LEFT":
        return "LEFT", "DOWN", "UP"
    if action == "RIGHT":
        return "RIGHT", "UP", "DOWN"


def expected_utility(U, r, c, action):
    """Expected successor utility for an action."""
    intended, left, right = get_action_distribution(action)

    ri, ci = get_next_state(r, c, intended)
    rl, cl = get_next_state(r, c, left)
    rr, cr = get_next_state(r, c, right)

    return (
        0.8 * U[ri, ci] +
        0.1 * U[rl, cl] +
        0.1 * U[rr, cr]
    )


def extract_policy(U):
    """Create a greedy policy from a utility table."""
    policy = {}

    for r in range(rows):
        for c in range(cols):
            if (r, c) in terminals:
                policy[(r, c)] = "GOAL" if (r, c) == (2, 3) else "TRAP"
                continue

            policy[(r, c)] = max(
                actions,
                key=lambda action: expected_utility(U, r, c, action)
            )

    return policy


# 4A. Value Iteration
def value_iteration(gamma=1.0, epsilon=1e-4):
    U = np.zeros((rows, cols))

    # Terminal utilities remain fixed.
    for r, c in terminals:
        U[r, c] = R[r, c]

    while True:
        U_next = U.copy()
        delta = 0.0

        for r in range(rows):
            for c in range(cols):
                if (r, c) in terminals:
                    continue

                best_action_value = max(
                    expected_utility(U, r, c, action)
                    for action in actions
                )

                U_next[r, c] = R[r, c] + gamma * best_action_value
                delta = max(delta, abs(U_next[r, c] - U[r, c]))

        U = U_next

        if delta < epsilon:
            break

    return U, extract_policy(U)


# 4B. Policy Iteration
def policy_iteration(gamma=1.0, epsilon=1e-4):
    U = np.zeros((rows, cols))

    # Terminal utilities remain fixed.
    for r, c in terminals:
        U[r, c] = R[r, c]

    # Start with an arbitrary policy.
    policy = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) == (2, 3):
                policy[(r, c)] = "GOAL"
            elif (r, c) == (1, 3):
                policy[(r, c)] = "TRAP"
            else:
                policy[(r, c)] = "UP"

    while True:
        # Policy evaluation
        while True:
            U_next = U.copy()
            delta = 0.0

            for r in range(rows):
                for c in range(cols):
                    if (r, c) in terminals:
                        continue

                    action = policy[(r, c)]
                    U_next[r, c] = (
                        R[r, c] +
                        gamma * expected_utility(U, r, c, action)
                    )
                    delta = max(delta, abs(U_next[r, c] - U[r, c]))

            U = U_next

            if delta < epsilon:
                break

        # Policy improvement
        policy_stable = True

        for r in range(rows):
            for c in range(cols):
                if (r, c) in terminals:
                    continue

                old_action = policy[(r, c)]

                best_action = max(
                    actions,
                    key=lambda action: expected_utility(U, r, c, action)
                )

                policy[(r, c)] = best_action

                if best_action != old_action:
                    policy_stable = False

        if policy_stable:
            break

    return U, policy


# 5. Choose one algorithm

# Value Iteration:
U, policy = value_iteration(gamma, epsilon)

# Or, use Policy Iteration instead:

#U, policy = policy_iteration(gamma, epsilon)


# 6. Display Results
print("--- Final Utility Table ---")
print(np.round(np.flipud(U), 3))

print("\n--- Extracted Policy Layout ---")
p_grid = np.empty((rows, cols), dtype=object)

for r in range(rows):
    for c in range(cols):
        p_grid[r, c] = policy[(r, c)]

print(np.flipud(p_grid))
