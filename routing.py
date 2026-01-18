# routing.py
# Multi-objective routing using a Pareto-optimal label-setting algorithm
# Objectives: minimize (risk, time, fuel)

import heapq
import math


# ----------------------------
# Data Structures
# ----------------------------

class Label:
    def __init__(self, risk, time, fuel, predecessor):
        self.risk = risk
        self.time = time
        self.fuel = fuel
        self.predecessor = predecessor  # (node, label)

    def __lt__(self, other):
        # Required for heapq; lexicographic ordering
        return (self.risk, self.time, self.fuel) < (other.risk, other.time, other.fuel)


def dominates(a, b):
    """
    Label A dominates Label B if:
    A.risk <= B.risk AND
    A.time <= B.time AND
    A.fuel <= B.fuel
    AND at least one is strictly less
    """
    return (
        a.risk <= b.risk and
        a.time <= b.time and
        a.fuel <= b.fuel and
        (a.risk < b.risk or a.time < b.time or a.fuel < b.fuel)
    )


# ----------------------------
# Helper Functions
# ----------------------------

def neighbors(node, grid_size):
    """
    4-connected grid (up, down, left, right)
    """
    r, c = node
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            yield (nr, nc)


def distance(a, b):
    """
    Grid distance between adjacent cells
    """
    return 1.0


# ----------------------------
# Core Algorithm
# ----------------------------

def pareto_optimal_path(
    grid,
    start,
    goal,
    ship,
    alpha=0.5,
    gamma=0.3
):
    """
    grid[r][c] must have:
        - grid[r][c].risk     (base risk, e.g. 1–10)
        - grid[r][c].current  (environmental current strength)

    ship must have:
        - ship.base_speed
        - ship.base_fuel_rate
        - ship.durability
    """

    GRID_SIZE = len(grid)

    # Label sets per node
    LABELS = {
        (r, c): []
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
    }

    # Priority queue
    OPEN = []

    start_label = Label(
        risk=0.0,
        time=0.0,
        fuel=0.0,
        predecessor=None
    )

    LABELS[start].append(start_label)
    heapq.heappush(OPEN, (start_label, start))

    # Main loop
    while OPEN:
        current_label, current_node = heapq.heappop(OPEN)

        if current_node == goal:
            # Do NOT early-exit (Pareto search)
            continue

        for nb in neighbors(current_node, GRID_SIZE):
            d = distance(current_node, nb)

            cell = grid[nb[0]][nb[1]]

            base_risk = cell.risk
            current_strength = cell.current

            # ----------------------------
            # COST COMPUTATION
            # ----------------------------

            effective_risk = (
                base_risk
                + alpha * current_strength / ship.durability
            )
            effective_risk = max(0.0, effective_risk)

            fuel_cost = (
                ship.base_fuel_rate * d
                * (1 + gamma * current_strength / ship.durability)
            )

            time_cost = d / ship.base_speed

            new_label = Label(
                risk=current_label.risk + effective_risk,
                time=current_label.time + time_cost,
                fuel=current_label.fuel + fuel_cost,
                predecessor=(current_node, current_label)
            )

            # ----------------------------
            # DOMINANCE CHECK
            # ----------------------------

            dominated = False
            for existing in LABELS[nb]:
                if dominates(existing, new_label):
                    dominated = True
                    break

            if dominated:
                continue

            # Remove labels dominated by new_label
            LABELS[nb] = [
                lbl for lbl in LABELS[nb]
                if not dominates(new_label, lbl)
            ]

            LABELS[nb].append(new_label)
            heapq.heappush(OPEN, (new_label, nb))

    return LABELS[goal]


# ----------------------------
# Path Reconstruction (Optional)
# ----------------------------

def reconstruct_path(label):
    """
    Returns path as list of nodes from start to goal
    """
    path = []
    current = label
    while current and current.predecessor:
        node, prev_label = current.predecessor
        path.append(node)
        current = prev_label
    return list(reversed(path))
