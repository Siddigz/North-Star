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


class Ship:
    def __init__(self, base_speed, base_fuel_rate, durability):
        self.base_speed = base_speed
        self.base_fuel_rate = base_fuel_rate
        self.durability = durability


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

def neighbors(node, rows, cols):
    """
    8-connected grid (up, down, left, right + diagonals)
    """
    r, c = node
    # Cardinal and diagonal directions
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1) # Diagonal
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def distance(a, b):
    """
    Euclidean distance between grid cells
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def line_of_sight(grid, start, end):
    """
    Check if a straight line between two nodes is entirely over water.
    Samples points at small intervals and checks a small buffer for clearance.
    """
    r0, c0 = start
    r1, c1 = end
    
    # Distance in grid cells
    dist = distance(start, end)
    if dist == 0:
        return True
        
    # Number of samples based on distance (one sample every 0.2 grid cells for high precision)
    num_samples = int(dist * 5) + 1
    
    ROWS = len(grid)
    COLS = len(grid[0])
    
    for i in range(num_samples + 1):
        t = i / num_samples
        curr_r = r0 + (r1 - r0) * t
        curr_c = c0 + (c1 - c0) * t
        
        # Check a small neighborhood around the point for a "fat line" clearance
        # This ensures the line doesn't just graze land corners
        for dr in [-0.3, 0.3]:
            for dc in [-0.3, 0.3]:
                check_r = int(round(curr_r + dr))
                check_c = int(round(curr_c + dc))
                
                if not (0 <= check_r < ROWS and 0 <= check_c < COLS):
                    return False
                if not getattr(grid[check_r][check_c], 'is_clickable', False):
                    return False
            
    return True


def prune_path(grid, path):
    """
    Simplify the path by removing intermediate nodes if a direct 
    line of sight exists between waypoints (String Pulling).
    """
    if len(path) < 3:
        return path
    
    pruned = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Look for the furthest visible node in the remaining path
        furthest_visible = current_idx + 1
        for check_idx in range(len(path) - 1, current_idx + 1, -1):
            if line_of_sight(grid, path[current_idx], path[check_idx]):
                furthest_visible = check_idx
                break
        
        pruned.append(path[furthest_visible])
        current_idx = furthest_visible
        
    return pruned


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
        - grid[r][c].time     (time weight/multiplier for the cell)
        - grid[r][c].fuel     (fuel consumption multiplier for the cell)
        - grid[r][c].weather  (environmental/weather factor)

    ship must have:
        - ship.base_speed
        - ship.base_fuel_rate
        - ship.durability
    """

    ROWS = len(grid)
    COLS = len(grid[0]) if ROWS > 0 else 0

    # Label sets per node (using 2D list for faster access than dict)
    LABELS = [[[] for _ in range(COLS)] for _ in range(ROWS)]

    # Priority queue: stores (total_priority, label, node)
    OPEN = []

    start_label = Label(
        risk=0.0,
        time=0.0,
        fuel=0.0,
        predecessor=None
    )

    # Pre-calculate ship constants to avoid redundant math in loop
    inv_durability = 1.0 / max(0.1, ship.durability)
    inv_speed = 1.0 / max(0.1, ship.base_speed)
    base_fuel = ship.base_fuel_rate

    # Heuristic: Euclidean distance * inv_speed
    h_time = distance(start, goal) * inv_speed
    priority = (0.0, h_time, 0.0) # (risk, time, fuel) priority

    LABELS[start[0]][start[1]].append(start_label)
    heapq.heappush(OPEN, (priority, start_label, start))

    # Main loop
    while OPEN:
        _, current_label, current_node = heapq.heappop(OPEN)
        cr, cc = current_node

        # Early pruning: if current label is already dominated by a label at the goal, skip it
        goal_labels = LABELS[goal[0]][goal[1]]
        if goal_labels:
            if any(dominates(goal_label, current_label) for goal_label in goal_labels):
                continue

        if current_node == goal:
            continue

        for nb in neighbors(current_node, ROWS, COLS):
            nr, nc = nb
            cell = grid[nr][nc]
            
            # Navigation constraint: only traverse clickable (water) cells
            if not getattr(cell, 'is_clickable', False):
                continue

            d = distance(current_node, nb)

            # ----------------------------
            # COST COMPUTATION (Optimized)
            # ----------------------------

            # Risk: additive, influenced by weather and durability
            effective_risk = (cell.risk + alpha * cell.weather * inv_durability)
            effective_risk = max(0.0, effective_risk)

            # Fuel: cell multiplier * base rate, influenced by weather and durability
            fuel_cost = (base_fuel * cell.fuel) * (1 + gamma * cell.weather * inv_durability) * d

            # Time: cell time weight * d * inv_speed
            time_cost = (cell.time * d) * inv_speed

            new_label = Label(
                risk=current_label.risk + effective_risk,
                time=current_label.time + time_cost,
                fuel=current_label.fuel + fuel_cost,
                predecessor=(current_node, current_label)
            )

            # ----------------------------
            # DOMINANCE CHECK
            # ----------------------------
            
            nb_labels = LABELS[nr][nc]
            dominated = False
            for existing in nb_labels:
                if dominates(existing, new_label):
                    dominated = True
                    break

            if dominated:
                continue

            # Remove labels dominated by new_label
            LABELS[nr][nc] = [
                lbl for lbl in nb_labels
                if not dominates(new_label, lbl)
            ]

            LABELS[nr][nc].append(new_label)
            
            # A* Heuristic: estimate remaining time to goal
            h_time = distance(nb, goal) * inv_speed
            priority = (new_label.risk, new_label.time + h_time, new_label.fuel)
            
            heapq.heappush(OPEN, (priority, new_label, nb))

    return LABELS[goal[0]][goal[1]]


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
