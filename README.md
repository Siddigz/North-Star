# Arctic Ship Routing Optimizer - uOttaHack 8

<p align="center">
  <img src="visuals/fastest.png" alt="Fastest route mode screenshot" width="49%" />
  <img src="visuals/eco.png" alt="Eco-friendly route mode screenshot" width="49%" />
</p>
<p align="center">
  <sub>Left: <b>Fastest</b> route with the <b>time grid/layer</b> visible. Right: <b>Eco-Friendly</b> route.</sub>
</p>

## Project Overview
This project is an advanced ship routing and pathfinding tool designed for navigating the challenging conditions of the Arctic. It utilizes a **Pareto-optimal label-setting algorithm** to calculate multiple route options based on different optimization priorities: **Time**, **Fuel Consumption**, and **Safety (Risk)**.

Developed during **uOttaHack 8**, this tool aims to provide maritime navigators with data-driven insights into the most efficient and safest paths through ice-prone waters.

## The Team
This project was developed by:
- **Arwa Khan**
- **Hasan Ahmed**
- **Shanzay Chaudhry**
- **Siddig Ahmed**

## Key Features
- **Multi-Objective Optimization**: Choose between "Fastest", "Eco-Friendly", and "Safest" routes.
- **Dynamic Environmental Analysis**: Analyzes satellite-style imagery to determine ice concentration and navigability.
- **Spatially Coherent Environmental Noise**: Uses Perlin noise to simulate realistic and varying weather/ice conditions.
- **Interactive Ship Selection**: Support for multiple ship types with unique attributes (speed, fuel consumption, durability) loaded directly from technical specifications.
- **Interactive Map Interface**: Set start and end points directly on the map and visualize the optimized routes with smooth animations.
- **Grid Visualization Layers**: Toggle between different data layers (Risk, Time, Fuel) to understand the underlying environmental costs.

## Walkthrough

**1. Pick a ship.** Each ship's speed, fuel use and durability come from `Ships.xlsx` and change how
routes are scored.

![Ship selection screen with the Icebreaker selected](docs/ship-select.png)

**2. Set points A and B and get routes.** The router finds the Pareto-optimal set of routes, then
offers the fastest, the most fuel-efficient, the safest, and a balanced "Optimal" pick.

![The balanced optimal route from the Beaufort Sea to Baffin Bay](docs/route-optimal.png)

**3. See why a route goes where it does.** Grid layers colour every water cell by its time, fuel or
risk cost. Here the safest route is drawn over the risk layer.

![The safest route over the risk layer, green for low risk and red for high](docs/risk-layer.png)

**4. Compare the numbers.** Clicking a route opens its time, fuel and risk totals.

<p align="center"><img src="docs/route-metrics.png" alt="Bar chart of the optimal route's time, fuel and risk" width="60%" /></p>

## Technical Stack
- **Python**: Core logic and algorithm.
- **Pygame**: Graphical User Interface and map rendering.
- **Openpyxl**: Ship data management from Excel.
- **Noise**: Perlin noise generation for environmental simulation.
- **Custom Routing Engine**: Implementation of a Pareto-optimal label-setting algorithm for complex pathfinding.

## How to Run
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install pygame openpyxl noise
   ```
3. Run the main application:
   ```bash
   python main.py
   ```
