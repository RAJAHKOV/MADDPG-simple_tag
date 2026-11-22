# Implementation and Optimization of MADDPG with PyTorch and PettingZoo (Branch: dev-alternative)

## Project Overview

As an alternative development branch to the main project, this branch independently reproduces the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm and deeply customizes it in the PettingZoo `simple_tag_v3` environment. With `num_obstacles=0`, 3 adversaries must learn to cooperate and utilize boundary constraints to hunt down 1 agent.

## Core Contribution: Comparative Study of Environmental Constraints

In this branch, I independently designed and implemented two different boundary constraint mechanisms to solve the issue of agents flying out of physical boundaries during training, which degrades training efficiency.

### Approach 1: Soft Constraint - Boundary Penalty

- Files: main_penalty.py / eval_penalty.py
- Implementation: Calculates the out-of-bounds distance after `env.step()` and applies a negative reward penalty based on a coefficient (e.g., 0.5).
- Mechanism: Allows temporary boundary violations, using reward signals to "teach" agents to avoid the edges.
- Advantage: Maintains dynamic continuity of the environment, serving as a baseline for comparison.

### Approach 2: Hard Constraint - Physical Rebound

- Files: main_rebound.py / eval_rebound.py
- Implementation: Overrides the underlying physics engine's `World.integrate_state` method. When an agent's coordinates touch the [-1.5, 1.5] boundary, its velocity vector is reversed (bounce) or the position is clamped.
- Mechanism: Completely prevents boundary violations at the physics integration level, ensuring `next_obs` perfectly matches the true physical state.
- Advantage: 100% eliminates the possibility of agents flying out of the rendered view. It forces both predators and prey into extreme cornering and escaping maneuvers within a confined space, significantly enhancing the intensity of the game and the visual quality of the GIFs.

## Compatibility and Engineering Optimizations

- API Adaptation: Adapted to the major updates in `mpe2` (PettingZoo v3) (e.g., `step` returning 5 values, `env.agents` shrinking due to early agent death). Reconstructed the state-completion mechanism to ensure stable sampling in the Replay Buffer.
- Customized Evaluation Rendering: The eval scripts support pure Python drawing of the boundary box and real-time dynamic scoreboards (step rewards and cumulative episode rewards).
- Automated Metrics: Automatically computes and outputs the Capture Rate, Escape Rate, and Average Capture Step.

## Usage

1. Training (Penalty Mode):

```{python}
python main_penalty.py simple_tag_v3 --episode_num 5000
```

2. Training (Rebound Mode):

```{python}
python main_rebound.py simple_tag_v3 --episode_num 5000
```

3. Evaluation & GIF Generation:

   Configure the `folder` path to your results directory in `eval_rebound.py` or `eval_penalty.py`, then run the script to generate high-quality GIFs with score panels.

## File Structure

- main_penalty.py / eval_penalty.py : Training and evaluation scripts for the penalty mechanism.
- main_rebound.py / eval_rebound.py : Training and evaluation scripts for the rebound mechanism.
- MADDPG.py / Agent.py / Buffer.py : Core components of the MADDPG algorithm.

