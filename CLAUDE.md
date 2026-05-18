# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI course assignment (TUC 2025-2026) implementing local search agents — Hill Climbing, Random Restart Hill Climbing, and Simulated Annealing — for the [ns-gym](https://nsgym.io/) non-stationary reinforcement learning framework. Agents are evaluated on two domains: `CartPole-v1` and `MountainCar-v0`.

## Installation

Requires Python 3.10 (use conda or similar virtualenv). The downgrade of pip/setuptools is mandatory for `gym==0.21` compatibility:

```bash
pip install "setuptools==65.5.0" "pip==21"
pip install ns-gym --no-cache-dir
```

## Running

Run the test/demo script from inside `AI-agents-in-ns-gym/`:

```bash
cd AI-agents-in-ns-gym
python3 test/test_ns_gymnasium.py
```

To switch domain, edit the `domain` variable at the bottom of `test/test_ns_gymnasium.py` (`"CartPole-v1"` or `"MountainCar-v0"`).

## Architecture

All code lives under `AI-agents-in-ns-gym/`.

**`agents/local_search_agents.py`** — core agent hierarchy:
- `LocalSearchAgent` (base): provides `find_best_route()`, `project_act()`, and `calculate_value()`. The `find_best_route()` loop runs the real env; at each timestep it deep-copies the planning env (`env.get_planning_env()`), wraps it with a reward modifier, and passes it to `project_act(sim_env)`.
- `act(env)` is the only method subclasses must implement. It receives a simulation copy of the env and must return an action index, or `-1` if all actions lead to a terminal state.
- `calculate_value()` encodes domain-specific objectives: CartPole rewards survival time; MountainCar penalizes time (reach goal fast).

**`wrappers/wrappers.py`** — gymnasium wrappers:
- `ModifyTerminalStateRewardCartPole` / `ModifyTerminalStateRewardMountainCar`: wrap the planning env copy passed to `act()`. They add small noise to rewards and implement analytic `reward(action)` methods for one-step lookahead without calling `env.step()`.
- `ForwardWrapper`: transparent attribute proxy — forwards unknown attributes to the underlying unwrapped env.

**`cfgs/cfg_dqn.py`** — configuration dict for a DQN agent (planned for a later phase; not yet integrated).

**Key ns-gym concepts used in `test/test_ns_gymnasium.py`:**
- `NSClassicControlWrapper`: wraps a gymnasium env to make physics parameters non-stationary.
- `ContinuousScheduler` / `PeriodicScheduler`: control when parameters change.
- `IncrementUpdate` / `RandomWalk`: define how parameters change.
- `tunable_params` dict maps parameter names (e.g., `"masspole"`, `"gravity"`) to update functions.

## What Needs Implementation

- `HillClimbingAgent.act()` in `agents/local_search_agents.py:105`
- Random Restart Hill Climbing (new class, not yet stubbed)
- `SimulatedAnnealingAgent.act()` in `agents/local_search_agents.py:112`
- `ModifyTerminalStateRewardMountainCar` reward shaping (commented-out sections in `wrappers/wrappers.py`)
