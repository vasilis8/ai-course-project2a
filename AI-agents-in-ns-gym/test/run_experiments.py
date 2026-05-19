import gymnasium as gym
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.local_search_agents import HillClimbingAgent, RandomRestartHillClimbingAgent, SimulatedAnnealingAgent
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


def make_env(domain, max_timesteps):
    env = gym.make(domain, render_mode=None, max_episode_steps=max_timesteps)
    name, _ = domain.split("-")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_fn1 = IncrementUpdate(scheduler_1, k=1)
    update_fn2 = RandomWalk(scheduler_2)

    if name == "CartPole":
        tunable_params = {"masspole": update_fn1, "gravity": update_fn2}
    else:
        scheduler_3 = ContinuousScheduler()
        update_fn3 = IncrementUpdate(scheduler_3, k=1)
        tunable_params = {"gravity": update_fn1, "force": update_fn3}

    return NSClassicControlWrapper(env, tunable_params, change_notification=True)


def run_episodes(agent_class, agent_kwargs, domain, max_timesteps, n_episodes, seed=42):
    name, _ = domain.split("-")
    env = make_env(domain, max_timesteps)
    agent = agent_class(num_actions=env.action_space.n, domain_name=name, **agent_kwargs)

    rewards_per_ep = []
    for ep in range(n_episodes):
        _, rewards, _, _, _ = agent.find_best_route(env, seed + ep)
        rewards_per_ep.append(float(np.sum(rewards)))

    env.close()
    return rewards_per_ep


def experiment_mean_vs_timesteps(domain, max_timesteps_list, n_episodes=100, seed=42):
    agents = [
        ('Hill Climbing', HillClimbingAgent, {}),
        ('Random Restart HC', RandomRestartHillClimbingAgent, {}),
        ('Simulated Annealing', SimulatedAnnealingAgent, {}),
    ]

    results = {name: [] for name, _, _ in agents}

    for ts in max_timesteps_list:
        print(f"  max_timesteps={ts}")
        for name, cls, kwargs in agents:
            ep_rewards = run_episodes(cls, kwargs, domain, ts, n_episodes, seed)
            mean_r = float(np.mean(ep_rewards))
            results[name].append(mean_r)
            print(f"    {name}: {mean_r:.3f}")

    return results


def experiment_rewards_over_episodes(domain, max_timesteps=100, n_episodes=1000, seed=42):
    agents = [
        ('Hill Climbing', HillClimbingAgent, {}),
        ('Random Restart HC', RandomRestartHillClimbingAgent, {}),
        ('Simulated Annealing', SimulatedAnnealingAgent, {}),
    ]

    results = {}
    for name, cls, kwargs in agents:
        print(f"  {name}")
        results[name] = run_episodes(cls, kwargs, domain, max_timesteps, n_episodes, seed)

    return results


def plot_mean_rewards(domain_name, max_timesteps_list, results, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_name, means in results.items():
        ax.plot(max_timesteps_list, means, marker='o', label=agent_name)
    ax.set_xlabel('Max Timesteps per Episode')
    ax.set_ylabel('Mean Cumulative Reward (100 episodes)')
    ax.set_title(f'{domain_name} - Mean Reward vs Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_rewards_over_episodes(domain_name, results, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent_name, rewards in results.items():
        ax.plot(rewards, alpha=0.7, label=agent_name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'{domain_name} - Reward per Episode (max_timesteps=100, 1000 episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    max_timesteps_list = [10, 100, 500, 1000]
    domains = [("CartPole-v1", "CartPole"), ("MountainCar-v0", "MountainCar")]

    for domain, domain_name in domains:
        print(f"\n=== {domain_name}: mean reward vs max_timesteps ===")
        res = experiment_mean_vs_timesteps(domain, max_timesteps_list)
        plot_mean_rewards(domain_name, max_timesteps_list, res,
                          f"results/{domain_name.lower()}_mean_rewards.png")

    for domain, domain_name in domains:
        print(f"\n=== {domain_name}: per-episode rewards (1000 eps, ts=100) ===")
        res = experiment_rewards_over_episodes(domain)
        plot_rewards_over_episodes(domain_name, res,
                                   f"results/{domain_name.lower()}_episode_rewards.png")
