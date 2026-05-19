from copy import deepcopy
import numpy as np
from copy import deepcopy
from wrappers.wrappers import ModifyTerminalStateRewardCartPole, ModifyTerminalStateRewardMountainCar
import time

class LocalSearchAgent:
    '''
    This is an 'abstract' implementation of a Local Search Agent.
    Every Local search agent e.g., HiLLClimbing .. should inherit 
    and implement the functionalities of this exact agent.
    '''
    name = 'Local_Search_Agent'

    def __init__(self, num_actions, domain_name):
        self.num_actions = num_actions
        self.domain_name = domain_name
        self.current_value = -np.inf
        pass

    def act(self, env) -> int:
        '''
        The method responsible for generating an action based on the current
        observation/state of the environment. It either returns a valid action
        or -1 en the case that all the actions results to a terminal state
        '''


        raise NotImplementedError()

    def project_act(self, env) -> int:
        """ 
        This method caalls the act function in rder to return an action.
        If this action is not valid (i.e., -1) then it returns a
        random action since it make no difference (all action evaluated within act() 
        result to a terminal state) 
        """

        action = self.act(env)
        return action if action != -1 else np.random.choice(self.num_actions)
    
    
    def calculate_value(self, t, reward, terminated, truncated):
        if self.domain_name == 'CartPole':
            #In cart pole, we would like the agent to survive as long as it can
            return reward + 0.1 * truncated + t * (not terminated)
        #however, in Mountain Car we would like to reach the goal
        #the faster that it can
        return reward + 0.1 * truncated - t * (not terminated)

    def find_best_route(self, env, seed):
        '''
        method that simulates a whole run.
        it returns the initial state, 
        the path of the actions taken by the agent (when removing the no op action)
        and if the episode terminated succesfully (truncated) or terminated  
        '''
        visulalize = True
        terminated = False
        truncated = False
        action = -1
        states = []
        rewards = []
        actions = []
        values = []
        #in that way, the initial point is always the same
        obs, info = env.reset(seed = seed)
        
        while not (terminated or truncated):
            
            
            """ while hasattr(sim_env, "env"):
                sim_env = sim_env.env """
            #sim_env = sim_env.unwrapped
            if self.domain_name == "CartPole":
                sim_env = ModifyTerminalStateRewardCartPole(deepcopy(env.get_planning_env()))
            elif self.domain_name == "MountainCar":
                sim_env = ModifyTerminalStateRewardMountainCar(deepcopy(env.get_planning_env()))

            action = self.project_act(sim_env)
            state, reward, terminated, truncated, info = env.step(action)
            value = self.calculate_value(
                t = env.t,
                reward = reward,
                terminated = terminated,
                truncated = truncated
            )

            env.render()
            values.append(value)
            self.current_value = value
            states.append(state); rewards.append(reward); actions.append(action)

        return states, rewards, actions, terminated, truncated



class TetsingAgent(LocalSearchAgent):
    def __init__(self, num_actions, domain_name):
        super().__init__(num_actions, domain_name)

    def act(self, env):
        return np.random.choice(self.num_actions)

class HillClimbingAgent(LocalSearchAgent):
    name = 'Hill_Climbing'

    def __init__(self, num_actions, domain_name):
        super().__init__(num_actions, domain_name)

    def act(self, env):
        best_action = -1
        best_value = -np.inf

        for action in range(self.num_actions):
            sim = deepcopy(env)
            _, reward, terminated, truncated, _ = sim.step(action)
            if terminated:
                continue
            v = self.calculate_value(t=sim.env.t, reward=reward, terminated=terminated, truncated=truncated)
            if v > best_value:
                best_value = v
                best_action = action

        return best_action


class RandomRestartHillClimbingAgent(LocalSearchAgent):
    name = 'Random_Restart_Hill_Climbing'

    def __init__(self, num_actions, domain_name, n_restarts=20, horizon=5):
        super().__init__(num_actions, domain_name)
        self.n_restarts = n_restarts
        self.horizon = horizon

    def act(self, env):
        action_values = np.full(self.num_actions, -np.inf)

        for _ in range(self.n_restarts):
            sim = deepcopy(env)
            first_action = np.random.choice(self.num_actions)
            _, reward, terminated, truncated, _ = sim.step(first_action)

            if terminated:
                continue

            total = self.calculate_value(t=sim.env.t, reward=reward, terminated=terminated, truncated=truncated)

            for _ in range(self.horizon - 1):
                if terminated or truncated:
                    break
                a = np.random.choice(self.num_actions)
                _, reward, terminated, truncated, _ = sim.step(a)
                total += self.calculate_value(t=sim.env.t, reward=reward, terminated=terminated, truncated=truncated)

            if total > action_values[first_action]:
                action_values[first_action] = total

        if np.all(action_values == -np.inf):
            return -1
        return int(np.argmax(action_values))


class SimulatedAnnealingAgent(LocalSearchAgent):
    name = 'Simulated_Annealing_Agent'

    def __init__(self, num_actions, domain_name, T_0=2.0, cooling_rate=0.95, schedule='exponential'):
        super().__init__(num_actions, domain_name)
        self.T_0 = T_0
        self.cooling_rate = cooling_rate
        self.schedule = schedule
        self.step_count = 0

    def get_temperature(self):
        n = self.step_count
        if self.schedule == 'exponential':
            return self.T_0 * (self.cooling_rate ** n)
        elif self.schedule == 'linear':
            return max(self.T_0 - self.cooling_rate * n, 1e-3)
        else:  # logarithmic
            return self.T_0 / np.log(n + 2)

    def act(self, env):
        T = self.get_temperature()
        self.step_count += 1

        action_values = {}
        for action in range(self.num_actions):
            sim = deepcopy(env)
            _, reward, terminated, truncated, _ = sim.step(action)
            if not terminated:
                v = self.calculate_value(t=sim.env.t, reward=reward, terminated=terminated, truncated=truncated)
                action_values[action] = v

        if not action_values:
            return -1

        best_action = max(action_values, key=action_values.get)
        candidate = np.random.choice(list(action_values.keys()))
        delta = action_values[candidate] - action_values[best_action]

        if delta >= 0 or (T > 1e-9 and np.random.rand() < np.exp(delta / T)):
            return candidate
        return best_action