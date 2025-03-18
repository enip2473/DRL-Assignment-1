import numpy as np
import random
import pickle
from simple_custom_taxi_env import TaxiEnv

class State:
    def __init__(self):
        self.current_target = 0
        self.has_passenger = False
    
    def set_target(self, target):
        self.current_target = target
    
    def set_passenger(self, passenger):
        self.has_passenger = passenger

    def reset(self):
        self.current_target = 0
        self.has_passenger = False

is_train = False
policy_table = {}

def softmax(x):
    y = x - np.max(x)
    return np.exp(y) / np.sum(np.exp(y), axis=0)

state = State()

def get_state(obs):
    loc = [[0, 0] for i in range(4)]
    taxi_row, taxi_col, loc[0][0], loc[0][1], loc[1][0], loc[1][1], loc[2][0], loc[2][1], loc[3][0], loc[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    target = state.current_target
    rel_pos = [loc[target][0] - taxi_row, loc[target][1] - taxi_col] 
    table_state = (rel_pos[0], rel_pos[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    return table_state
  
def get_action(obs):
    table_state = get_state(obs)
    rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = table_state
    
    if passenger_look == 0:
        state.set_passenger(False)

    global policy_table
    if not policy_table and not is_train:
        with open("policy_table.pkl", "rb") as f:
            policy_table = pickle.load(f)
    
    if table_state not in policy_table:
        policy_table[table_state] = np.zeros(4)

    if (rel_x, rel_y) == (0, 0) and not state.has_passenger and passenger_look == True:
        state.set_passenger(True)
        target = state.current_target
        next_target = [i for i in range(4) if i != target]
        state.set_target(random.choice(next_target))
        return 4
    
    if (rel_x, rel_y) == (0, 0) and state.has_passenger and destination_look == True:
        state.set_passenger(False)
        target = state.current_target
        next_target = [i for i in range(4) if i != target]
        state.set_target(random.choice(next_target))
        return 5
    
    prob = softmax(policy_table[table_state])
    action = np.random.choice(list(range(4)), p=prob)

    if (rel_x, rel_y) == (0, 0):
        target = state.current_target
        next_target = [i for i in range(4) if i != target]
        state.set_target(random.choice(next_target))
    
    return action


def run_one_episode(env, render=False):
    obs, _ = env.reset()
    state.reset()
    total_reward = 0
    done = False
    step_count = 0

    trajectory = []

    if render:
        env.render_env((obs[0], obs[1]), action=None, step=step_count, fuel=env.current_fuel)

    while not done:
        prev_obs = obs
        prev_state = get_state(prev_obs)
        
        action = get_action(obs)

        prev_target = state.current_target
        obs, reward, done, _ = env.step(action)
        current_state = get_state(obs)
        current_target = state.current_target
        trajectory.append((prev_state, action, reward))

        extra_reward = -1
        if prev_target == current_target:
            prev_distance = abs(prev_state[0]) + abs(prev_state[1])
            current_distance = abs(current_state[0]) + abs(current_state[1])
            if current_distance < prev_distance:
                extra_reward = 10
            elif current_distance > prev_distance:
                extra_reward = -10
        
        reward += extra_reward

        total_reward += reward
        step_count += 1
        if render:
            env.render_env((obs[0], obs[1]), action=action, step=step_count, fuel=env.current_fuel)

    G = 0
    gamma = 0.5
    for t in reversed(range(len(trajectory))):
        state_t, action_t, reward_t = trajectory[t]
        G = reward_t + gamma * G
        trajectory[t] = (state_t, action_t, G)

    return trajectory, total_reward


def update_policy_table(trajectory, alpha = 0.001):
    for state, action, reward in trajectory:
        if action == 4 or action == 5:
            continue
        if not state in policy_table:
            policy_table[state] = np.zeros(4)
        logits = policy_table[state]
        action_probs = softmax(logits)
        for a in range(4):
            if a == action:
                grad = (1 - action_probs[a])
            else:
                grad = -action_probs[a]
            policy_table[state][a] += alpha * reward * grad


def main(num_episodes=1000, render=False):
    global policy_table
    episode_rewards = []
    
    for _ in range(num_episodes):
        grid_size = random.randint(5, 8)
        num_obstacles = random.randint(0, grid_size)
        env = TaxiEnv(grid_size=grid_size, fuel_limit=1000, num_obstacles=num_obstacles)
        render = _ % 100 == 99
        trajectory, total_reward = run_one_episode(env, render)
        episode_rewards.append(total_reward)
        update_policy_table(trajectory)
        if _ % 100 == 99:
            print(f"Episode {_+1}: {np.mean(episode_rewards[-100:])}")

    with open("policy_table.pkl", "wb") as f:
        pickle.dump(policy_table, f)


if __name__ == "__main__":
    is_train = True
    main(num_episodes=10000, render=False)


