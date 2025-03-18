import numpy as np
import random
import pickle
from simple_custom_taxi_env import TaxiEnv
from utils import State, get_state, convert_to_table_state, softmax, distance

is_train = False
policy_table = {}

state = State()
  
def get_action(obs):
    loc = [[0, 0] for i in range(4)]
    taxi_row, taxi_col, loc[0][0], loc[0][1], loc[1][0], loc[1][1], loc[2][0], loc[2][1], loc[3][0], loc[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    current_state = get_state(obs, state)
    rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = current_state
    
    if state.current_phase == 1 and passenger_look == False: # Wrongly picked up passenger
        state.current_phase = 0
        state.set_new_target()
        current_state = get_state(obs, state)
        rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = current_state

    if (rel_x, rel_y) == (0, 0) and state.current_phase == 0 and passenger_look == True:
        state.current_phase = 1
        state.not_passenger = [state.current_target]
        state.not_target.append(state.current_target)
        state.set_new_target()
        return 4
    
    if (rel_x, rel_y) == (0, 0) and state.current_phase == 1 and destination_look == True:
        state.current_phase = 0
        state.not_passenger = []
        state.not_target = []
        state.set_new_target()
        return 5

    for i in range(4):
        if distance((taxi_row, taxi_col), loc[i]) <= 1:
            if not passenger_look:
                state.add_not_passenger(i)
            if not destination_look:
                state.add_not_target(i)
        
    if (rel_x, rel_y) == (0, 0):
        state.set_new_target()
    
    current_state = get_state(obs, state)
    table_state = convert_to_table_state(current_state)
    
    if table_state not in policy_table:
        policy_table[table_state] = np.zeros(4)

    prob = softmax(policy_table[table_state])
    action = np.random.choice(list(range(4)), p=prob)

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
        print(state)

    while not done:
        prev_obs = obs
        prev_state = get_state(prev_obs, state)
        prev_target = state.current_target
        prev_table_state = convert_to_table_state(prev_state)

        action = get_action(obs)
        
        obs, reward, done, _ = env.step(action)
        current_state = get_state(obs, state)
        current_target = state.current_target

        extra_reward = 0
        if prev_target == current_target:
            prev_distance = abs(prev_state[0]) + abs(prev_state[1])
            current_distance = abs(current_state[0]) + abs(current_state[1])
            if current_distance < prev_distance:
                extra_reward += 2
            else:
                extra_reward -= 2
        
        reward += extra_reward
        reward = min(10, max(-10, reward))
        
        trajectory.append((prev_table_state, action, reward))
        total_reward += reward
        step_count += 1
        if render:
            env.render_env((obs[0], obs[1]), action=action, step=step_count, fuel=env.current_fuel)
            print(state)

    G = 0
    gamma = 0.1
    for t in reversed(range(len(trajectory))):
        state_t, action_t, reward_t = trajectory[t]
        G = reward_t + gamma * G
        trajectory[t] = (state_t, action_t, G)

    return trajectory, total_reward


def update_policy_table(trajectory, alpha = 0.01):
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
        grid_size = random.randint(5, 10)
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


