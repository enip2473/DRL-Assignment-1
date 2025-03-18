# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

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

state = State()
state.reset()
policy_table = pickle.load(open("policy_table.pkl", "rb"))

def softmax(x):
    y = x - np.max(x)
    return np.exp(y) / np.sum(np.exp(y), axis=0)

def get_state(obs):
    loc = [[0, 0] for i in range(4)]
    taxi_row, taxi_col, loc[0][0], loc[0][1], loc[1][0], loc[1][1], loc[2][0], loc[2][1], loc[3][0], loc[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    target = state.current_target
    rel_pos = [loc[target][0] - taxi_row, loc[target][1] - taxi_col] 
    table_state = (rel_pos[0], rel_pos[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    return table_state

def get_action(obs):
    table_state = get_state(obs)
    print("Table state:", table_state)
    rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = table_state
    
    if passenger_look == 0:
        state.set_passenger(False)

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