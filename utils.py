import numpy as np

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

def get_state(obs, state):
    loc = [[0, 0] for i in range(4)]
    taxi_row, taxi_col, loc[0][0], loc[0][1], loc[1][0], loc[1][1], loc[2][0], loc[2][1], loc[3][0], loc[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    target = state.current_target
    rel_pos = [loc[target][0] - taxi_row, loc[target][1] - taxi_col]
    table_state = (rel_pos[0], rel_pos[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    return table_state

def convert_to_table_state(state):
    rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = state
    rel_x = max(-3, min(3, rel_x))
    rel_y = max(-3, min(3, rel_y))
    return (rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west)

def softmax(x):
    y = x - np.max(x)
    return np.exp(y) / np.sum(np.exp(y), axis=0)

