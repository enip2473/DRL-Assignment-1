import numpy as np
import random

class State:
    def __init__(self):
        self.current_target = 0
        self.current_phase = 0
        self.not_passenger = []
        self.not_target = []
    
    def __str__(self):
        return f"Current target: {self.current_target}, Current phase: {self.current_phase}, Not passenger: {self.not_passenger}, Not target: {self.not_target}"
    
    def set_new_target(self):
        if self.current_phase == 0:
            possible_target = [i for i in range(4) if i not in self.not_passenger]
        elif self.current_phase == 1:
            possible_target = [i for i in range(4) if i not in self.not_target]
        if not possible_target:
            print("Error! Restarting...")
            self.reset()
            possible_target = [i for i in range(4)]
        self.current_target = random.choice(possible_target)
    
    def add_not_passenger(self, index):
        if index not in self.not_passenger:
            self.not_passenger.append(index)

    def add_not_target(self, index):
        if index not in self.not_target:
            self.not_target.append(index)

    def reset(self):
        self.current_target = 0
        self.current_phase = 0
        self.not_passenger = []
        self.not_target = []



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

def distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])