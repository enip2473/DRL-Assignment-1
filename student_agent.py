import numpy as np
import pickle
import random
from utils import State, get_state, convert_to_table_state, softmax, distance

global_state = State()
global_state.reset()

global_policy_table = pickle.load(open("policy_table.pkl", "rb"))
# print("Policy table loaded!", policy_table)

def get_action(obs, state=None, policy_table=None, train=False):
    if not state:
        state = global_state

    if not policy_table:
        policy_table = global_policy_table

    scale = 1
    if not train:
        scale = 2

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
        state.add_not_target(state.current_target)
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

    # print("Table state:", table_state, policy_table[table_state])
    
    if table_state not in policy_table:
        policy_table[table_state] = np.zeros(4)

    prob = softmax(scale * policy_table[table_state])
    action = np.random.choice(list(range(4)), p=prob)

    return action
