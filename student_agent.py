import numpy as np
import pickle
import random
from utils import State, get_state, convert_to_table_state, softmax

state = State()

policy_table = pickle.load(open("policy_table.pkl", "rb"))


def get_action(obs):
    current_state = get_state(obs, state)
    rel_x, rel_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = current_state
    
    table_state = convert_to_table_state(current_state)

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