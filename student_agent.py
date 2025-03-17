# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

next_target = 0

def get_action(obs):
    global next_target


    return random.choice([0, 1, 2, 3]) # Choose a random action

