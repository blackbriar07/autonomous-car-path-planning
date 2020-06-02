import numpy as np
import random
# variables

grid = 5
state_size = 3
action_size = 6

#initial_state = [np.random.randint(0,high=5),np.random.randint(0,high=5),np.random.randint(6,high=10)]
#initial_state = [0,1,7]
target_state = [3,4,8]
blockages = []



EPISODES = 2000
stats_per = 20
steps_in_ep = 20