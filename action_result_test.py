import numpy as np
from variables import *
import math


#target_state = [2,5,8]
#blockages = []
#blockages = [[2,1],[2,2],[2,3],[2,4]]
#step_time_decay = 0.002
#blockages = [[0,5],[3,4],[2,7],[1,9],[2,8],[2,9],[3,6]]

def reward(state,next_state,step):
    block_hit = False
    next_state_location = [[next_state[0][0],next_state[0][1]]]
    
    agent_reward = 0.08#5.5 - math.sqrt((state[0][0] - target_state[0])**2 + (state[0][1] - target_state[1])**2 ) # 2 - (step * 0.001)
    done = False
    
    if next_state[0][0] < 0 or next_state[0][1] < 0 or next_state[0][0] > (grid-1) or next_state[0][1] > (grid-1)  :
        agent_reward = 0 #+ (step * 0.05)
        block_hit = True

    for i in blockages:
        if (np.array(next_state_location[0]) == np.array(i)).all() == True:
            agent_reward = 0 #+ (step * 0.05)
            block_hit = True
            break
    
    if (np.array(next_state[0]) == np.array(target_state)).all():
        agent_reward = 1  #- (step * 0.09)
        done = True

    return agent_reward , done , block_hit

def agent_6_action(state, action):

    if action == 0:
        next_state = [[state[0][0]-1, state[0][1]+1,7]]
    if action == 1:
        next_state = [[state[0][0]+1,  state[0][1] + 1,9]]
    if action == 2 :
        next_state = [[state[0][0] +1 , state[0][1] -1, 7]]
    if action == 3 :
        next_state = [[state[0][0]-1 , state[0][1] - 1,9]]
    if action == 4 :
        next_state = [[state[0][0]-1 , state[0][1] ,6]]
    if action == 5 :
        next_state = [[state[0][0]+1 , state[0][1] ,6]]
    
    next_state = np.reshape(next_state, [1, 3])

    return next_state

def agent_7_action(state, action):

    if action == 0:
        next_state = [[state[0][0]+1, state[0][1]+1,8]]
    if action == 1:
        next_state = [[state[0][0]+1,  state[0][1] - 1,6]]
    if action == 2 :
        next_state = [[state[0][0] -1 , state[0][1] -1, 8]]
    if action == 3 :
        next_state = [[state[0][0]-1 , state[0][1] + 1,6]]
    if action == 4 :
        next_state = [[state[0][0] , state[0][1]+1 ,7]]
    if action == 5 :
        next_state = [[state[0][0] , state[0][1] -1 ,7]]
    
    next_state = np.reshape(next_state, [1, 3])

    return next_state

def agent_8_action(state, action):

    if action == 0:
        next_state = [[state[0][0]+1, state[0][1]-1,9]]
    if action == 1:
        next_state = [[state[0][0]-1,  state[0][1] - 1,7]]
    if action == 2 :
        next_state = [[state[0][0] -1 , state[0][1] +1, 9]]
    if action == 3 :
        next_state = [[state[0][0]+1 , state[0][1] + 1,7]]
    if action == 4 :
        next_state = [[state[0][0]+1 , state[0][1] ,8]]
    if action == 5 :
        next_state = [[state[0][0]-1 , state[0][1] ,8]]
    
    next_state = np.reshape(next_state, [1, 3])

    return next_state

def agent_9_action(state, action):

    if action == 0:
        next_state = [[state[0][0]-1, state[0][1]-1,6]]
    if action == 1:
        next_state = [[state[0][0]-1,  state[0][1] + 1,8]]
    if action == 2 :
        next_state = [[state[0][0] +1 , state[0][1] +1, 6]]
    if action == 3 :
        next_state = [[state[0][0]+1 , state[0][1] - 1,8]]
    if action == 4 :
        next_state = [[state[0][0] , state[0][1]-1 ,9]]
    if action == 5 :
        next_state = [[state[0][0] , state[0][1]+1 ,9]]
    
    next_state = np.reshape(next_state, [1, 3])

    return next_state

def step(action, state, step):
    next_state = state
    

    if state[0][2] == 6:
        next_state = agent_6_action(state,action)
    if state[0][2] == 7:
        next_state = agent_7_action(state,action)
    if state[0][2] == 8:
        next_state = agent_8_action(state,action)
    if state[0][2] == 9:
        next_state = agent_9_action(state,action) 

    '''
    if action == 0 :
        if state[0][0] != 0:
            next_state = [[state[0][0]-1, state[0][1]]]
            next_state = np.reshape(next_state, [1, 2])
    
    if action == 1 :
        if state[0][1] != 9:
            next_state = [[state[0][0],  state[0][1] + 1]]
            next_state = np.reshape(next_state, [1, 2])

    if action == 2 :
        if state[0][0] != 9:
            next_state = [[state[0][0] +1 , state[0][1] ]]
            next_state = np.reshape(next_state, [1, 2])

    if action == 3 :
        if state[0][1] != 0:
            next_state = [[state[0][0] , state[0][1] - 1]]
            next_state = np.reshape(next_state, [1, 2])
    
    for i in blockages:
        if (np.array(next_state[0]) == np.array(i)).all() == True:
            next_state = state
            break
    '''
    
    ag_reward, done ,block_hit = reward(state,next_state,step)
    

    return next_state, ag_reward, done, block_hit
    

        

    
        