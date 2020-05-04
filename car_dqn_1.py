import random
import os
import numpy as np
#from numba import jit, cuda 
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import action_result_1
import matplotlib.pyplot as plt
import timeit
from variables import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start = timeit.default_timer()
state2 = np.reshape([0,1,7], [1, 3])
#q_now = 0
#q_previous = 0
state2_ac0 = []
state2_ac1 = []
state2_ac2 = []
state2_ac3 = []
state2_ac4 = []
state2_ac5 = []

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9   # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01
        self.model = self._build_model()

    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20 , activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='MSE',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_epsilon(self, t):
        return self.epsilon_min + (self.epsilon - self.epsilon_min)*math.exp(-self.epsilon_decay * t)
        #return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
    
    def act(self, state ):
        if np.random.rand() <= self.epsilon:
        #if np.random.rand() <= change_epsilon:
            #print("randomly calculating")
            return random.randrange(self.action_size)
        #print("predicting action by neural network")
        act_values = self.model.predict(state)
        #print("calculating neural networ")
        return np.argmax(act_values[0])
    '''
    def act(self, state , change_epsilon):
        if np.random.rand() <= self.epsilon:
        #if np.random.rand() <= change_epsilon:
            #print("randomly calculating")
            return random.randrange(self.action_size)
        #print("predicting action by neural network")
        act_values = self.model.predict(state)
        #print("calculating neural networ")
        return np.argmax(act_values[0])  # returns action
    '''
    def state_action_graph(self,state_nn_value):
        state2_ac0.append(state_nn_value[0][0])
        state2_ac1.append(state_nn_value[0][1])
        state2_ac2.append(state_nn_value[0][2])
        state2_ac3.append(state_nn_value[0][3])
        state2_ac4.append(state_nn_value[0][4])
        state2_ac5.append(state_nn_value[0][5])


    def calculate_loss(self,state,next_state,reward):
        q_previous = np.amax(self.model.predict(state)[0])
        q_now = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

        return (q_now - q_previous)**2
    #@jit 
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        #print(" ")
        #print("minibatch: ",minibatch)
        #print(" ")
        #print("minibatch : ",minibatch)
        for state, action, reward, next_state, done in minibatch:
            #print(next_state)
            target = reward
            
            if not done:
                #print("amax: ",np.amax(self.model.predict(next_state)[0]))
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
                #print("target :", target)
            target_f = self.model.predict(state)
            #print("target_f before :", target_f)
            target_f[0][action] = target
            #print("target f after:", target_f)
            #print("before1 :",self.model.predict(state2))
            #print("before2 :",self.model.predict(state2))
            print("haha", state)
            print("haha", target_f)
            self.model.fit(state, target_f, epochs=10, verbose=0)
            #print("after fit state :",self.model.predict(state))
            #print(" ")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
   


    action_direction = {
        'front_right' : 0,
        'back_right' : 1, 
        'back_left' : 2,
        'front_left' : 3,
        'front' : 4,
        'back' : 5
    }

    agent_orientation = {
        'N' : 6,
        'E' : 7, 
        'S' : 8,
        'W' : 9
    }
   

    #print("hello")
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    
    batch_size = 60
    blockages = [] 
    cum_reward = [] # cumulative average reward
    success_ep = [] # success episodes

    north_boundary =[]
    south_boundary = []
    east_boundary = []
    west_boundary = []
    success_per_n_ep = []
    epsilon_change = []
    cal_loss = []
    n_i = 0
    w_i = 0
    e_i = 0
    s_i = 0
    ep_i = 0
    count_c_loss = 0

    for e in range(EPISODES):
        state = [np.random.randint(0,high=5),np.random.randint(0,high=5),np.random.randint(6,high=10)]
        #state = initial_state
        state = np.reshape(state, [1, state_size])
        
        
        #print("state :", state[0][1])
        ep_reward = []
        path = []
        print("\nepisode no : ",e)
        print("---------")
        print("initial state:", state)
        state_nn_value = agent.model.predict(state2)
        agent.state_action_graph(state_nn_value)
        for steps in range(steps_in_ep):
            # env.render()
            #action = 3
            #print("next state :" , state)
            #action = agent.act(state, agent.get_epsilon(e))
            action = agent.act(state)
            #print("action", action)
            
            next_state, reward, done, block_hit = action_result_1.step(action, state,steps)  # have to be made for the action values (next state, reward, done, _)
            #print("next state :" , next_state)
            #print("reward :",reward)
            #print("done :" , done)
            
            
            path.append(next_state[0])
            ep_reward.append(reward)  
            
            next_state = np.reshape(next_state, [1, state_size])
            #print("state: ",state)
            agent.memorize(state, action, reward, next_state, done)
            #print("memory : ", agent.memory)
            count_c_loss = count_c_loss +1
            cal_loss.append([count_c_loss , agent.calculate_loss(state,next_state,reward)])
            state = next_state
            if block_hit == True:
                print("exit due to block hit  ")
                
                
                break
            if done == True :
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, steps, agent.epsilon))
                #print(path)
                success_ep.append([e,steps])
                ep_i = ep_i +1
                break
            #agent.replay(batch_size)
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        
        if (path[len(path)-1][0] < 0):
            n_i = n_i + 1
            print("north boundary hit")
            
        if (path[len(path)-1][1] < 0):
            w_i = w_i + 1
            print("west boundary hit")

        if (path[len(path)-1][1] > (grid-1)):
            e_i = e_i + 1
            print("east boundary hit")

        if (path[len(path)-1][0] > (grid-1)):
            s_i = s_i + 1
            print("south boundary hit")

        if ((e+1)%stats_per) == 0 :
            north_boundary.append([n_i,e+1])
            n_i = 0
            south_boundary.append([s_i,e+1])
            s_i = 0
            east_boundary.append([e_i,e+1])
            e_i = 0
            west_boundary.append([w_i,e+1])
            w_i = 0
            success_per_n_ep.append([ep_i,e+1])
            ep_i = 0
        
        print("\npath :",path)
        print("epsilon :", agent.epsilon)
        epsilon_change.append([e,agent.epsilon])
        #print("reward :", ep_reward)
        #cum_reward.append(sum(ep_reward) )
        cum_reward.append([e,sum(ep_reward) / len(ep_reward)])
        #if e % 10 == 0:
        #    agent.save("./save/cartpole-dqn.h5")


count = []
for i in range(len(state2_ac0)):
    count.append(i)


#print("weights :", agent.)
print("\n[succeeded in episodes, no. of steps] :", success_ep)   
print("")




plt.figure(1)
plt.plot((np.array(north_boundary))[:,1],(np.array(north_boundary))[:,0],label = 'north boundary')
plt.plot((np.array(south_boundary))[:,1],(np.array(south_boundary))[:,0],label = 'south boundary')
plt.plot((np.array(east_boundary))[:,1],(np.array(east_boundary))[:,0],label ='east boundary')
plt.plot((np.array(west_boundary))[:,1],(np.array(west_boundary))[:,0], label = 'west boundary')
plt.xlabel("Hit per " + str(stats_per) +" episodes")
plt.ylabel("Number of hits")
plt.title("Learning of boundaries")
plt.legend()

plt.figure(2)
plt.plot((np.array(success_per_n_ep))[:,1],(np.array(success_per_n_ep))[:,0],label="success rate")
plt.xlabel("Success per " + str(stats_per) +" episodes")
plt.ylabel("Number of success")
plt.title("success rate")
plt.legend()
'''
plt.plot(count,state2_ac0, label = 'action 0')
plt.plot(count,state2_ac1, label = 'action 1')
plt.plot(count,state2_ac2, label = 'action 2')
plt.plot(count,state2_ac3, label = 'action 3')   
plt.plot(count,state2_ac4, label = 'action 4')
plt.plot(count,state2_ac5, label = 'action 5')
plt.xlabel("number")
plt.ylabel("action values")
'''
plt.figure(3)
plt.plot(np.array(cum_reward)[:,0], np.array(cum_reward)[:,1],label='cumulative reward')
plt.plot(np.array(epsilon_change)[:,0],np.array(epsilon_change)[:,1],label='epsilon value')
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.title("Reward per episode")
plt.legend()

plt.figure(4)
plt.plot(np.array(cal_loss)[:,0],np.array(cal_loss)[:,1],label='loss')
plt.xlabel("count")
plt.ylabel("loss")
plt.title("NN loss")
plt.legend()

stop = timeit.default_timer()
print('\nTime taken: ', stop - start) 

plt.show()


