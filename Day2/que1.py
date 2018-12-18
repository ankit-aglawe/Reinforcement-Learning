import gym
import numpy as np

env = gym.make('Taxi-v2')
state = env.reset()
env.render()

#---------------------------------------------------


#env.render()


reward = 0
cnt = 1

while reward!=20:
    action = env.action_space.sample()
    state,reward,done,info = env.step(action)
    cnt = cnt+1

print('Needed '+str(cnt)+ ' moves to reach final state')

#-------------------------------------------------------

Q = np.zeros([500,6])

for episode in range(1,1000):
    state = env.reset()
    #env.render()
    gamma =0.801
    cnt = 1
    reward =0
    while reward!=20:
        action = np.argmax(Q[state])
        state2,reward,done,info = env.step(action)
        Q[state,action]=reward+gamma*np.max(Q[state2])
        state = state2
        cnt = cnt + 1

        if episode%50==0:
            print('Needed '+str(cnt)+' move to reach final stage during episode '+str(episode))
    
    


#------------------------------------------------------------

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    
    
