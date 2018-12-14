
import gym

env = gym.make('Taxi-v2')

state = env.reset()

env.render()

print(state)

env.env.s = 282

env.render()

print('Different states'+ str(env.observation_space.n))
print('Different states'+ str(env.action_space.n))


env.step(3)
env.render()

env.step(3)
env.render()

env.step(3)
env.render()

env.step(3)
env.render()

env.step(0)
env.render()

env.step(0)
env.render()

env.step(5)
env.render()

'''
import gym
env = gym.make("Taxi-v2")
observation = env.reset()
for _ in range(10):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)



import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
  
'''
