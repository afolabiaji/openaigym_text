import numpy as np
import gym
import random

# Step 1: Create Gym Environment
env = gym.make("FrozenLake-v0", map_name="4x4")
env.render()

# Step 2: Create and Iniitialise Q-Table
action_size = env.action_space.n
print("Action Size", action_size)

state_size = env.observation_space.n
print("State Size", state_size)

q_table = np.zeros((state_size, action_size))

# 3: Hyperparameters
episodes = 10**7
learning_rate = 0.1
discount = 0.9


# Exploration Parameters
epsilon = 1.0           
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01 

# 4: Training
for episode in range(episodes):
    done = False
    state= env.reset()
    while done == False:
        exp_exp_tradeoff = np.random.uniform(0,1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        # print(reward, done)
        q_table[state, action] = q_table[state, action] + learning_rate*(reward + discount*(np.max(q_table[new_state, :])) - q_table[state, action])

        state = new_state

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

#5: Algorithm now plays frozen lake
total_reward = 0


for episode in range(5):
    state = env.reset()
    done = False
    while done == False:
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
        state = new_state
        total_reward += reward

print(f"Q Table final reward = {total_reward/5}")
