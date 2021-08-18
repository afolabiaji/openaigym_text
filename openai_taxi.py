import numpy as np
import gym
import random

# Step 1: Create Gym Environment
env = gym.make("Taxi-v3")
env.render()

# Step 2: Create and Iniitialise Q-Table
action_size = env.action_space.n
print("Action Size", action_size)

state_size = env.observation_space.n
print("State Size", state_size)

q_table = np.zeros((state_size, action_size))

# Step 3: Create Hyperparameters
total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618

# Exploration Parameters
epsilon = 1.0           
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01 

# Step 4: The Q Learning Algorithm

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :] - q_table[state, action]))

        state = new_state

        if done == True:
            break

    episode += 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# Step 5: Use Q-Table to play Taxi

env.reset()
rewards = []

for epsiode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("***************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state,:])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))