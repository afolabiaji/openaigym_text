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
episodes = 50000
learning_rate = 0.1
discount = 0.9


# Exploration Parameters
epsilon = 1.0           
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01 
step_limit = 50
goal_count = 0

# 4: Training
for episode in range(episodes):
    done = False
    state= env.reset()
    step_count = 0
    while done == False:
        exp_exp_tradeoff = np.random.uniform(0,1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        env.render()
        if new_state == 12 or new_state == 5 or new_state == 7 or new_state == 11:
            reward = -10
        if step_count > step_limit:
            reward = -1
            done = True
        if new_state == 15:
            goal_count += 1

        print(goal_count)
        q_table[state, action] = q_table[state, action] + learning_rate*(reward + discount*(np.max(q_table[new_state, :])) - q_table[state, action])

        state = new_state

        step_count += 1

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

#5: Algorithm now plays frozen lake
total_reward = 0


for episode in range(10000):
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

print(f"Q Table Accuracy = {total_reward/10000}")

#I've tweaked hyperparameters in training. It seems algorithm learns quickest when punished harsly for falling into hole and punished for taking too long to move
#Punishment for taking long balances out punishment applied to surronding tiles of holes (these take a hit to q value during exploration)
#Balance between reward/punishment/max_steps seems to dictacte how quickly algorithm learns