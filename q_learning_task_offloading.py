import numpy as np
#This is used for numerical operations, particularly to create and manipulate the Q-table (a matrix of state-action values).
import random
#This is used to introduce randomness when selecting actions
from mobile_edge_computing_env import MobileEdgeComputingEnv

# Initialize the environment
env = MobileEdgeComputingEnv()


#q_table: This is a table (a matrix) where each row represents a state (edge node load level) and each column represents an action (offload or process locally). It stores the expected reward for each state-action pair. Initially, it’s filled with zeros because the agent hasn’t learned anything yet.
# Q-learning parameters
q_table = np.zeros((env.num_states, env.num_actions))  # Q-table for state-action values
#Controls how much the new information overrides the old information when updating the Q-table.
#A smaller value means the agent learns more slowly.
alpha = 0.1  # Learning rate

#Determines how much future rewards influence the current action. A value close to 1 means future rewards are very important.
#This encourages the agent to think long-term rather than just focusing on immediate rewards.
#
gamma = 0.9  # Discount factor for future rewards
#This controls the balance between exploration and exploitation.
#With probability epsilon, the agent will explore and take a random action to try something new.
#With probability 1 - epsilon, the agent will exploit what it has already learned by choosing the best-known action (based on the Q-table).
epsilon = 0.1  # Exploration rate

# Q-learning loop for training
for episode in range(1000):  # Train over 1000 episodes
    state = env.reset()  # Reset the environment at the beginning of each episode
    for step in range(10):  # Simulate multiple steps within an episode
        # Choose action (explore or exploit)
        if random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.num_actions)  # Explore random action
        else:
            action = np.argmax(q_table[state, :])  # Exploit the best action

        # Take the action in the environment
        next_state, reward = env.step(action)

        # Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state  # Move to the next state

print("Trained Q-table:\n", q_table)
