import numpy as np
from mobile_edge_computing_env import MobileEdgeComputingEnv
from q_learning_task_offloading import q_table  # Import the trained Q-table

# Initialize the environment
env = MobileEdgeComputingEnv()

# Testing parameters
num_test_episodes = 100
total_rewards = 0

# Test the trained Q-learning model
for episode in range(num_test_episodes):
    state = env.reset()  # Start at a random state (edge load level)
    for step in range(10):  # Test over multiple steps
        action = np.argmax(q_table[state, :])  # Always exploit the best action
        next_state, reward = env.step(action)
        total_rewards += reward
        state = next_state  # Move to the next state

print(f"Average reward per test episode: {total_rewards / num_test_episodes}")
