import numpy as np

class MobileEdgeComputingEnv:
    def __init__(self, num_states=5, num_actions=2):
        """
        Initialize the environment with states (edge node load levels) and actions.
        :param num_states: Number of different load levels (e.g., 0 to 4).
        :param num_actions: Number of actions (offload or process locally).
        """
        self.num_states = num_states  # Possible load levels (0: no load, 4: full load)
        self.num_actions = num_actions  # Offload (0) or process locally (1)
        self.state = self._get_initial_state()  # Initial state (random load)
    
    def _get_initial_state(self):
        """Randomly initialize the state (edge load level)."""
        return np.random.randint(0, self.num_states)
    
    def reset(self):
        """Reset the environment to an initial state."""
        self.state = self._get_initial_state()
        return self.state
    
    def step(self, action):
        """
        Take an action in the environment and return the next state, reward, and done flag.
        :param action: The action chosen by the agent (offload or process locally).
        :return: next_state, reward
        """
        state = self.state
        reward = self._get_reward(state, action)  # Calculate reward based on current state and action
        next_state = self._get_next_state()  # Move to the next state (random new load)
        self.state = next_state
        return next_state, reward
    
    def _get_next_state(self):
        """Simulate the next state by generating a new random load level."""
        return np.random.randint(0, self.num_states)
    
    def _get_reward(self, state, action):
        """
        Determine the reward based on the current state (load level) and action.
        :param state: The current load level at the edge node.
        :param action: The action taken by the agent (0: offload, 1: process locally).
        :return: Reward value (positive for successful task completion, negative for failure).
        """
        if action == 0:  # Offload task
            if state < 3:  # Low load, offloading successful
                return 10  # Positive reward for low delay
            else:
                return -10  # Penalty for offloading when the node is overloaded
        else:  # Process locally
            return 5  # Moderate reward for processing locally (no risk of task drop)

# Example usage (for testing, you can remove it later):
if __name__ == "__main__":
    env = MobileEdgeComputingEnv()
    state = env.reset()
    print(f"Initial state (edge node load): {state}")
    action = 0  # Offload
    next_state, reward = env.step(action)
    print(f"Next state: {next_state}, Reward: {reward}")
