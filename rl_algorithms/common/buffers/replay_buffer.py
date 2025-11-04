import numpy as np
from gymnasium.spaces import Space, Box, Discrete

class ReplayBuffer:
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        buffer_size: int = 1_000_000,
        batch_size: int = 32,
        seed: int | None = None
    ):
        # Observation buffer
        # Only Box observation spaces are supported
        if isinstance(observation_space, Box):
            self.observations = np.zeros((buffer_size, *observation_space.shape),
                                         dtype=observation_space.dtype)
        else:
            raise NotImplementedError("Observation space type not supported.")

        # Action buffer
        # Only Box and Discrete action spaces are supported
        if isinstance(action_space, Box):
            self.actions = np.zeros((buffer_size, *action_space.shape),
                                    dtype=action_space.dtype)
        elif isinstance(action_space, Discrete):
            self.actions = np.zeros((buffer_size,),
                                    dtype=action_space.dtype)
        else:
            raise NotImplementedError("Action space type not supported.")

        # Reward buffer
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)

        # Done buffer
        self.dones = np.zeros((buffer_size,), dtype=np.bool_)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory_index = 0
        self.is_memory_full = False

        # Numpy random generator
        self.np_random = np.random.default_rng(seed=seed)

    def store(self, observation, action, reward, next_observation, done):
        self.observations[self.memory_index] = observation
        self.actions[self.memory_index] = action
        self.rewards[self.memory_index] = reward
        self.observations[(self.memory_index + 1) % self.buffer_size] = next_observation
        self.dones[self.memory_index] = done

        self.memory_index += 1
        if self.memory_index >= self.buffer_size:
            self.memory_index = 0
            self.is_memory_full = True

    def sample(self):
        if self.is_memory_full:
            # We cannot take the transition at memory_index because observation and other data will be from different timesteps
            batch_indices = self.np_random.choice(
                (np.arange(start=1, stop=self.buffer_size) + self.memory_index) % self.buffer_size,
                size=self.batch_size,
                replace=False
            )
        else:
            if self.memory_index < self.batch_size:
                raise ValueError("Not enough transitions in the buffer to sample a batch.")
            else:
                batch_indices = self.np_random.choice(self.memory_index, size=self.batch_size, replace=False)

        observations = self.observations[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        next_observations = self.observations[(batch_indices + 1) % self.buffer_size]
        dones = self.dones[batch_indices]

        return observations, actions, rewards, next_observations, dones