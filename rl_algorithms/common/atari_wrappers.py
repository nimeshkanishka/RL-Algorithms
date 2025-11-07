import numpy as np
import gymnasium as gym

class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        action_meanings = env.unwrapped.get_action_meanings()
        # Environment must have at least 3 actions as we will perform a sequence of
        # actions ('FIRE' and another action) to get out of the waiting state
        if len(action_meanings) < 3:
            raise ValueError("Environment must have at least 3 actions.")
        # Second action of the environment must be 'FIRE'
        if action_meanings[1] != "FIRE":
            raise ValueError("Second action of the environment must be 'FIRE'.")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Perform 'FIRE' action
        observation, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)

        # Perform another action (action 2)
        # Some Atari games require a sequence of actions (like 'FIRE' and a second button press)
        # to get out of the waiting state
        observation, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)

        return observation, info
    
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        action_meanings = env.unwrapped.get_action_meanings()
        # First action of the environment must be 'NOOP'
        if len(action_meanings) == 0 or action_meanings[0] != "NOOP":
            raise ValueError("First action of the environment must be 'NOOP'.")

        self._lives = 0
        # Ensure environment will always be reset initially
        self._was_real_done = True

    def reset(self, **kwargs):
        if self._was_real_done:
            observation, info = self.env.reset(**kwargs)

        else:
            # No-op step to advance from lost life state
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)

        # Update life count
        self._lives = self.env.unwrapped.ale.lives()

        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # _was_real_done = True only when all lives have been lost or game has been truncated
        self._was_real_done = terminated or truncated

        # terminated = True when a life has been lost, even if there are more lives left
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives

        return observation, reward, terminated, truncated, info
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)

        if skip < 1:
            raise ValueError("skip must be at least 1.")
        
        self._skip = skip
        self._observation_buffer = np.zeros((2, *self.env.observation_space.shape),
                                            dtype=self.env.observation_space.dtype)

    def step(self, action: int):
        total_reward = 0.0

        for i in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)

            if i == self._skip - 2:
                self._observation_buffer[0] = observation
            elif i == self._skip - 1:
                self._observation_buffer[1] = observation

            total_reward += float(reward)

            if terminated or truncated:
                break
        
        max_frame = self._observation_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
    
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)

        action_meanings = env.unwrapped.get_action_meanings()
        # First action of the environment must be 'NOOP'
        if len(action_meanings) == 0 or action_meanings[0] != "NOOP":
            raise ValueError("First action of the environment must be 'NOOP'.")

        if noop_max < 1:
            raise ValueError("noop_max must be at least 1.")
        self._noop_max = noop_max

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Perform 'NOOP' action
        num_noops = self.unwrapped.np_random.integers(1, self._noop_max + 1)
        for _ in range(num_noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)

        return observation, info