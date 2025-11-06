from typing import Optional, Any
import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from rl_algorithms.common.buffers.replay_buffer import ReplayBuffer
from rl_algorithms.DQN.policies import MLPPolicy, CNNPolicy

class DQN:
    def __init__(
        self,
        policy: str,
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: str = "auto"
    ):
        # Validate policy
        valid_policies = ["MLPPolicy", "CNNPolicy"]
        if policy not in valid_policies:
            raise ValueError(f"Unknown policy '{policy}'. Must be one of {valid_policies}.")            
        
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.policy_kwargs = policy_kwargs or {}
        self.seed = seed

        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)

        self.np_random = np.random.default_rng(seed=seed)

        self.env = None
        self.env_observation_space = None
        self.env_action_space = None
        self.replay_buffer = None
        self.q_network = None
        self.q_network_optimizer = None
        self.target_q_network = None

        if env is not None:
            # Only environments with Box observation spaces are supported
            if not isinstance(env.observation_space, Box):
                raise NotImplementedError("Only Box observation spaces are supported.")
            # Only environments with Discrete action spaces are supported
            if not isinstance(env.action_space, Discrete):
                raise NotImplementedError("Only Discrete action spaces are supported.")
            self.env = env
            self.env_observation_space = env.observation_space
            self.env_action_space = env.action_space

            self._init_networks()

    @classmethod
    def load(
        cls,
        path: str,
        env: gym.Env | None = None,
        device: str = "auto"
    ):
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Extract hyperparameters
        policy = checkpoint["policy"]
        learning_rate = checkpoint["learning_rate"]
        buffer_size = checkpoint["buffer_size"]
        learning_starts = checkpoint["learning_starts"]
        batch_size = checkpoint["batch_size"]
        gamma = checkpoint["gamma"]
        train_freq = checkpoint["train_freq"]
        gradient_steps = checkpoint["gradient_steps"]
        target_update_interval = checkpoint["target_update_interval"]
        exploration_fraction = checkpoint["exploration_fraction"]
        exploration_initial_eps = checkpoint["exploration_initial_eps"]
        exploration_final_eps = checkpoint["exploration_final_eps"]
        policy_kwargs = checkpoint["policy_kwargs"]
        seed = checkpoint["seed"]
        env_observation_space = checkpoint["env_observation_space"]
        env_action_space = checkpoint["env_action_space"]

        if env is not None:
            if env.observation_space != env_observation_space or env.action_space != env_action_space:
                raise ValueError(
                    "Environment spaces do not match the ones used during training. "
                    "Please provide an environment with matching observation and action spaces."
                )

        # Create new DQN instance
        dqn = cls(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            gamma,
            train_freq,
            gradient_steps,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            policy_kwargs,
            seed,
            device
        )

        # If an environment is not provided, set attributes env_observation_space and env_action_space
        # of the DQN instance and call _init_networks function to initialize networks and replay buffer
        if env is None:
            dqn.env_observation_space = env_observation_space
            dqn.env_action_space = env_action_space

            dqn._init_networks()

        # Restore network parameters
        dqn.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        dqn.target_q_network.load_state_dict(checkpoint["q_network_state_dict"])

        return dqn

    def learn(
        self,
        total_timesteps: int,
        eval_env: gym.Env | None = None,
        num_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        best_model_save_path: str | None = None,
        verbose: bool = True
    ):
        # Check if self.env has been set
        if self.env is None:
            raise ValueError(
                "Environment has not been set. Please initialize DQN with an environment (env != None)."
            )
        
        best_reward = float("-inf")

        # Initialize sequence
        observation, _ = self.env.reset(seed=self.seed)

        for t in range(total_timesteps):
            epsilon = self._calculate_epsilon(t, total_timesteps)

            # Select action
            action = self.select_action(observation, epsilon)            

            # Execute action and observe reward and next state
            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            # Store transition
            self.replay_buffer.store(observation, action, reward, next_observation, terminated or truncated)

            if terminated or truncated:
                observation, _ = self.env.reset()
            else:
                observation = next_observation

            if t >= self.learning_starts and (t + 1) % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._train_step()

            # Every C steps reset target action-value function with same weights as action-value function
            if (t + 1) % self.target_update_interval == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

            if eval_env is not None:
                if (t + 1) % eval_freq == 0:
                    episode_length, episode_reward = self._evaluate(eval_env, num_eval_episodes)

                    if verbose:
                        print(f"\nEvaluation: Timesteps = {t + 1}")
                        print(f"Episode length = {episode_length['mean']:.2f} +/- {episode_length['std']:.2f}")
                        print(f"Episode reward = {episode_reward['mean']:.2f} +/- {episode_reward['std']:.2f}")

                    if episode_reward["mean"] > best_reward:
                        best_reward = episode_reward["mean"]

                        if verbose:
                            print("New best mean reward!")

                        if best_model_save_path is not None:
                            self.save(path=f"{best_model_save_path}/best_model.pth")

                            if verbose:
                                print("Saved model checkpoint")

    def save(
        self,
        path: str
    ):
        torch.save({
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "policy_kwargs": self.policy_kwargs,
            "seed": self.seed,
            "env_observation_space": self.env_observation_space,
            "env_action_space": self.env_action_space,
            "q_network_state_dict": self.q_network.state_dict()
        }, path)

    def select_action(
        self,
        observation: np.ndarray,
        epsilon: float = 0
    ):
        # With probability ε select a random action
        if self.np_random.random() < epsilon:
            return self.env.action_space.sample()

        # Otherwise select action with highest Q-value
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            q_values = self.q_network(observation_tensor.unsqueeze(dim=0).to(self.device)).squeeze()
            action = torch.argmax(q_values).item()
        return action
    
    def _init_networks(self):
        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(
            observation_space=self.env_observation_space,
            action_space=self.env_action_space,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            seed=self.seed
        )

        match self.policy:
            case "MLPPolicy":
                network = MLPPolicy
            case "CNNPolicy":
                network = CNNPolicy

        # Initialize action-value function
        self.q_network = network(
            input_dim=self.env_observation_space.shape[0],
            output_dim=self.env_action_space.n,
            **self.policy_kwargs
        ).to(self.device)
        self.q_network_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        # Initialize target action-value function with same weights as action-value function
        self.target_q_network = network(
            input_dim=self.env_observation_space.shape[0],
            output_dim=self.env_action_space.n,
            **self.policy_kwargs
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _calculate_epsilon(self, t, T):
        decay_steps = T * self.exploration_fraction
        epsilon = self.exploration_initial_eps - \
            (self.exploration_initial_eps - self.exploration_final_eps) / decay_steps * t
        return max(self.exploration_final_eps, epsilon)
    
    def _train_step(self):
        # Sample random minibatch of transitions
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample()

        observations_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_observations_tensor = torch.tensor(next_observations, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # max returns both values and indices, we only need values
            max_next_q_values = self.target_q_network(next_observations_tensor).max(dim=1).values

            # Set y_j = r_j                                     if episode terminated at step j+1
            #           r_j + γ * max_a' Q_target(s_{j+1}, a')  otherwise
            y = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Perform a gradient descent step on (y_j - Q(s_j, a_j))^2
        self.q_network_optimizer.zero_grad()
        q_values = self.q_network(observations_tensor).gather(dim=1, index=actions_tensor.unsqueeze(dim=1)).squeeze(dim=1)
        loss = torch.nn.functional.huber_loss(q_values, y)        
        loss.backward()
        self.q_network_optimizer.step()

    def _evaluate(self, env, num_episodes):
        episode_lengths = []
        episode_rewards = []

        for _ in range(num_episodes):
            observation, _ = env.reset()
            episode_length = 0
            episode_reward = 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.select_action(observation)

                observation, reward, terminated, truncated, _ = env.step(action)

                episode_length += 1
                episode_reward += reward

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)

        length_data = {
            "mean": np.mean(episode_lengths),
            "std": np.std(episode_lengths)
        }
        reward_data = {
            "mean": np.mean(episode_rewards),
            "std": np.std(episode_rewards)
        }
        return length_data, reward_data