# RL Algorithms: Reinforcement Learning Implementations

RL Algorithms contains my implementations of popular reinforcement learning algorithms in PyTorch.

## RL Algorithms

1. DQN

## Installation

1. Clone the repository

```bash
git clone https://github.com/nimeshkanishka/RL-Algorithms.git
```

2. Install the package

```bash
cd RL-Algorithms
pip install .
```

## Usage

Here is a quick example of how to train and run DQN on the CartPole environment:

```python
import gymnasium as gym
from rl_algorithms import DQN

# Create training environment
env = gym.make("CartPole-v1")

# Instantiate DQN agent and train
model = DQN(env)
model.learn(total_timesteps=50_000)

# Watch the trained agent play a game
test_env = gym.make("CartPole-v1", render_mode="human")
observation, _ = test_env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = model.select_action(observation)
    observation, reward, terminated, truncated, _ = test_env.step(action)
```

### Training, Evaluation, Saving and Loading

The following example demonstrates how to train, evaluate, save and load a DQN model on the Lunar Lander environment:

```python
import gymnasium as gym
from rl_algorithms import DQN

# Create training environment
env = gym.make("LunarLander-v3")
# Create separate environment for evaluation
eval_env = gym.make("LunarLander-v3")

# Instantiate DQN agent
model = DQN(env)
# Train the agent
# Evaluate the agent every 10k timesteps and save the best version of the agent
model.learn(total_timesteps=500_000, eval_env=eval_env, eval_freq=10_000, best_model_save_path=".")
# Save the final version of the agent
model.save("final_model.pth")
# Delete trained agent (to demonstrate loading)
del model

# Load the trained agent
model = DQN.load("final_model.pth", env=env)

# Watch the trained agent play a game
test_env = gym.make("LunarLander-v3", render_mode="human")
observation, _ = test_env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = model.select_action(observation)
    observation, reward, terminated, truncated, _ = test_env.step(action)
```