from datetime import datetime
from pathlib import Path
import random
import logging

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import DQN, ReplayBuffer, preprocess_frame

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(__file__).resolve().parent / "model_run_2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Logfiles
start_datetime = datetime.now().strftime("%Y%m%d_%H%M")
log_file = LOG_DIR / f"logfile_{start_datetime}.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = AtariPreprocessing(
    env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30
)
env = FrameStackObservation(env, stack_size=4)  # Stack 4 frames
action_dim = env.action_space.n

# Initialize models and replay buffer
model = DQN(action_dim)
target_model = DQN(action_dim)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=0.0001)
replay_buffer = ReplayBuffer(capacity=10000)

# Training parameters
NUM_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
UPDATE_TARGET_EVERY = 1000


def train_dqn(
    env,
    model,
    target_model,
    optimizer,
    replay_buffer,
    batch_size=32,
    gamma=0.99,
):
    if len(replay_buffer) < batch_size:
        return

    # Sample a mini-batch from the replay buffer
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones)

    # Compute Q-values and target Q-values
    current_q_values = model(states).gather(1, actions.unsqueeze(1))
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss and update the model
    loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = preprocess_frame(state)
    # state = np.stack([state] * 4, axis=0)  # Stack 4 frames
    done = False
    total_reward = 0

    while not done:
        # Select action using epsilon-greedy policy
        epsilon = max(0.1, 1.0 - episode / 500)  # Decay epsilon
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()

        # Take action and observe next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_frame(next_state)
        # next_state = np.stack([next_state, state[0], state[1], state[2]], axis=0)
        done = terminated or truncated

        # Store experience in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train the model
        train_dqn(env, model, target_model, optimizer, replay_buffer, BATCH_SIZE, GAMMA)

    # Update target network
    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.load_state_dict(model.state_dict())

    # Save the model periodically
    if (episode + 1) % 100 == 0:
        torch.save(
            model.state_dict(), MODEL_DIR / f"pong_dqn_episode_{episode + 1}.pth"
        )
        logging.info(f"Model saved at episode {episode + 1}")

    logging.info(f"Episode {episode + 1}, Total Reward: {total_reward}")
