import random
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model import DQN, ReplayBuffer, preprocess_frame, train_dqn

gym.register_envs(ale_py)

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
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

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = preprocess_frame(state)
    state = np.stack([state] * 4, axis=0)  # Stack 4 frames
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
        next_state = np.stack([next_state, state[0], state[1], state[2]], axis=0)
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
        print(f"Model saved at episode {episode + 1}")

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
