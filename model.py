from collections import deque
import random

from PIL import Image

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def preprocess_frame(frame):
    """Preprocess the frame: grayscale, resize, normalize."""
    # frame = np.mean(frame, axis=2)  # Convert to grayscale
    # frame = frame[35:195]  # Crop the frame to focus on the game area
    # frame = frame[::2, ::2]  # Downsample to 80x80
    # frame = np.array(Image.fromarray(frame).resize((84, 84)))
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame


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
