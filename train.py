from datetime import datetime
from pathlib import Path
import pickle
import random
import logging
import glob

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import DQN, ReplayBuffer


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(__file__).resolve().parent / "model_run_6"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "checkpoint.pkl"

# LOAD_MODEL_DIR = MODEL_DIR
# if not LOAD_MODEL_DIR.exists:
#     raise ("LOAD_MODEL_DIR does not exist.")

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

# Set device
# Check that MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

gym.register_envs(ale_py)

env = gym.make(
    "ALE/Pong-v5",
    render_mode="rgb_array",
    # frameskip=1 disables ALE's internal frameskip and allows `frame_skip=4`
    # in AtariPreprocessing.
    frameskip=1,
)
env = AtariPreprocessing(
    env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30
)
env = FrameStackObservation(env, stack_size=4)  # Stack 4 frames
action_dim = env.action_space.n


def save_checkpoint(
    model,
    optimizer,
    target_model,
    replay_buffer,
    episode,
    step,
    best_reward,
    checkpoint_path,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "replay_buffer": replay_buffer,
        "episode": episode,
        "step": step,
        "best_reward": best_reward,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_latest_checkpoint(checkpoint_path: Path, model, optimizer, target_model):
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    target_model.load_state_dict(checkpoint["target_model_state_dict"])
    replay_buffer = checkpoint["replay_buffer"]
    episode = checkpoint["episode"]
    step = checkpoint["step"]
    best_reward = checkpoint["best_reward"]

    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return replay_buffer, episode, step, best_reward


# def load_latest_model(model_dir):
#     # Load the model
#     model = DQN(action_dim).to(device)

#     """Load the latest saved model from the model directory."""
#     model_files = glob.glob(str(model_dir / "pong_dqn_episode_*.pth"))
#     if not model_files:
#         return model, 0

#     # Get the latest model file
#     latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
#     episode_num = int(latest_model.split("_")[-1].split(".")[0])
#     logging.info(f"Found latest_model @ {latest_model}")
#     # logging.info(f"Found latest_model @ {latest_model}")

#     logging.info(f"Loading latest_model @ {latest_model}")
#     model.load_state_dict(
#         torch.load(latest_model, map_location=device, weights_only=False)
#     )
#     logging.info(f"Loaded model from episode {episode_num}")

#     return model, episode_num


# Training parameters
NUM_EPISODES = 10000
BATCH_SIZE = 64  # Increased from 32
GAMMA = 0.99
UPDATE_TARGET_EVERY = 100  # More frequent updates
REPLAY_BUFFER_SIZE = 50000  # Increased from 10000
LEARNING_RATE = 0.00025  # Adjusted learning rate
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Slower decay
MIN_REPLAY_SIZE = 20000  # Minimum experiences before training


# Initialize models and replay buffer
model = DQN(action_dim).to(device)
target_model = DQN(action_dim).to(device)

# # Try to load the latest model
# loaded_model, start_episode = load_latest_model(LOAD_MODEL_DIR)
# if loaded_model is not None:
#     model = loaded_model
#     target_model.load_state_dict(model.state_dict())
#     logging.info("Successfully loaded model and synchronized target model")
# else:
#     logging.info("No saved model found, starting from scratch")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-4)


def train_dqn(
    env,
    model,
    target_model,
    optimizer,
    replay_buffer,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
):
    if len(replay_buffer) < batch_size:
        return

    # Sample a mini-batch from the replay buffer
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute Q-values and target Q-values
    current_q_values = model(states).gather(1, actions.unsqueeze(1))
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss and update the model
    loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# Training loop
if not CHECKPOINT_PATH.exists():
    best_reward = float("-inf")
    start_episode = 0
    step_count = 0
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    logging.info("No checkpoint found, starting fresh.")
else:
    replay_buffer, start_episode, step_count, best_reward = load_latest_checkpoint(
        CHECKPOINT_PATH, model, optimizer, target_model
    )

epsilon = EPSILON_START

for episode in range(start_episode, NUM_EPISODES):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    episode_losses = []

    while not done:
        step_count += 1
        # Select action using epsilon-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = torch.argmax(q_values).item()

        # Take action and observe next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated or truncated

        # Store experience in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train the model if we have enough experiences
        if len(replay_buffer) > MIN_REPLAY_SIZE:
            loss = train_dqn(
                env, model, target_model, optimizer, replay_buffer, BATCH_SIZE, GAMMA
            )
            episode_losses.append(loss)

    # Update target network
    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.load_state_dict(model.state_dict())
        logging.info(f"Updated target network at episode {episode}")

    # Update epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Save the model if it's the best so far
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(model.state_dict(), MODEL_DIR / "pong_dqn_best.pth")
        logging.info(f"New best model saved with reward: {best_reward}")

    # <-- Save checkpoint every 100 episodes
    if (episode + 1) % 100 == 0:
        save_checkpoint(
            model,
            optimizer,
            target_model,
            replay_buffer,
            episode,
            step_count,
            best_reward,
            CHECKPOINT_PATH,
        )
        logging.info(f"Checkpointing at episode {episode + 1}")

    # # Save the model periodically
    # if (episode + 1) % 100 == 0:
    #     torch.save(
    #         model.state_dict(), MODEL_DIR / f"pong_dqn_episode_{episode + 1}.pth"
    #     )
    #     logging.info(f"Model saved at episode {episode + 1}")

    # Log episode statistics
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    logging.info(
        f"Episode {episode + 1}, Total Reward: {total_reward}, "
        f"Average Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}"
    )
