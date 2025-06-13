import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import logging
import sys

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('fetch_logs.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create the environment
try:
    # Try with render_mode="human" first
    env = gym.make("FetchPickAndPlace-v2", render_mode="human")
except Exception as e:
    logger.warning(f"Failed to create environment with render_mode='human': {e}")
    logger.warning("Falling back to render_mode=None")
    env = gym.make("FetchPickAndPlace-v2", render_mode=None)

# Create the SAC model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=1000000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1
)

# Train the model
logger.info("Starting training...")
model.learn(total_timesteps=100000)
logger.info("Training completed.")

# Save the model
model.save("sac_fetch_pick_and_place")

# Test the trained model
obs, _ = env.reset()
for i in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if env.render_mode == "human":
        env.render()
    # Log key metrics
    gripper_pos = obs['observation'][:3]  # Gripper position
    cube_pos = obs['observation'][3:6]  # Cube position
    target_pos = obs['desired_goal']  # Target position
    dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
    dist_cube_target = np.linalg.norm(cube_pos - target_pos)
    logger.info(f"Step {i}: Gripper Pos = {gripper_pos}, Cube Pos = {cube_pos}, "
                f"Target Pos = {target_pos}, Dist Gripper-Cube = {dist_gripper_cube:.3f}, "
                f"Dist Cube-Target = {dist_cube_target:.3f}, Reward = {reward:.3f}, "
                f"Success = {info.get('is_success', 0.0)}")
    if done or truncated:
        logger.info(f"Episode ended at step {i}. Final Dist Cube-Target = {dist_cube_target:.3f}")
        break

env.close()
