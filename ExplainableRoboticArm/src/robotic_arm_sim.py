# robotic_arm_sim.py
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class RoboticArmEnv(gym.Env):
    def __init__(self):
        super(RoboticArmEnv, self).__init__()
        # Connect to PyBullet
        self.client = p.connect(p.GUI)
        if self.client == -1:
            raise RuntimeError("Failed to connect to PyBullet")
        
        # Set the search path and gravity
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load the plane and robotic arm (using a simple URDF)
        self.plane_id = p.loadURDF("plane.urdf")
        if self.plane_id < 0:
            raise RuntimeError("Failed to load plane.urdf")
        
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        if self.arm_id < 0:
            raise RuntimeError("Failed to load kuka_iiwa/model.urdf")
        
        self.cube_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        if self.cube_id < 0:
            raise RuntimeError("Failed to load cube_small.urdf")
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 3 joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)  # 7 joints (pos+vel) + cube pos
        
        self.target_pos = [0.5, 0.5, 0.05]  # Target position to place the cube
    
    def reset(self):
        p.resetBasePositionAndOrientation(self.cube_id, [0.5, 0, 0.05], [0, 0, 0, 1])
        for i in range(p.getNumJoints(self.arm_id)):
            p.resetJointState(self.arm_id, i, 0)
        obs = self._get_obs()
        return obs, {}  # Return observation and info dict

    def _get_obs(self):
        # Get joint states (positions and velocities)
        joint_states = []
        for i in range(p.getNumJoints(self.arm_id)):
            pos, vel = p.getJointState(self.arm_id, i)[:2]
            joint_states.extend([pos, vel])
        # Get cube position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = list(cube_pos)  # Convert tuple to list
        return np.array(joint_states + cube_pos, dtype=np.float32)

    def step(self, action):
        for i in range(3):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        dist = np.linalg.norm(np.array(cube_pos) - np.array(self.target_pos))
        reward = -dist
        done = dist < 0.05
        obs = self._get_obs()
        return obs, reward, done, False, {}  # obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect()

def train_model():
    env = RoboticArmEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_robotic_arm")
    env.close()

if __name__ == "__main__":
    train_model()
