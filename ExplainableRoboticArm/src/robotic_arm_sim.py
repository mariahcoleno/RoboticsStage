# robotic_arm_sim.py
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class RoboticArmEnv(gym.Env):
    def __init__(self):
        super(RoboticArmEnv, self).__init__()
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load the plane and robotic arm (using a simple URDF)
        self.plane_id = p.loadURDF("plane.urdf")
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        self.cube_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 3 joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)  # Joint positions + cube position
        
        self.target_pos = [0.5, 0.5, 0.05]  # Target position to place the cube
    
    def reset(self):
        # Reset the arm and cube positions
        p.resetBasePositionAndOrientation(self.cube_id, [0.5, 0, 0.05], [0, 0, 0, 1])
        for i in range(p.getNumJoints(self.arm_id)):
            p.resetJointState(self.arm_id, i, 0)
        return self._get_obs()
    
    def _get_obs(self):
        # Get joint positions and cube position
        joint_states = [p.getJointState(self.arm_id, i)[0] for i in range(3)]
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(joint_states + cube_pos, dtype=np.float32)
    
    def step(self, action):
        # Apply action to the arm
        for i in range(3):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        
        # Compute reward
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        dist = np.linalg.norm(np.array(cube_pos) - np.array(self.target_pos))
        reward = -dist  # Negative distance as reward
        done = dist < 0.05  # Done if cube is close to target
        return self._get_obs(), reward, done, {}
    
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
