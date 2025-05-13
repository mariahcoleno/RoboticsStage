import os
import argparse
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import logging

# Configure logging to show INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoboticArmEnv(gym.Env):
    def __init__(self, use_gui=False):
        super(RoboticArmEnv, self).__init__()
        self.use_gui = use_gui
        self.physics_client = None
        self.arm_id = None
        self.cube_id = None
        self.target_id = None
        self.total_steps = 0
        self.rng = np.random.default_rng()
        self.grasp_constraint_id = -1
        
        # Define workspace boundaries
        self.workspace_limits = {
            'x_min': -0.5, 'x_max': 1.0,
            'y_min': -0.5, 'y_max': 0.5,
            'z_min': 0.0, 'z_max': 0.5
        }

        # Initialize state variables
        self.ee_pos = np.zeros(3)
        self.cube_pos = np.zeros(3)
        self.target_pos = np.array([0.5, 0.2, 0.05])
        self.gripper_state = 0
        self.grasping = False
        self.ever_grasped = False
        self.prev_cube_target_dist = None
        self.prev_dist_ee_cube = None
        
        # More simulation steps for better stability
        self.sim_steps_per_action = 20  # Reduced from 50 for better control

        # Connect to PyBullet
        self._connect_physics()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def _connect_physics(self):
        if self.physics_client is not None and p.isConnected():
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Set a smaller time step for better physics stability
        p.setTimeStep(1/240.0)  # Default is 1/240, but explicit is better
        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation()
        plane_id = p.loadURDF("plane.urdf")
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.cube_start_pos = [0.5, 0, 0.05]
        
        # Add damping to the cube to prevent it from flying away
        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos, globalScaling=0.5)
        p.changeDynamics(self.cube_id, -1, 
                         linearDamping=0.9,      # Add damping to reduce velocity
                         angularDamping=0.9,     # Add damping to reduce rotation
                         mass=0.1)              # Reduce mass for stability
        
        # Create target visualization
        if self.use_gui:
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
            self.target_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual,
                basePosition=self.target_pos
            )
            
        # Create workspace boundary visualization (only in GUI mode)
        if self.use_gui:
            ws = self.workspace_limits
            self._create_workspace_visualization(ws)
            
        p.setRealTimeSimulation(0)

        # Log joint info for debugging
        num_joints = p.getNumJoints(self.arm_id)
        logger.info(f"Number of joints in arm: {num_joints}")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.arm_id, i)
            logger.info(f"Joint {i}: {joint_info}")
    
    def _create_workspace_visualization(self, ws):
        """Create transparent boundary visualization for workspace limits"""
        # Create a line for each edge of the workspace (12 edges in total for a cuboid)
        line_width = 2
        color = [0.8, 0.8, 0.8, 0.3]  # Light gray with transparency
        
        # Bottom square
        p.addUserDebugLine([ws['x_min'], ws['y_min'], ws['z_min']], 
                           [ws['x_max'], ws['y_min'], ws['z_min']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_min'], ws['z_min']], 
                           [ws['x_max'], ws['y_max'], ws['z_min']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_max'], ws['z_min']], 
                           [ws['x_min'], ws['y_max'], ws['z_min']], color, line_width)
        p.addUserDebugLine([ws['x_min'], ws['y_max'], ws['z_min']], 
                           [ws['x_min'], ws['y_min'], ws['z_min']], color, line_width)
        
        # Top square
        p.addUserDebugLine([ws['x_min'], ws['y_min'], ws['z_max']], 
                           [ws['x_max'], ws['y_min'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_min'], ws['z_max']], 
                           [ws['x_max'], ws['y_max'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_max'], ws['z_max']], 
                           [ws['x_min'], ws['y_max'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_min'], ws['y_max'], ws['z_max']], 
                           [ws['x_min'], ws['y_min'], ws['z_max']], color, line_width)
        
        # Vertical lines
        p.addUserDebugLine([ws['x_min'], ws['y_min'], ws['z_min']], 
                           [ws['x_min'], ws['y_min'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_min'], ws['z_min']], 
                           [ws['x_max'], ws['y_min'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_max'], ws['y_max'], ws['z_min']], 
                           [ws['x_max'], ws['y_max'], ws['z_max']], color, line_width)
        p.addUserDebugLine([ws['x_min'], ws['y_max'], ws['z_min']], 
                           [ws['x_min'], ws['y_max'], ws['z_max']], color, line_width)

    def _get_obs(self):
        if not p.isConnected():
            logger.warning("Physics client disconnected during _get_obs. Attempting to reconnect...")
            self.reset()
        
        try:
            cube_pos_data = p.getBasePositionAndOrientation(self.cube_id)
            if cube_pos_data is None:
                logger.error("Failed to get cube position: p.getBasePositionAndOrientation returned None")
                raise RuntimeError("Failed to get cube position")
            cube_pos = np.array(cube_pos_data[0])
            self.cube_pos = cube_pos
        except Exception as e:
            logger.error(f"getBasePositionAndOrientation failed in _get_obs: {e}")
            raise RuntimeError("getBasePositionAndOrientation failed")

        try:
            ee_pos_data = p.getLinkState(self.arm_id, 6)
            if ee_pos_data is None:
                logger.error("Failed to get EE position: p.getLinkState returned None")
                raise RuntimeError("Failed to get EE position")
            ee_pos = np.array(ee_pos_data[0])
            self.ee_pos = ee_pos
        except Exception as e:
            logger.error(f"getLinkState failed in _get_obs: {e}")
            raise RuntimeError("getLinkState failed")

        target_pos = self.target_pos
        ee_to_cube = ee_pos - cube_pos
        cube_to_target = cube_pos - target_pos
        return np.concatenate([cube_pos, ee_pos, target_pos, ee_to_cube, cube_to_target, [self.gripper_state]], dtype=np.float32)

    def _enforce_workspace_limits(self):
        """Keep cube within workspace limits"""
        ws = self.workspace_limits
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        
        # Check if cube is outside workspace
        if (cube_pos[0] < ws['x_min'] or cube_pos[0] > ws['x_max'] or
            cube_pos[1] < ws['y_min'] or cube_pos[1] > ws['y_max'] or
            cube_pos[2] < ws['z_min'] or cube_pos[2] > ws['z_max']):
            
            # Reset cube to be inside workspace
            new_pos = [
                np.clip(cube_pos[0], ws['x_min'] + 0.05, ws['x_max'] - 0.05),
                np.clip(cube_pos[1], ws['y_min'] + 0.05, ws['y_max'] - 0.05),
                np.clip(cube_pos[2], ws['z_min'] + 0.05, ws['z_max'] - 0.05)
            ]
            
            # If cube was flying away, reset it fully
            dist_from_origin = np.linalg.norm(np.array(cube_pos))
            if dist_from_origin > 2.0:  # If cube is way out
                new_pos = self.cube_start_pos
                linear_vel = [0, 0, 0]
                angular_vel = [0, 0, 0]
                p.resetBasePositionAndOrientation(self.cube_id, new_pos, [0, 0, 0, 1])
                p.resetBaseVelocity(self.cube_id, linear_vel, angular_vel)
                logger.warning(f"Cube reset from {cube_pos} to {new_pos} due to leaving workspace")
            else:
                # Just reset position, keep orientation and add some damping
                cube_lin_vel, cube_ang_vel = p.getBaseVelocity(self.cube_id)
                p.resetBasePositionAndOrientation(self.cube_id, new_pos, cube_orn)
                # Dampen velocity
                p.resetBaseVelocity(
                    self.cube_id, 
                    [v * 0.5 for v in cube_lin_vel],  # Dampen linear velocity
                    [v * 0.5 for v in cube_ang_vel]   # Dampen angular velocity
                )
                logger.info(f"Cube contained within workspace: {new_pos}")

    def _apply_action(self, action):
        # Scale action for smoother control
        dx, dy, dz, gripper_action = action
        dx = dx * 0.03  # Reduced from 0.05
        dy = dy * 0.03  # Reduced from 0.05
        dz = dz * 0.03  # Reduced from 0.05
        gripper_action = gripper_action * 0.05  # Keep the same

        try:
            ee_pos_data = p.getLinkState(self.arm_id, 6)
            if ee_pos_data is None:
                logger.error("Failed to get EE position: p.getLinkState returned None")
                raise RuntimeError("Failed to get EE position")
            current_pos = np.array(ee_pos_data[0])
        except Exception as e:
            logger.error(f"getLinkState failed in _apply_action: {e}")
            raise RuntimeError("getLinkState failed")

        # Compute new position
        new_pos = current_pos + np.array([dx, dy, dz])
        
        # Constrain end-effector to workspace limits
        ws = self.workspace_limits
        new_pos[0] = np.clip(new_pos[0], ws['x_min'] + 0.05, ws['x_max'] - 0.05)
        new_pos[1] = np.clip(new_pos[1], ws['y_min'] + 0.05, ws['y_max'] - 0.05)
        new_pos[2] = np.clip(new_pos[2], ws['z_min'] + 0.05, ws['z_max'] - 0.05)

        # Calculate IK
        joint_positions = p.calculateInverseKinematics(self.arm_id, 6, new_pos)

        # Apply joint positions with smoother motion
        for i in range(len(joint_positions)):
            p.setJointMotorControl2(
                self.arm_id, 
                i, 
                p.POSITION_CONTROL, 
                targetPosition=joint_positions[i],
                maxVelocity=0.5  # Limit velocity for smoother motion
            )

        # Run simulation with smaller steps for stability
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation()
            self._enforce_workspace_limits()  # Keep cube in workspace
            
            # Check for extremely high velocities and dampen them
            if self.grasp_constraint_id != -1:  # Only when grasping
                lin_vel, ang_vel = p.getBaseVelocity(self.cube_id)
                lin_vel_norm = np.linalg.norm(lin_vel)
                ang_vel_norm = np.linalg.norm(ang_vel)
                
                if lin_vel_norm > 2.0 or ang_vel_norm > 5.0:
                    p.resetBaseVelocity(
                        self.cube_id,
                        [v * 0.8 for v in lin_vel],  # Dampen velocity
                        [v * 0.8 for v in ang_vel]   # Dampen angular velocity
                    )
                    logger.info(f"Dampened high velocity: linear={lin_vel_norm:.2f}, angular={ang_vel_norm:.2f}")

        # Update gripper state
        prev_gripper_state = self.gripper_state
        self.gripper_state = 1 if gripper_action > 0 else 0

        # Calculate distance for grasping
        dist_ee_cube = np.linalg.norm(current_pos - np.array(p.getBasePositionAndOrientation(self.cube_id)[0]))
        grasp_threshold = 0.15

        # Handle grasping with more careful constraint parameters
        if self.gripper_state == 1 and dist_ee_cube < grasp_threshold and self.grasp_constraint_id == -1:
            # Add a small delay for stability
            for _ in range(5):
                p.stepSimulation()
                
            # Create constraint with more specific parameters
            self.grasp_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.arm_id,
                parentLinkIndex=6,
                childBodyUniqueId=self.cube_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.05],  # Offset slightly for stability
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=[0, 0, 0, 1]
            )
            # Set constraint parameters for stability
            p.changeConstraint(
                self.grasp_constraint_id, 
                maxForce=20,  # Limit maximum force
                erp=0.9       # Error reduction parameter (0-1) for stability
            )
            logger.info("Grasp constraint created with stability parameters.")

        # Handle release with gradual force reduction
        if self.gripper_state == 0 and prev_gripper_state == 1 and self.grasp_constraint_id != -1:
            # Gradually reduce constraint force before removing
            p.changeConstraint(self.grasp_constraint_id, maxForce=10)
            for _ in range(5):
                p.stepSimulation()
            p.changeConstraint(self.grasp_constraint_id, maxForce=5)
            for _ in range(5):
                p.stepSimulation()
                
            # Now remove constraint
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = -1
            logger.info("Grasp constraint gradually removed.")

    def step(self, action):
        self.total_steps += 1
        self._apply_action(action)
        obs = self._get_obs()

        try:
            cube_pos_data = p.getBasePositionAndOrientation(self.cube_id)
            if cube_pos_data is None:
                logger.error("Failed to get cube position: p.getBasePositionAndOrientation returned None")
                raise RuntimeError("Failed to get cube position")
            self.cube_pos = np.array(cube_pos_data[0])
        except Exception as e:
            logger.error(f"getBasePositionAndOrientation failed in step: {e}")
            raise RuntimeError("getBasePositionAndOrientation failed")

        try:
            ee_pos_data = p.getLinkState(self.arm_id, 6)
            if ee_pos_data is None:
                logger.error("Failed to get EE position: p.getLinkState returned None")
                raise RuntimeError("Failed to get EE position")
            self.ee_pos = np.array(ee_pos_data[0])
        except Exception as e:
            logger.error(f"getLinkState failed in step: {e}")
            raise RuntimeError("getLinkState failed")

        dist_ee_cube = np.linalg.norm(self.ee_pos - self.cube_pos)
        dist_cube_target = np.linalg.norm(self.cube_pos - self.target_pos)
        dist_cube_start = np.linalg.norm(self.cube_pos - np.array(self.cube_start_pos))

        grasp_threshold = 0.15
        self.grasping = self.grasp_constraint_id != -1
        if self.grasping:
            self.ever_grasped = True

        # IMPROVED REWARD FUNCTION WITH SCALED VALUES
        reward = 0.0
        done = False
        
        # Small base penalty for time
        reward -= 0.1  # Reduced from 1.0
        
        # Stage 1: Approaching the cube
        if not self.ever_grasped:
            # Reward for getting close to the cube - scaled down
            reward -= 5.0 * dist_ee_cube  # Reduced from 50.0
            
            # Reward for closing gripper when close to cube
            if dist_ee_cube < grasp_threshold and self.gripper_state == 1:
                reward += 5.0  # Reduced from 50.0
                
            # Shaping reward for approaching cube - scaled down
            if self.prev_dist_ee_cube is not None:
                approach_reward = 20.0 * (self.prev_dist_ee_cube - dist_ee_cube)  # Reduced from 200.0
                reward += approach_reward
        
        # Stage 2: Grasping and moving the cube
        if self.grasping:
            # Base reward for maintaining grasp
            reward += 10.0  # Reduced from 100.0
            
            # Reward for moving cube toward target - scaled down
            reward -= 30.0 * dist_cube_target  # Reduced from 300.0
            
            # Shaping reward for moving toward target - scaled down
            if self.prev_cube_target_dist is not None:
                target_approach_reward = 50.0 * (self.prev_cube_target_dist - dist_cube_target)  # Reduced from 500.0
                reward += target_approach_reward
        
        # Stage 3: Releasing at target
        if dist_cube_target < 0.15:
            # Near target, give extra reward
            reward += 10.0  # Reduced from 100.0
            
            # If close to target and gripper open, give success reward
            if dist_cube_target < 0.1 and self.gripper_state == 0:
                reward += 100.0  # Reduced from 1000.0
                done = True  # End episode on success
        
        # Stage 4: Penalties - scaled down
        # Penalty for dropping cube away from target
        if not self.grasping and self.ever_grasped and dist_cube_target > 0.15:
            reward -= 20.0  # Reduced from 200.0
        
        # Update previous distances for next step
        self.prev_dist_ee_cube = dist_ee_cube
        self.prev_cube_target_dist = dist_cube_target

        # Handle case where cube flies away too far
        if np.linalg.norm(self.cube_pos) > 5.0:
            logger.warning(f"Cube flew too far: {self.cube_pos}. Ending episode.")
            reward -= 50.0  # Penalty for cube flying away
            done = True
            
        # Force cube to stay within reasonable bounds
        ws = self.workspace_limits
        if (self.cube_pos[0] < ws['x_min'] or self.cube_pos[0] > ws['x_max'] or
            self.cube_pos[1] < ws['y_min'] or self.cube_pos[1] > ws['y_max'] or
            self.cube_pos[2] < ws['z_min'] or self.cube_pos[2] > ws['z_max']):
            reward -= 1.0  # Small penalty for going out of bounds

        truncated = self.total_steps >= 400

        done = bool(done)
        truncated = bool(truncated)

        info = {
            "dist_ee_cube": dist_ee_cube,
            "dist_cube_target": dist_cube_target,
            "dist_cube_start": dist_cube_start,
            "grasping": self.grasping,
            "gripper_state": self.gripper_state,
            "ee_pos": self.ee_pos,
            "cube_pos": self.cube_pos
        }

        # Log first step of each episode
        if self.total_steps == 1:
            logger.info(f"First step: EE position = {self.ee_pos}, Cube position = {self.cube_pos}, "
                        f"Dist EE-Cube = {dist_ee_cube:.3f}, Grasping = {self.grasping}, Gripper State = {self.gripper_state}")

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not p.isConnected():
            logger.warning("Physics client disconnected during reset. Attempting to reconnect...")
            self._connect_physics()

        if self.grasp_constraint_id != -1:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = -1

        self.total_steps = 0
        self._setup_scene()
        self.ee_pos = np.zeros(3)
        self.cube_pos = np.zeros(3)
        self.gripper_state = 0
        self.grasping = False
        self.ever_grasped = False
        self.prev_cube_target_dist = None
        self.prev_dist_ee_cube = None

        # Start the EE closer to the cube
        initial_ee_pos = [0.5, 0.0, 0.1]
        joint_positions = p.calculateInverseKinematics(self.arm_id, 6, initial_ee_pos)
        for i in range(len(joint_positions)):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation()

        # Start with cube in slightly randomized position but more constrained
        cube_start_pos = list(self.cube_start_pos)  # Make a copy
        cube_start_pos[0] += self.rng.uniform(-0.03, 0.03)  # Reduced randomization
        cube_start_pos[1] += self.rng.uniform(-0.03, 0.03)  # Reduced randomization
        p.resetBasePositionAndOrientation(self.cube_id, cube_start_pos, [0, 0, 0, 1])
        # Reset velocities explicitly
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])

        self.target_pos = np.array([0.5, 0.2, 0.05])
        obs = self._get_obs()
        logger.info(f"After reset: EE position = {self.ee_pos}, Cube position = {self.cube_pos}, "
                    f"Dist EE-Cube = {np.linalg.norm(self.ee_pos - self.cube_pos):.3f}")
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        if self.grasp_constraint_id != -1:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = -1
        if p.isConnected():
            p.disconnect(self.physics_client)

class LoggingCallback(BaseCallback):
    def __init__(self, log_freq=10000, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            env = self.model.env
            env = env.envs[0] if hasattr(env, 'envs') else env
            env = env.env if hasattr(env, 'env') else env

            infos = self.locals.get('infos', [{}])
            info = infos[0] if infos else {}

            reward = self.locals.get('rewards', [0])[0]
            action = self.locals.get('actions', np.zeros(4))
            if action.ndim > 1:
                action = action[0]

            print(f"Timestep {self.num_timesteps}: EE position = {info.get('ee_pos', [0, 0, 0])}, "
                  f"Cube position = {info.get('cube_pos', [0, 0, 0])}, "
                  f"Dist EE-Cube = {info.get('dist_ee_cube', 0):.3f}, "
                  f"Dist Cube-Target = {info.get('dist_cube_target', 0):.3f}, "
                  f"Dist Cube-Start = {info.get('dist_cube_start', 0):.3f}, "
                  f"Reward = {reward:.3f}, "
                  f"Grasping = {info.get('grasping', False)}, "
                  f"Gripper State = {info.get('gripper_state', 0)}")
            
            if info.get("grasping", False):
                cube_vel_linear, cube_vel_angular = p.getBaseVelocity(env.cube_id)
                print(f"Timestep {self.num_timesteps}: Cube velocity while grasping: "
                      f"linear = {cube_vel_linear}, angular = {cube_vel_angular}")
            if info.get("dist_ee_cube", float('inf')) < 0.15 and info.get("gripper_state", 0) == 1:
                print(f"Timestep {self.num_timesteps}: Grasped cube")
            if info.get("dist_cube_target", float('inf')) < 0.1 and info.get("gripper_state", 0) == 0:
                print(f"Timestep {self.num_timesteps}: Released cube")
        return True

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f"{self.name_prefix}_checkpoint_{self.num_timesteps}")
            self.model.save(checkpoint_path)
            print(f"Checkpoint saved at total timestep {self.num_timesteps}: {checkpoint_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Robotic Arm Simulation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"], help="Mode to run the script in")
    parser.add_argument("--timesteps", type=int, default=2000000, help="Number of timesteps to train for")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()

    if args.mode == "train":
        env = RoboticArmEnv(use_gui=False)
        check_env(env)
        # Removed NormalizeReward wrapper to get raw rewards
        logger = configure("./logs", ["stdout", "tensorboard"])
        total_timesteps = args.timesteps
        checkpoint_path = args.checkpoint

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model = PPO.load(checkpoint_path, env=env)
            model.ent_coef = 0.001  # Reduced exploration
        else:
            ent_coef = 0.001  # Reduced exploration
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs", ent_coef=ent_coef, learning_rate=0.0003, batch_size=64, n_steps=4096)

        model.set_logger(logger)
        checkpoint_callback = CheckpointCallback(save_freq=512000, save_path=".", name_prefix="ppo_robotic_arm")
        logging_callback = LoggingCallback(log_freq=10000)
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, logging_callback])
        model.save("ppo_robotic_arm")
        env.close()
        logger.info("Training completed and model saved.")

    elif args.mode == "evaluate":
        import time
        os.makedirs("screenshots", exist_ok=True)
        env = RoboticArmEnv(use_gui=True)
        model = PPO.load("ppo_robotic_arm", env=env)

        for episode in range(5):
            obs, _ = env.reset()
            total_reward = 0
            for step in range(400):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

                print(f"Timestep {step + 1}: Raw action = {action}")
                print(f"Timestep {step + 1}: Gripper action = {action[3]:.3f}, Gripper State = {info['gripper_state']}")
                if info["grasping"]:
                    cube_vel_linear, cube_vel_angular = p.getBaseVelocity(env.cube_id)
                    print(f"Timestep {step + 1}: Cube velocity while grasping: linear = {cube_vel_linear}, angular = {cube_vel_angular}")
                if info["dist_ee_cube"] < 0.15 and info["gripper_state"] == 1:
                    print(f"Timestep {step + 1}: Grasped cube")
                if info["dist_cube_target"] < 0.1 and info["gripper_state"] == 0:
                    print(f"Timestep {step + 1}: Released cube")

                if step % 10 == 0 or done or truncated:
                    print(f"Timestep {step + 1}: EE position = {info['ee_pos']}, Cube position = {info['cube_pos']}, "
                          f"Dist EE-Cube = {info['dist_ee_cube']:.3f}, Dist Cube-Target = {info['dist_cube_target']:.3f}, "
                          f"Dist Cube-Start = {info['dist_cube_start']:.3f}, Reward = {reward:.3f}, "
                          f"Grasping = {info['grasping']}, Gripper State = {info['gripper_state']}")
                    if done:
                        print(f"Timestep {step + 1}: Episode completed successfully!")
                    p.saveBullet(f"screenshots/episode_{episode + 1}_step_{step}.bullet")
                    img = p.getCameraImage(640, 480)[2]
                    img = np.reshape(img, (480, 640, 4))
                    img = img.astype(np.uint8)
                    import imageio
                    imageio.imwrite(f"screenshots/episode_{episode + 1}_step_{step}.png", img)
                    print(f"Screenshot saved as screenshots/episode_{episode + 1}_step_{step}.png")

                time.sleep(0.005)
                if done or truncated:
                    break

            print(f"Episode {episode + 1} completed. Total reward: {total_reward:.3f}")
        env.close()

if __name__ == "__main__":
    main()
