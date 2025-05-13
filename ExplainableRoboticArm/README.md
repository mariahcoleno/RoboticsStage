# RoboticsProjects
### ExplainableRoboticArm 
This project simulates a robotic arm that learns to pick and place objects in a 3D environment using reinforcement learning (PPO algorithm with Stable Baselines3). A Tkinter GUI allows users to run the simulation, train the model, and evaluate its performance. The project demonstrates skills in robotics simulation, reinforcement learning, and GUI development—key areas for advancing AI-driven robotics and human scientific discovery.

### Overview
- Objective: Train a robotic arm to pick up a cube and place it at a target location using reinforcement learning.
- Performance: Achieved a 67% success rate with an average episode length of 62 steps over 4.5M timesteps.
- GUI Features:
  - Train the model with configurable timesteps.
  - Evaluate the trained model to observe its behavior.
  - Visualize the arm’s actions in a 3D simulation environment.

### Files
- `requirements.txt`: Lists dependencies required to run scripts.
- `src/robotic_arm_sim.py`: Script used to train and evaluate the robot.
- `train_logs_backup.txt`: Logs used to monitor training progress.
- `ppo_robotic_arm.zip`: Saved model.
- `eval_logs.txt`: Logs used to evaluate results. 

### Setup and Usage
#### Option 1: From GitHub (Clone)
- **Note**:
  - Start in your preferred directory (e.g., cd ~/Desktop/ or cd ~/Downloads/ or cd ~/Documents/) to control where the repository clones. 
  - If you skip this step, it clones to your current directory.
1. Clone the repository: `git clone https://github.com/mariahcoleno/RoboticsStage.git`
2. Navigate to the ExplainableRoboticArm directory: `cd ExplainableRoboticArm/` (from the root of your cloned repository)
3. Create virtual environment: `python3 -m venv venv`
4. Activate: `source venv/bin/activate` # On Windows: venv\Scripts\activate
5. Install dependencies: `pip install -r requirements.txt`
6. Proceed to the "Run the Tool" section below.

#### Option 2: Local Setup (Existing Repository)
1. Navigate to your local repository `cd ~/Documents/RoboticsStage/` # Adjust path as needed
2. Navigate to ExplainableRoboticArm directory: `cd ExplainableRoboticArm/`
3. Setup and activate a virtual environment:
   - If existing: `source venv/bin/activate` # Adjust path if venv is elsewhere
   - If new:
     - `python3 -m venv venv`
     - `source venv/bin/activate` # On Windows: venv\Scripts\activate
4. Install dependencies (if not already): `pip install -r requirements.txt` 
5. Proceed to the "Run the Tool" section below.

### Run the Tool (Both Options):
- **Note**:
  - The project supports two modes: training and evaluation, controlled via command-line arguments.
1. Train the Model:
   - Run the script in training mode for a specified number of timesteps (e.g., 4.5M):
     - `python3 src/robotic_arm_sim.py --mode train --timesteps 4500000 2>&1 | tee train_logs_backup.txt`
   - Monitor Training Progress:
     - `tail -f train_logs_backup.txt`
   - The model will be saved as ppo_robotic_arm.zip.
2. Evaluate the Model: 
   - Run the script in evaluation mode to observe the trained model’s performance:
     - `python3 src/robotic_arm_sim.py --mode evaluate > eval_logs.txt 2>&1`
   - Check eval_logs.txt for results (e.g., success rate across episodes).
3. Load a Checkpoint (Optional): 
   - To resume training from a saved model:
     - `python3 src/robotic_arm_sim.py --mode train --timesteps 1000000 --checkpoint ppo_robotic_arm 2>&1 | tee train_logs_backup.txt`
     - **Note**:
       - Checkpoint loading may require troubleshooting if issues persist (see project notes). 
   
### Project Structure
- RoboticsStage/
  - ExplainableRoboticArm/
    - data/                 # (Optional) Directory for sample data or logs
    - logs/                 # Training logs (TensorBoard logs, ignored by Git)
    - src/                  # Source code
      - __init__.py         # Package initialization
      - robotic_arm_sim.py  # Main script for simulation, training, and GUI
    - train_logs_backup.txt # Training log file (ignored by Git)
    - ppo_robotic_arm.zip   # Saved PPO model (ignored by Git)
    - requirements.txt      # Dependencies
    - README.md             # Project Documentation

### Dependencies
- Listed in requirements.txt:
  - stable-baseline3 (for PPO implementation)
  - numpy (for numerical operations)
  - tkinter (for GUI, built-in with Python)
  - gym (for reinforcement learning environment)

### Features
- Reinforcement Learning: Trains the robotic arm using the PPO algorithm (Stable Baselines3) to optimize pick-and-place tasks.
- GUI Interaction: Tkinter-based GUI for running simulations, training, and evaluation.
- Logging: Detailed logging of training metrics (e.g., ep_rew_mean, ep_len_mean) and simulation states.

### Notes
- Ensure Python 3.8+ and virtual environment activation for dependency management.
- Training for 4.5M timesteps takes ~2.2 hours at 568 FPS (based on recent run).
- Checkpoint loading may encounter issues; consider starting fresh runs or debugging PPO.load() functionality.
