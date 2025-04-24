# explain_decisions.py
import numpy as np
import shap
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def explain_action(model_path, env):
    # Load the trained model
    model = PPO.load(model_path)
    
    # Get a sample observation
    obs = env.reset()
    
    # Define a prediction function for SHAP
    def predict_fn(observations):
        actions, _ = model.predict(observations, deterministic=True)
        return actions
    
    # Use SHAP to explain the prediction
    explainer = shap.KernelExplainer(predict_fn, np.array([obs]))
    shap_values = explainer.shap_values(np.array([obs]))
    
    # Plot the SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=np.array([obs]), feature_names=["Joint 1", "Joint 2", "Joint 3", "Cube X", "Cube Y", "Cube Z"], show=False)
    plt.title("SHAP Explanation of Robotic Arm Action")
    plt.savefig("shap_explanation.png")
    plt.close()

if __name__ == "__main__":
    from robotic_arm_sim import RoboticArmEnv
    env = RoboticArmEnv()
    explain_action("ppo_robotic_arm", env)
    env.close()
