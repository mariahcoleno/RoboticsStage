# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from robotic_arm_sim import RoboticArmEnv
from explain_decisions import explain_action

class RoboticArmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Explainable Robotic Arm Simulator")
        self.env = RoboticArmEnv()
        
        # Buttons
        ttk.Button(root, text="Run Simulation", command=self.run_simulation).pack(pady=5)
        ttk.Button(root, text="Explain Action", command=self.explain_action).pack(pady=5)
        ttk.Button(root, text="Quit", command=self.quit).pack(pady=5)
        
        # Image display for SHAP explanation
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=5)
    
    def run_simulation(self):
        obs = self.env.reset()
        model = PPO.load("ppo_robotic_arm")
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.env.step(action)
        messagebox.showinfo("Simulation", "Simulation completed!")
    
    def explain_action(self):
        explain_action("ppo_robotic_arm", self.env)
        # Display the SHAP plot
        img = Image.open("shap_explanation.png")
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def quit(self):
        self.env.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RoboticArmGUI(root)
    root.mainloop()
