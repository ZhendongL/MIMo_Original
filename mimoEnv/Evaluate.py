import os
import time
import cv2
import mimoEnv
import gymnasium as gym
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from stable_baselines3.common.evaluation import evaluate_policy

algorithm = 'PPO'  # PPO  TD3
actuation_model = MuscleModel
env = gym.make('MIMoSelfBody-v0', actuation_model=actuation_model)

from stable_baselines3 import PPO as RL

num='1.0'
load_model = 'models/selfbody/selfbody'+algorithm+'version'+num+'/model_1'
model = RL.load(load_model, env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20,render=False)
print(f"Mean reward version{num}: {mean_reward} +/- {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()

DiyTouchList = [22, 18, 21, 17, 20, 16, 2, 3, 4, 5, 6, 7, 8, 12, 13, 11]

# Evaluate
for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human")
    # env.mujoco_renderer.render(render_mode="human")



