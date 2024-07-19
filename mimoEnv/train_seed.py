import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO as RL
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gymnasium as gym
import time
import cv2

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel


# Define a function to create the environment
def make_env():
    return gym.make('MIMoSelfBody-v0', actuation_model=MuscleModel)

# Define the number of seeds you want to use and a list to store the results
num_seeds = 10
results = []

# Iterate over different seeds, create an environment, and train the agent using PPO or SAC
for seed in range(num_seeds):
    # Set the random seed
    set_random_seed(seed)

    # Create the environment
    env = make_vec_env(make_env, n_envs=1)

    # Create and train the PPO agent
    model = RL("MultiInputPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)  # Adjust total_timesteps as needed

    # Evaluate the agent and store the result
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    results.append(mean_reward)
    print('result',results)
