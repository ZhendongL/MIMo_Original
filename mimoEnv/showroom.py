""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""

import gymnasium as gym
import time
import numpy as np
import mimoEnv
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """
    #MIMoShowroom
    env = gym.make("MIMoShowroom-v0")

    max_steps = 100

    _ = env.reset()

    set_random_seed(50)

    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()

if __name__ == "__main__":
    main()
