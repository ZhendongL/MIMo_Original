import gymnasium as gym
import time
import numpy as np
import mimoEnv
from stable_baselines3 import A2C
def main():

    env = gym.make('MIMoSelfBody-v0')
    max_steps = 1000

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    env = model.get_env()
    obs = env.reset()

    for step in range(max_steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done :
            env.reset()

    # print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    # env.close()

if __name__ == "__main__":
    main()


