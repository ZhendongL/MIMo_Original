""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""
import gymnasium as gym
import time
import numpy as np
import mimoEnv

def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """
    env = gym.make("MIMoSelfBody-v0")

    _ = env.reset()

    start = time.time()

    for step in range(1000):
        action = env.action_space.sample()
        # action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.mujoco_renderer.render(render_mode="human")
        # if done or trunc:
        #     env.reset()

    # print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    # env.close()

def learn():
    from stable_baselines3 import PPO
    env = gym.make("MIMoSelfBody-v0")

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        env.mujoco_renderer.render(render_mode="human")

        # if done:
        #   obs = vec_env.reset()

if __name__ == "__main__":
    main()
    # learn()

