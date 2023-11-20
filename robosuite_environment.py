import robosuite as suite
import numpy as np
import torch

from robosuite.wrappers import Wrapper

class RoboSuiteWrapper(Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.max_episode_steps = 200
        self.current_episode_step = 0

        # if not test:

        # else:
        #     self.env = suite.make(
        #         env_name,  # Environment
        #         robots=["Panda"],  # Use two Panda robots
        #         controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        #         has_renderer=True,  # Enable rendering
        #         render_camera="sideview",           # Camera view
        #         has_offscreen_renderer=True,        # No offscreen rendering
        #         control_freq=20,  # Control frequency
        #     )



    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.observation_to_tensor(observation)

        # Increment timesteps and set done if max timesteps reached
        self.current_episode_step += 1

        if self.current_episode_step == self.max_episode_steps:
            done = True

        return observation, reward, done, info

    def reset(self):
        self.current_episode_step = 0
        observation = super().reset()
        observation = self.observation_to_tensor(observation)
        return observation


    def observation_to_tensor(self, obs):
        # Convert each array in the ordered dictionary to a flattened numpy array
        flattened_arrays = [np.array(item).flatten() for item in obs.values()]

        # Concatenate all the flattened arrays to get a single array
        concatenated_array = np.concatenate(flattened_arrays)

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(concatenated_array, dtype=torch.float32)
