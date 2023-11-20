import robosuite as suite
import numpy as np
import os
import torch
from agent import Agent
from robosuite_environment import RoboSuiteWrapper

# Set up device as either the GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Loading device: {device}")

# Setup the necessary paths for training
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./plots"):
    os.makedirs("./plots")

env_name = "Door"

env = suite.make(
    env_name,  # Environment
    robots=["Panda"],  # Use two Panda robots
    controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
    # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,  # Enable rendering
    use_camera_obs=False,
    # render_camera="sideview",           # Camera view
    # has_offscreen_renderer=True,        # No offscreen rendering
    reward_shaping=True,
    control_freq=20,  # Control frequency
)

env = RoboSuiteWrapper(env)



# Reset the environment and get environment params
obs = env.reset()

state_dim = obs.shape[0]
action_dim = env.action_dim
max_action = 1

print(action_dim)
# min_action, max_action = env.action_spec
#
# print(max_action)



agent = Agent(state_dim, action_dim, max_action=max_action, batch_size=16, policy_freq=2,
            discount=0.99, device=device, tau=0.005, policy_noise=0.1, expl_noise=0.1,
            noise_clip=0.5, start_timesteps=10000, learning_rate=0.0001, env_name=env_name,
            lr_decay_factor=0.999, epsilon=0.01, min_epsilon=0.01)


stats = agent.train(env, max_timesteps=1e7, batch_identifier=1)
