#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
# from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
# from flax.training import orbax_utils
# from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
# from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from brax.base import System

from copy import deepcopy as dc

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="URM",
    # track hyperparameters and run metadata
    name="binary",
    config={
    "HI": 0.02,
    }
)


# instantiate the environment
env_name = 'urm2arm_binary'
env = envs.get_environment(env_name)

def policy_params_fn(current_step, make_policy, params):
  model_path = '/home/tlee_theaiinstitute_com/mjx_brax_policy/test_new_{}'.format(current_step)
  model.save_params(model_path, params)


num_evals= 50
num_envs = 1000
train_fn = functools.partial(
    ppo.train, num_timesteps=40*num_evals*num_envs, 
    num_evals=num_evals, reward_scaling=0.1,
    episode_length=40, normalize_observations=True, action_repeat=1,
    unroll_length=40, num_minibatches=10, num_updates_per_batch=2,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=num_envs, clipping_epsilon=0.3,
    batch_size=100, seed=0, discrete_action = True, spectral_norm_actor = False, policy_params_fn=policy_params_fn)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = -400, 0
def progress(num_steps, metrics):
#   dict_log = {"steps": num_steps, "reward": metrics['eval/episode_reward'], "y_reward": metrics['eval/episode_y_reward'], "theta_reward": metrics['eval/episode_theta_reward'] }
# #   dict_log.update(metrics)
#   wandb.log(dict_log)
  print("hello",datetime.now())
  print(metrics['eval/episode_reward'])
  print(metrics['eval/episode_y_reward'])
  print(metrics['eval/episode_theta_reward'])

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
wandb.finish()