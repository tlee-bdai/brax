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
import json

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
from orbax import checkpoint as ocp
from flax.training import orbax_utils

from copy import deepcopy as dc

import wandb

env_name = 'urm2arm_binary'
exp_name = env_name+"_no_actionclip_noobsnormal_eps0dot1_damp1_xyt_bs0dot1_d0dot24_obsqpos_constvel6kp5_spectral"

num_envs = 2048
num_evals =150
rl_config = {
  "num_evals": num_evals,
  "num_envs": num_envs,
  "num_timesteps": 40*num_evals*num_envs, 
  "reward_scaling": 0.01,
  "episode_length": 40,
  "normalize_observations": False,
  "action_repeat": 1,
  "unroll_length": 40,
  "num_minibatches": 8,
  "num_updates_per_batch": 2,
  "discounting": 0.97,
  "learning_rate": 3e-4,
  "entropy_cost": 1e-2,
  "num_envs": num_envs,
  "clipping_epsilon": 0.3,
  "batch_size": 256,
  "seed": 0,
  "spectral_norm_actor": True,
}
w_config = {"env_name" : env_name}
w_config.update(rl_config)
  

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="URM",
    # track hyperparameters and run metadata
    name=exp_name,
    config=w_config
)


# instantiate the environment

env = envs.get_environment(env_name)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './output/{}/'.format(exp_name))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
json.dump(w_config, open(os.path.join(output_dir, "config.json"), "w") )
    

def policy_params_fn(current_step, make_policy, params):
  model_path = os.path.join(output_dir, "model_ckpt_{}".format(current_step)) 
  model.save_params(model_path, params)
# def policy_params_fn(current_step, make_policy, params):
#   # save checkpoints
#   orbax_checkpointer = ocp.PyTreeCheckpointer()
#   save_args = orbax_utils.save_args_from_target(params)
#   model_path = os.path.join(output_dir, "model_ckpt_{}".format(current_step)) 
#   orbax_checkpointer.save(model_path, params, force=True, save_args=save_args)


train_fn = functools.partial(
    ppo.train, **rl_config, policy_params_fn=policy_params_fn)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = -400, 0
def progress(num_steps, metrics):
  times.append(datetime.now())
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