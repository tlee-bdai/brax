import time
import mediapy as media
import os
import json

import jax
from jax import numpy as jp
import numpy as np
from matplotlib import pyplot as plt

import mujoco
from mujoco import mjx
from copy import deepcopy as dc

from brax import envs
from brax.io import html, mjcf, model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

model_dir = "./output/urm2arm_binary_no_actionclip_noobsnormal_eps0dot1_damp1_xyt_bs0dot1_d0dot24_obsqpos_constvel6kp5_spectral"
ckpt = 149*163840
config = json.load(open(os.path.join(model_dir, "config.json")))
model_path = os.path.join(model_dir, "model_ckpt_{}".format(ckpt))

params = model.load_params(model_path)
# print(params)



eval_env = envs.get_environment(config["env_name"])
jit_reset = jax.jit(eval_env.reset, device = jax.devices('cpu')[0])
jit_step = jax.jit(eval_env.step, device = jax.devices('cpu')[0])

# initialize the state
rng = jax.random.PRNGKey(0)
state = eval_env.reset(rng)
rollout = [state.pipeline_state]


normalize = lambda x, y: x
if config["normalize_observations"]:
    normalize = running_statistics.normalize
ppo_network = ppo_networks.make_ppo_networks(
      state.obs.shape[-1],
      dim_c = eval_env.discrete_action_size,
      dim_n = eval_env.cont_action_size,
      preprocess_observations_fn=normalize,
      spectral_norm_actor = config["spectral_norm_actor"]
      )
make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_inference_fn((params[0], params[1].policy))
jit_inference_fn = jax.jit(inference_fn, device = jax.devices('cpu')[0])



# grab a trajectory
n_steps = 40
render_every = 1

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)
  # print(state.pipeline_state.qpos, state.reward)
  print(ctrl, state.reward, state.pipeline_state.ncon)
  # print(state.done)
  # if state.done:
  #   break
print("done")
media.write_video(os.path.join(model_dir, "video_{}.mp4".format(ckpt)), eval_env.render(rollout[::render_every], camera='top'), fps=1.0 / eval_env.dt / render_every)
# media.show_video(eval_env.render(rollout[::render_every], camera='top'), fps=1.0 / eval_env.dt / render_every)