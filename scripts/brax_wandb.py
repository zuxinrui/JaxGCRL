#!/usr/bin/env python3
import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output

import brax
import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.sac import train as sac

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath

import wandb
import xmltodict

from notebooks.ppo import ppo_train
from notebooks.env import HalfcheetahWithObstacles

# Import the JaxGCRL  environment:
from jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

run = wandb.init(
    project='test',
    group='vu',
    name='zuxinrui',
    mode="online",
)
# env = envs.get_environment(env_name='pusher', backend='mjx')
env = ArmPushEasy(backend='mjx')  # Use mjx backend for better rendering
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), [state.pipeline_state], height=1024)
wandb.log({"env render": wandb.Html(url)})

# env = HalfcheetahWithObstacles(
#     obstacle_height=0.4,  # (0.2 - 0.5)
#     obstacle_width=0.2,  # (0.1 - 0.5)
#     obstacle_spacing=1.0,  # (0.5 - 2.0)
#     n_obstacles=10,  # 10
#     design=None,
#     backend='spring',
# )
# state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
# url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), [state.pipeline_state], height=1024)
# with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
#     file.write(url)
# wandb.log({"env render": wandb.Html(url)})

episode_length = 1000

# # for halfcheetah:
# train_fn = functools.partial(
#     ppo_train,
#     num_timesteps=10_000_000,
#     num_evals=3,
#     reward_scaling=1,
#     episode_length=episode_length,
#     normalize_observations=True,
#     action_repeat=1,
#     unroll_length=20,
#     num_minibatches=32,
#     num_updates_per_batch=8,
#     discounting=0.95,
#     learning_rate=3e-4,
#     entropy_cost=0.001,
#     num_envs=4096,  # 2048 on 4070 ti s is the fastest  p.s.: num_envs must be divisible by n_batch * batch_size (1+ times per env simulation in the batch)
#     batch_size=128,
#     seed=3,
# )

# for pusher:
train_fn = functools.partial(
    ppo_train,
    num_timesteps=1_000_000,
    num_evals=3,
    reward_scaling=5,
    episode_length=episode_length,  # 1000
    normalize_observations=True,
    action_repeat=1,
    unroll_length=30,
    num_minibatches=16,
    num_updates_per_batch=8,
    discounting=0.95,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=512,  # 2048 on 4070 ti s is the fastest  p.s.: num_envs must be divisible by n_batch * batch_size (1+ times per env simulation in the batch)
    batch_size=512,
    seed=3,
)

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics, params, make_policy):
    render(make_policy, params, env, './logs/htmls/', 'halfcheetah', num_steps, metrics)

def render(make_policy, params, env, exp_dir, exp_name, num_steps, metrics=None):
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rollout = []
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key)
    state = jit_env_reset(rng=subkey)
    for i in range(episode_length):  # 1000 = 50s
        rollout.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = jit_policy(state.obs, subkey)  # Policy requires batched dimension
        # action = action[0]  # Remove batch dimension
        state = jit_env_step(state, action)
        # if i % 1000 == 0:
        #     key, subkey = jax.random.split(key)
        #     state = jit_env_reset(rng=subkey)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
        file.write(url)
    wandb.log({
        "video": wandb.Html(url),
        'training/reward': metrics['eval/episode_reward'],
    })

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
print(f'time overall: {times[-1] - times[0]}')
# n minibatch really doesn't matter too much!
inference_fn = make_inference_fn(params)
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(200):
  rollout.append(state.pipeline_state)
  act_rng, rng = jax.random.split(rng)
  act, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_env_step(state, act)

url = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
wandb.log({"video": wandb.Html(url)})
# run.finish()