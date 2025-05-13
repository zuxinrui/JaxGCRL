import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp

from brax import base

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath

from typing import Tuple

import wandb
import xmltodict
import xml.etree.ElementTree as ET

import types



def adapt_xml(file, design=None):
    with open(file, 'r') as fd:
        xml_string = fd.read()
    if design is None:
        bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=2.0, size=6)
    else:
        bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
    height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r + .07 * ffo_r)
    height *= 2.0 + 0.01

    xml_dict = xmltodict.parse(xml_string)

    torso = None
    for body in xml_dict['mujoco']['worldbody']['body']:
        if body.get('@name') == 'torso':
            torso = body
            break

    if torso is None:
        raise ValueError("Could not find torso body in XML")

    # Update torso position
    torso['@pos'] = f"0 0 {height}"

    # Update leg components
    # Back leg
    back_thigh = torso['body'][0]  # bthigh
    back_shin = back_thigh['body']  # bshin
    back_foot = back_shin['body']  # bfoot

    # Front leg
    front_thigh = torso['body'][1]  # fthigh
    front_shin = front_thigh['body']  # fshin
    front_foot = front_shin['body']  # ffoot

    # Back leg modifications
    back_thigh['geom']['@pos'] = f"{.1 * bth_r} 0 {-.13 * bth_r}"
    back_thigh['geom']['@size'] = f"0.046 {.145 * bth_r}"
    back_thigh['body']['@pos'] = f"{.16 * bth_r} 0 {-.25 * bth_r}"

    back_shin['geom']['@pos'] = f"{-.14 * bsh_r} 0 {-.07 * bsh_r}"
    back_shin['geom']['@size'] = f"0.046 {.15 * bsh_r}"
    back_shin['body']['@pos'] = f"{-.28 * bsh_r} 0 {-.14 * bsh_r}"

    back_foot['geom']['@pos'] = f"{.03 * bfo_r} 0 {-.097 * bfo_r}"
    back_foot['geom']['@size'] = f"0.046 {.094 * bfo_r}"

    # Front leg modifications
    front_thigh['geom']['@pos'] = f"{-.07 * fth_r} 0 {-.12 * fth_r}"
    front_thigh['geom']['@size'] = f"0.046 {.133 * fth_r}"
    front_thigh['body']['@pos'] = f"{-.14 * fth_r} 0 {-.24 * fth_r}"

    front_shin['geom']['@pos'] = f"{.065 * fsh_r} 0 {-.09 * fsh_r}"
    front_shin['geom']['@size'] = f"0.046 {.106 * fsh_r}"
    front_shin['body']['@pos'] = f"{.13 * fsh_r} 0 {-.18 * fsh_r}"

    front_foot['geom']['@pos'] = f"{.045 * ffo_r} 0 {-.07 * ffo_r}"
    front_foot['geom']['@size'] = f"0.046 {.07 * ffo_r}"

    xml_string = xmltodict.unparse(xml_dict, pretty=True)
    with open(file, 'w') as fd:
        fd.write(xml_string)
    return bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r


def add_obstacles(path, write_path, height, width, spacing, n_obstacles):

    with open(path, 'r') as fd:
        xml_string = fd.read()
    root = ET.fromstring(xml_string)
    worldbody = root.find('worldbody')
    for i in range(n_obstacles):
        x_pos = (i + 1) * spacing

        # Create obstacle body
        body = ET.SubElement(worldbody, 'body')
        body.set('name', f'obstacle_{i}')
        body.set('pos', f'{x_pos} 0 {height / 2}')

        # Add geom to body
        geom = ET.SubElement(body, 'geom')
        geom.set('type', 'box')
        geom.set('size', f'{width / 2} 1.0 {height / 2}')
        geom.set('rgba', '0.7 0.5 0.3 1.0')
        geom.set('mass', '0')
        # geom.set('material', '')  # no difference on the speed of training
        geom.set('contype', '1')
        geom.set('conaffinity', '1')

    # Convert back to string
    new_xml = ET.tostring(root, encoding='unicode')
    with open(write_path, 'w') as fd:
        fd.write(new_xml)


"""
mujoco half_cheetah in brax:
"""
class HalfcheetahWithObstacles(PipelineEnv):

    def __init__(
            self,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            # New obstacle parameters
            obstacle_height: float = 0.2,
            obstacle_width: float = 0.2,
            obstacle_spacing: float = 2.0,
            n_obstacles: int = 5,
            write_path='./logs/half_cheetah_modified.xml',
            design=None,
            backend='generalized',
            **kwargs
    ):
        path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'

        n_frames = 5

        # Create and add obstacles
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = 1., 1., 1., 1., 1., 1.
        sys = self._change_env(path, write_path, obstacle_height, obstacle_width,
                               obstacle_spacing, n_obstacles, design=design)

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.003125})
            n_frames = 16
            gear = jp.array([120, 90, 60, 120, 100, 100])
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._obstacle_height = obstacle_height
        self._obstacle_width = obstacle_width
        self._obstacle_spacing = obstacle_spacing
        self._n_obstacles = n_obstacles

    def _change_env(self, path, write_path, height, width, spacing, n_obstacles, design=None):

        if n_obstacles > 0:
            add_obstacles(path, write_path, height, width, spacing, n_obstacles)
        else:
            write_path = path

        with open(write_path, 'r') as fd:
            xml_string = fd.read()
        if design is None:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=2.0, size=6)
        else:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
        height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r + .07 * ffo_r)
        height *= 2.0 + 0.01

        xml_dict = xmltodict.parse(xml_string)

        torso = None
        for body in xml_dict['mujoco']['worldbody']['body']:
            if body.get('@name') == 'torso':
                torso = body
                break

        if torso is None:
            raise ValueError("Could not find torso body in XML")

        # Update torso position
        torso['@pos'] = f"0 0 {height}"

        # Update leg components
        # Back leg
        back_thigh = torso['body'][0]  # bthigh
        back_shin = back_thigh['body']  # bshin
        back_foot = back_shin['body']  # bfoot

        # Front leg
        front_thigh = torso['body'][1]  # fthigh
        front_shin = front_thigh['body']  # fshin
        front_foot = front_shin['body']  # ffoot

        # Back leg modifications
        back_thigh['geom']['@pos'] = f"{.1 * bth_r} 0 {-.13 * bth_r}"
        back_thigh['geom']['@size'] = f"0.046 {.145 * bth_r}"
        back_thigh['body']['@pos'] = f"{.16 * bth_r} 0 {-.25 * bth_r}"

        back_shin['geom']['@pos'] = f"{-.14 * bsh_r} 0 {-.07 * bsh_r}"
        back_shin['geom']['@size'] = f"0.046 {.15 * bsh_r}"
        back_shin['body']['@pos'] = f"{-.28 * bsh_r} 0 {-.14 * bsh_r}"

        back_foot['geom']['@pos'] = f"{.03 * bfo_r} 0 {-.097 * bfo_r}"
        back_foot['geom']['@size'] = f"0.046 {.094 * bfo_r}"

        # Front leg modifications
        front_thigh['geom']['@pos'] = f"{-.07 * fth_r} 0 {-.12 * fth_r}"
        front_thigh['geom']['@size'] = f"0.046 {.133 * fth_r}"
        front_thigh['body']['@pos'] = f"{-.14 * fth_r} 0 {-.24 * fth_r}"

        front_shin['geom']['@pos'] = f"{.065 * fsh_r} 0 {-.09 * fsh_r}"
        front_shin['geom']['@size'] = f"0.046 {.106 * fsh_r}"
        front_shin['body']['@pos'] = f"{.13 * fsh_r} 0 {-.18 * fsh_r}"

        front_foot['geom']['@pos'] = f"{.045 * ffo_r} 0 {-.07 * ffo_r}"
        front_foot['geom']['@size'] = f"0.046 {.07 * ffo_r}"

        xml_string = xmltodict.unparse(xml_dict, pretty=True)
        with open(write_path, 'w') as fd:
            fd.write(xml_string)

        # Modify the XML file to change the morphology
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r
        print('Changing morphology', self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r)
        sys = mjcf.load(write_path)

        return sys

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        x_velocity = (
                             pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
                     ) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        reward = forward_reward - ctrl_cost
        state.metrics.update(
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jp.concatenate((position, velocity))


"""
mujoco half_cheetah in brax:
"""
class HalfcheetahMorphTasks(PipelineEnv):

    def __init__(
            self,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            # New obstacle parameters
            task_id: str = 'forward',
            obstacle_height: float = 0.2,
            obstacle_width: float = 0.2,
            obstacle_spacing: float = 2.0,
            n_obstacles: int = 5,
            write_path='./logs/half_cheetah_modified.xml',
            design=None,
            backend='generalized',
            **kwargs
    ):
        path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'

        n_frames = 5

        # Create and add obstacles
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = 1., 1., 1., 1., 1., 1.
        sys = self._change_env(path, write_path, obstacle_height, obstacle_width,
                               obstacle_spacing, n_obstacles, design=design)

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.003125})
            n_frames = 16
            gear = jp.array([120, 90, 60, 120, 100, 100])
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._obstacle_height = obstacle_height
        self._obstacle_width = obstacle_width
        self._obstacle_spacing = obstacle_spacing
        self._n_obstacles = n_obstacles

        # store the task name once
        self._task = task_id.lower()  # forward, backward, …

    def _change_env(self, path, write_path, height, width, spacing, n_obstacles, design=None):

        if n_obstacles > 0:
            add_obstacles(path, write_path, height, width, spacing, n_obstacles)
        else:
            write_path = path

        with open(write_path, 'r') as fd:
            xml_string = fd.read()
        if design is None:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=2.0, size=6)
        else:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
        height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r + .07 * ffo_r)
        height *= 2.0 + 0.01

        xml_dict = xmltodict.parse(xml_string)

        torso = None
        for body in xml_dict['mujoco']['worldbody']['body']:
            if body.get('@name') == 'torso':
                torso = body
                break

        if torso is None:
            raise ValueError("Could not find torso body in XML")

        # Update torso position
        torso['@pos'] = f"0 0 {height}"

        # Update leg components
        # Back leg
        back_thigh = torso['body'][0]  # bthigh
        back_shin = back_thigh['body']  # bshin
        back_foot = back_shin['body']  # bfoot

        # Front leg
        front_thigh = torso['body'][1]  # fthigh
        front_shin = front_thigh['body']  # fshin
        front_foot = front_shin['body']  # ffoot

        # Back leg modifications
        back_thigh['geom']['@pos'] = f"{.1 * bth_r} 0 {-.13 * bth_r}"
        back_thigh['geom']['@size'] = f"0.046 {.145 * bth_r}"
        back_thigh['body']['@pos'] = f"{.16 * bth_r} 0 {-.25 * bth_r}"

        back_shin['geom']['@pos'] = f"{-.14 * bsh_r} 0 {-.07 * bsh_r}"
        back_shin['geom']['@size'] = f"0.046 {.15 * bsh_r}"
        back_shin['body']['@pos'] = f"{-.28 * bsh_r} 0 {-.14 * bsh_r}"

        back_foot['geom']['@pos'] = f"{.03 * bfo_r} 0 {-.097 * bfo_r}"
        back_foot['geom']['@size'] = f"0.046 {.094 * bfo_r}"

        # Front leg modifications
        front_thigh['geom']['@pos'] = f"{-.07 * fth_r} 0 {-.12 * fth_r}"
        front_thigh['geom']['@size'] = f"0.046 {.133 * fth_r}"
        front_thigh['body']['@pos'] = f"{-.14 * fth_r} 0 {-.24 * fth_r}"

        front_shin['geom']['@pos'] = f"{.065 * fsh_r} 0 {-.09 * fsh_r}"
        front_shin['geom']['@size'] = f"0.046 {.106 * fsh_r}"
        front_shin['body']['@pos'] = f"{.13 * fsh_r} 0 {-.18 * fsh_r}"

        front_foot['geom']['@pos'] = f"{.045 * ffo_r} 0 {-.07 * ffo_r}"
        front_foot['geom']['@size'] = f"0.046 {.07 * ffo_r}"

        xml_string = xmltodict.unparse(xml_dict, pretty=True)
        with open(write_path, 'w') as fd:
            fd.write(xml_string)

        # Modify the XML file to change the morphology
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r
        print('Changing morphology', self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r)
        sys = mjcf.load(write_path)

        return sys

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    # ---------------------------------------------------------------------
    # INSERT 2  :  modify step to add task-specific reward/termination
    # ---------------------------------------------------------------------
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        ps0 = state.pipeline_state
        assert ps0 is not None
        ps = self.pipeline_step(ps0, action)

        # base velocity and cost
        x_vel = (ps.x.pos[0, 0] - ps0.x.pos[0, 0]) / self.dt
        fwd_rew = self._forward_reward_weight * x_vel
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        # ------------------- task-specific reward --------------------
        task_rew = 0.0
        task_done = False

        if self._task == 'forward':
            task_rew = 0.0  # nothing extra
        elif self._task == 'backward':
            task_rew = -2.0 * x_vel  # encourage neg vel
        elif self._task == 'high_jump':
            task_rew = ps.x.pos[0, 2] - x_vel  # torso height (z)
        elif self._task == 'forward_long_jump':
            task_rew = jp.maximum(0.0, x_vel) * 2.0
        elif self._task == 'backward_long_jump':
            task_rew = jp.maximum(0.0, -x_vel) * 2.0
        elif self._task == 'backflip':
            pitch = self._torso_pitch(ps)
            task_rew = -pitch - x_vel
        elif self._task == 'forward_flip':
            pitch = self._torso_pitch(ps)
            task_rew = pitch - x_vel

        elif self._task == 'forward_long_jumpv2':
            # airborne flag this step
            airborne_now = self._is_airborne(ps)
            airborne_prev = self._is_airborne(ps0)
            new_air_time = jp.where(airborne_now, 1.0, 0.0)
            # accumulate horiz displacement only while airborne
            horiz_disp = jp.where(airborne_now, ps.x.pos[0, 0] - ps0.x.pos[0, 0], 0.)
            # small bonus when we first touch ground again (landing)
            landed = jp.logical_and(~airborne_now, airborne_prev)
            landing_bonus = jp.where(landed, 5.0, 0.0)
            task_rew = 3.0 * horiz_disp + new_air_time + landing_bonus - x_vel

        elif self._task == 'backflipv2':
            pitch = self._torso_pitch(ps)
            pitch_rate = (pitch - self._torso_pitch(ps0)) / self.dt
            # encourage negative pitch velocity (backward rotation)
            task_rew = -0.9 * pitch_rate - x_vel - 0.1 * pitch
            # bonus when completed > 330°
            # flipped = jp.abs(pitch) > (2.0 * jp.pi - 0.3)
            # task_rew += jp.where(flipped, 15.0, 0.0)
            # # encourage negative horizontal velocity so flip happens in-place
            # task_rew += jp.maximum(0.0, -x_vel)
            # task_done = bool(flipped)

        elif self._task == 'forward_flipv2':
            pitch = self._torso_pitch(ps)
            pitch_rate = (pitch - self._torso_pitch(ps0)) / self.dt
            # encourage positive pitch velocity (forward rotation)
            task_rew = 0.9 * pitch_rate - x_vel + 0.1 * pitch
            # flipped = jp.abs(pitch) > (2.0 * jp.pi - 0.3)
            # task_rew += jp.where(flipped, 15.0, 0.0)
            # # bonus for forward displacement while flipping
            # task_rew += jp.maximum(0.0, x_vel)
            # task_done = bool(flipped)
        else:
            raise ValueError(f'Unknown task id {self._task}')
        # ----------------------------------------------------------------

        obs = self._get_obs(ps)
        reward = fwd_rew - ctrl_cost + task_rew
        # done = jp.asarray(task_done) | state.done  # keep brax flag

        # state.metrics.update(
        #     x_position=pipeline_state.x.pos[0, 0],
        #     x_velocity=x_velocity,
        #     reward_run=forward_reward,
        #     reward_ctrl=-ctrl_cost,
        # )
        #
        # return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

        state.metrics.update(
            x_position=ps.x.pos[0, 0],
            x_velocity=x_vel,
            reward_run=fwd_rew,
            reward_ctrl=-ctrl_cost,
            # reward_task=task_rew,
        )
        return state.replace(pipeline_state=ps,
                             obs=obs,
                             reward=reward,
                             )

    # small helper for pitch
    def _torso_pitch(self, ps: base.State):
        """Return torso pitch (rotation around y-axis)."""
        qw, qx, qy, qz = ps.x.rot[0]
        t2 = 2 * (qw * qy - qz * qx)
        t2 = jp.clip(t2, -1.0, 1.0)
        return jp.arcsin(t2)

    # --------------------------------------------------------------
    # helper utilities (place them in your class)
    # --------------------------------------------------------------
    def _is_airborne(self, ps: base.State, thresh: float = 0.05):
        """Returns bool array [,] if all foot geoms are higher than thresh."""
        # indices 2 and 5 are the two feet in default half-cheetah
        z_back = ps.x.pos[2, 2]
        z_front = ps.x.pos[5, 2]
        return jp.logical_and(z_back > thresh, z_front > thresh)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jp.concatenate((position, velocity))


"""
mujoco hopper in brax:
"""
class Hopper(PipelineEnv):

    def __init__(
        self,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float('inf')),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        backend='generalized',
        **kwargs
    ):
        path = epath.resource_path('brax') / 'envs/assets/hopper.xml'
        # Create and add obstacles
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = 1., 1., 1., 1., 1., 1.
        sys = self._change_env(path, write_path, obstacle_height, obstacle_width,
                               obstacle_spacing, n_obstacles, design=design)

        n_frames = 4
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def _change_env(self, path, write_path, height, width, spacing, n_obstacles, design=None):

        add_obstacles(path, write_path, height, width, spacing, n_obstacles)

        with open(write_path, 'r') as fd:
            xml_string = fd.read()
        if design is None:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=2.0, size=6)
        else:
            bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
        height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r + .07 * ffo_r)
        height *= 2.0 + 0.01

        xml_dict = xmltodict.parse(xml_string)

        torso = None
        for body in xml_dict['mujoco']['worldbody']['body']:
            if body.get('@name') == 'torso':
                torso = body
                break

        if torso is None:
            raise ValueError("Could not find torso body in XML")

        # Update torso position
        torso['@pos'] = f"0 0 {height}"

        # Update leg components
        # Back leg
        back_thigh = torso['body'][0]  # bthigh
        back_shin = back_thigh['body']  # bshin
        back_foot = back_shin['body']  # bfoot

        # Front leg
        front_thigh = torso['body'][1]  # fthigh
        front_shin = front_thigh['body']  # fshin
        front_foot = front_shin['body']  # ffoot

        # Back leg modifications
        back_thigh['geom']['@pos'] = f"{.1 * bth_r} 0 {-.13 * bth_r}"
        back_thigh['geom']['@size'] = f"0.046 {.145 * bth_r}"
        back_thigh['body']['@pos'] = f"{.16 * bth_r} 0 {-.25 * bth_r}"

        back_shin['geom']['@pos'] = f"{-.14 * bsh_r} 0 {-.07 * bsh_r}"
        back_shin['geom']['@size'] = f"0.046 {.15 * bsh_r}"
        back_shin['body']['@pos'] = f"{-.28 * bsh_r} 0 {-.14 * bsh_r}"

        back_foot['geom']['@pos'] = f"{.03 * bfo_r} 0 {-.097 * bfo_r}"
        back_foot['geom']['@size'] = f"0.046 {.094 * bfo_r}"

        # Front leg modifications
        front_thigh['geom']['@pos'] = f"{-.07 * fth_r} 0 {-.12 * fth_r}"
        front_thigh['geom']['@size'] = f"0.046 {.133 * fth_r}"
        front_thigh['body']['@pos'] = f"{-.14 * fth_r} 0 {-.24 * fth_r}"

        front_shin['geom']['@pos'] = f"{.065 * fsh_r} 0 {-.09 * fsh_r}"
        front_shin['geom']['@size'] = f"0.046 {.106 * fsh_r}"
        front_shin['body']['@pos'] = f"{.13 * fsh_r} 0 {-.18 * fsh_r}"

        front_foot['geom']['@pos'] = f"{.045 * ffo_r} 0 {-.07 * ffo_r}"
        front_foot['geom']['@size'] = f"0.046 {.07 * ffo_r}"

        xml_string = xmltodict.unparse(xml_dict, pretty=True)
        with open(write_path, 'w') as fd:
            fd.write(xml_string)

        # Modify the XML file to change the morphology
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r
        print('Changing morphology', self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r)
        sys = mjcf.load(write_path)

        return sys


    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        x_velocity = (
            pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
        ) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity

        z, angle = pipeline_state.x.pos[0, 2], pipeline_state.q[2]
        state_vec = jp.concatenate([pipeline_state.q[2:], pipeline_state.qd])
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        min_state, max_state = self._healthy_state_range
        is_healthy = jp.all(
            jp.logical_and(min_state < state_vec, state_vec < max_state)
        )
        is_healthy &= jp.logical_and(min_z < z, z < max_z)
        is_healthy &= jp.logical_and(min_angle < angle, angle < max_angle)
        if self._terminate_when_unhealthy:
          healthy_reward = self._healthy_reward
        else:
          healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward,
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        position = position.at[1].set(pipeline_state.x.pos[0, 2])
        velocity = jp.clip(pipeline_state.qd, -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jp.concatenate((position, velocity))


# ------------------------------------------------------------
#   main class
# ------------------------------------------------------------
class HalfcheetahMML(PipelineEnv):
  """Half-Cheetah with obstacles + morphology mutation + multi-task."""

  # -------------- XML / morphology code  (unchanged) -----------------
  def __init__(self,
               # control & noise
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True,
               # task switch
               task_id=None,
               # obstacle params
               obstacle_height: float = 0.2,
               obstacle_width: float = 0.2,
               obstacle_spacing: float = 2.0,
               n_obstacles: int = 5,
               write_path='./logs/half_cheetah_modified.xml',
               design=None,
               backend='generalized',
               **kwargs):

    self._task = task_id        # save once

    path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'
    self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = design
    sys = self._change_env(path, write_path,
                           obstacle_height, obstacle_width,
                           obstacle_spacing, n_obstacles,
                           design=design)

    # backend tweaks
    n_frames = 5
    if backend in ('spring', 'positional'):
      sys = sys.tree_replace({'opt.timestep': 0.003125})
      n_frames = 16
    gear = jp.array([120, 90, 60, 120, 100, 100])
    sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
    super().__init__(sys=sys, backend=backend, **kwargs)

    # save hyper-params
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = \
        exclude_current_positions_from_observation

  # ----------------- terrain + morphology edits ----------------------
  def _change_env(self, path, write_path, height, width, spacing, n_obstacles, design=None):

      add_obstacles(path, write_path, height, width, spacing, n_obstacles)

      with open(write_path, 'r') as fd:
          xml_string = fd.read()
      if design is None:
          bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5, high=2.0, size=6)
      else:
          bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design
      height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r + .106 * fsh_r + .07 * ffo_r)
      height *= 2.0 + 0.01

      xml_dict = xmltodict.parse(xml_string)

      torso = None
      bodies = xml_dict['mujoco']['worldbody'].get('body', [])
      # If there was only one <body> tag, xmltodict gives us a dict (or even a str),
      # so force it into a list
      if not isinstance(bodies, list):
          bodies = [bodies]

      for body in bodies:
          # skip over any pure‐string entries
          if not isinstance(body, dict):
              continue
          if body.get('@name') == 'torso':
              torso = body
              break

      if torso is None:
          raise ValueError("Could not find torso body in XML")

      # Update torso position
      torso['@pos'] = f"0 0 {height}"

      # Update leg components
      # Back leg
      back_thigh = torso['body'][0]  # bthigh
      back_shin = back_thigh['body']  # bshin
      back_foot = back_shin['body']  # bfoot

      # Front leg
      front_thigh = torso['body'][1]  # fthigh
      front_shin = front_thigh['body']  # fshin
      front_foot = front_shin['body']  # ffoot

      # Back leg modifications
      back_thigh['geom']['@pos'] = f"{.1 * bth_r} 0 {-.13 * bth_r}"
      back_thigh['geom']['@size'] = f"0.046 {.145 * bth_r}"
      back_thigh['body']['@pos'] = f"{.16 * bth_r} 0 {-.25 * bth_r}"

      back_shin['geom']['@pos'] = f"{-.14 * bsh_r} 0 {-.07 * bsh_r}"
      back_shin['geom']['@size'] = f"0.046 {.15 * bsh_r}"
      back_shin['body']['@pos'] = f"{-.28 * bsh_r} 0 {-.14 * bsh_r}"

      back_foot['geom']['@pos'] = f"{.03 * bfo_r} 0 {-.097 * bfo_r}"
      back_foot['geom']['@size'] = f"0.046 {.094 * bfo_r}"

      # Front leg modifications
      front_thigh['geom']['@pos'] = f"{-.07 * fth_r} 0 {-.12 * fth_r}"
      front_thigh['geom']['@size'] = f"0.046 {.133 * fth_r}"
      front_thigh['body']['@pos'] = f"{-.14 * fth_r} 0 {-.24 * fth_r}"

      front_shin['geom']['@pos'] = f"{.065 * fsh_r} 0 {-.09 * fsh_r}"
      front_shin['geom']['@size'] = f"0.046 {.106 * fsh_r}"
      front_shin['body']['@pos'] = f"{.13 * fsh_r} 0 {-.18 * fsh_r}"

      front_foot['geom']['@pos'] = f"{.045 * ffo_r} 0 {-.07 * ffo_r}"
      front_foot['geom']['@size'] = f"0.046 {.07 * ffo_r}"

      xml_string = xmltodict.unparse(xml_dict, pretty=True)
      with open(write_path, 'w') as fd:
          fd.write(xml_string)

      # Modify the XML file to change the morphology
      self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r
      print('Changing morphology', self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r)
      sys = mjcf.load(write_path)

      return sys

  # --------------------- JAX helpers ---------------------------------
  @staticmethod
  def _torso_pitch(ps: base.State):
    """Compute torso pitch (y-axis)."""
    qw, qx, qy, qz = ps.x.rot[0]
    t2 = 2 * (qw*qy - qz*qx)
    t2 = jp.clip(t2, -1.0, 1.0)
    return jp.arcsin(t2)

  @staticmethod
  def _is_airborne(ps: base.State, thresh=0.05):
    """Both feet higher than thresh."""
    z_back  = ps.x.pos[2, 2]   # geom 2
    z_front = ps.x.pos[5, 2]   # geom 5
    return jp.logical_and(z_back > thresh, z_front > thresh)

  # -------------------------- reset ----------------------------------
  def reset(self, rng: jax.Array) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),),
                                                minval=low, maxval=hi)
    qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
    ps = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(ps)
    # zero = jp.zeros(())
    zero_b = jp.zeros((), dtype=jp.bool_)  # <── boolean zero
    metrics = dict(x_position=jp.zeros(()),
                   x_velocity=jp.zeros(()),
                   cum_pitch=jp.zeros(()),
                   reward_ctrl=jp.zeros(()),
                   reward_run=jp.zeros(()),
                   reward_task=jp.zeros(()),  # ← add this line
                   reward=jp.zeros(()),
                   t=jp.zeros(()))
    return State(ps, obs, reward=jp.zeros(()),
                 done=jp.zeros(()),  # <- use zero_b
                 metrics=metrics)

  # ------------------------- step ------------------------------------
  def step(self, state: State, action: jax.Array) -> State:
    ps0 = state.pipeline_state
    ps  = self.pipeline_step(ps0, action)

    # base run + cost
    x_vel = (ps.x.pos[0, 0] - ps0.x.pos[0, 0]) / self.dt
    run_rew  = self._forward_reward_weight * x_vel
    ctrl_pen = self._ctrl_cost_weight * jp.sum(jp.square(action))

    # ---------- task-specific reward & termination -------------------
    task_rew  = 0.0
    task_done = False
    task = self._task
    cum_pitch = 0.0

    if task == 'forward':
      pass                                                 # only run reward
    elif task == 'backward':
      task_rew = -2.0 * x_vel
    elif task == 'high_jump':
      task_rew = ps.x.pos[0, 2]                   # height
    elif task == 'forward_long_jump':
      air_now  = self._is_airborne(ps)
      air_prev = self._is_airborne(ps0)
      horiz = jp.where(air_now, ps.x.pos[0, 0] - ps0.x.pos[0, 0], 0.0)
      landed = jp.logical_and(~air_now, air_prev)
      task_rew = 3.0 * horiz + 1.0 * air_now + jp.where(landed, 5.0, 0.0)
    elif task == 'backward_long_jump':
      air_now  = self._is_airborne(ps)
      air_prev = self._is_airborne(ps0)
      horiz = jp.where(air_now, ps0.x.pos[0, 0] - ps.x.pos[0, 0], 0.0)
      landed = jp.logical_and(~air_now, air_prev)
      task_rew = 3.0 * horiz + 1.0 * air_now + jp.where(landed, 5.0, 0.0)
    elif task == 'backflip':
      pitch     = self._torso_pitch(ps)
      pitch_dot = (pitch - self._torso_pitch(ps0)) / self.dt
      task_rew  = -0.1 * pitch_dot - x_vel - 0.1 * pitch
      flipped   = jp.abs(pitch) > (2*jp.pi - 0.3)
      task_rew += jp.where(flipped, 15.0, 0.0)
      task_done = flipped
    elif task == 'forward_flip':
      pitch     = self._torso_pitch(ps)
      pitch_dot = (pitch - self._torso_pitch(ps0)) / self.dt
      task_rew  = 0.1 * pitch_dot - x_vel + 0.1 * pitch
      flipped   = jp.abs(pitch) > (2*jp.pi - 0.3)
      task_rew += jp.where(flipped, 15.0, 0.0)
      task_done = flipped
    elif isinstance(task, types.FunctionType):
        # task is a function
        task_rew, task_done, cum_pitch = task(ps, state, self.dt)
    else:
      raise ValueError(f"Unknown task")

    # ---------------------------------------------------------------
    total_rew = run_rew - ctrl_pen + task_rew
    # done      = state.done | jp.asarray(task_done)

    # task_done_flag = jp.asarray(task_done, dtype=jp.bool_)
    # done = jp.logical_or(state.done, task_done_flag)  # <── boolean OR
    # … your existing code that computes `done` as a bool array …
    task_done_flag = jp.asarray(task_done, dtype=jp.bool_)
    done_bool = jp.logical_or(state.done, task_done_flag)

    # force it back to the same dtype Brax expects
    done = done_bool.astype(jp.float32)

    # update metrics
    t = state.metrics['t'] + 1
    metrics = dict(x_position=ps.x.pos[0, 0],
                   x_velocity=x_vel,
                   cum_pitch=cum_pitch,
                   reward_run=run_rew,
                   reward_ctrl=-ctrl_pen,
                   reward_task=task_rew,
                   reward=total_rew,
                   t=t)

    return state.replace(pipeline_state=ps,
                         obs=self._get_obs(ps),
                         reward=total_rew,
                         done=done,
                         metrics=metrics)

  # ----------------------- observation ------------------------------
  def _get_obs(self, ps: base.State):
    q, qd = ps.q, ps.qd
    if self._exclude_current_positions_from_observation:
      q = q[1:]
    return jp.concatenate([q, qd], axis=0)
