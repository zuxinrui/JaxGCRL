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

import wandb
import xmltodict



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
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = 1., 1., 1., 1., 1., 1.

    def _change_env(self, path, write_path, height, width, spacing, n_obstacles, design=None):
        with open(path, 'r') as fd:
            xml_string = fd.read()
        import xml.etree.ElementTree as ET
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

        # Modify the XML file to change the morphology
        self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r = adapt_xml(write_path, design)
        print('Changing morphology', self.bth_r, self.bsh_r, self.bfo_r, self.fth_r, self.fsh_r, self.ffo_r)
        # # Save the new design parameters
        # with open('design.txt', 'w') as fd:
        #     fd.write(f'{bth_r} {bsh_r} {bfo_r} {fth_r} {fsh_r} {ffo_r}')

        # Load modified XML (should not implicitly change the system after loaded)
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
