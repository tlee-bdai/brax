from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

class TwoArmBinary(PipelineEnv):

  def __init__(
      self,
      forward_reward_weight=1.25,
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(1.0, 2.0),
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,
  ):
    d = 0.15
    l1, l2 = 0.14, 0.14
    bs = 0.15
    q_limit = 2.5
    savevideo =False

    xml = '''
    <mujoco>
        <compiler angle="radian" />
        <option  timestep="0.001" gravity="0 0 0"/>
        <default>
          <geom solref="0.002 1" contype="1" conaffinity="1" friction="0.3 0.005 0.0001" />
          <joint  armature="0.01" range="-{5} {5}"/>
        </default>
        <visual>
          <map stiffness="10000"/>
          <global offheight="1280" offwidth="1280"/>
        </visual>
        <asset>
          <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
          <texture name="body" type="cube" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1"/>
          <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
          <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
          <material name="grid" texture="grid" texrepeat="4 4" texuniform="false" reflectance=".0"/>
        </asset>

        <worldbody>
            <geom name="floor" type="plane" size=".4 .4 0.01" material="grid" pos="0 0.2 -0.025"/>
            <camera name="top" pos="0 0 1.0" resolution="1280 1280"/>
            <body name="robot">
            <geom name="base" type="capsule" size="0.01" fromto="-{3} -0.02 0 {3} -0.02 0" />
            <body name="R1" pos="{0} 0 0.0">
                <geom type="capsule" size="0.01" fromto="0 0 0 0 {1} 0" rgba="0 0 0 1"/>
                <joint name="R1j" type="hinge" pos="0 0 0" axis="0 0 1" />
                <body name="R2" pos="0 {1} 0">
                    <geom type="capsule" size="0.01" fromto="0 0 0 0 {2} 0" />
                    <joint name="R2j" type="hinge" pos="0 0 0" axis="0 0 1" />
                </body>
            </body>
            <body name="L1" pos="-{0} 0 0">
                <geom type="capsule" size="0.01" fromto="0 0 0 0 {1} 0" rgba="0 0 0 1"/>
                <joint name="L1j" type="hinge" pos="0 0 0" axis="0 0 1" />
                <body name="L2" pos="0 {1} 0">
                    <geom type="capsule" size="0.01" fromto="0 0 0 0 {2} 0"/>
                    <joint name="L2j" type="hinge" pos="0 0 0" axis="0 0 1" />
                </body>
            </body>
            </body>
            <body name="obj" pos="0 {4} 0.0">
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="hinge" axis="0 0 1"/>
              <!-- <geom type="capsule" size="{4} 0.025" pos="0 0 0"/> -->
              <geom type="box" size="{4} {4} 0.025" pos="0 0 0"/> 
              <!-- <geom type="box" size="{4} 0.01 0.025" pos="0 0 0"/> -->
              <!-- <geom type="box" size="0.01 {4} 0.025" pos="0 0 0"/> -->
              <!-- <geom type="box" size="0.02 0.02 0.025" pos="0 0 0"/> -->
            </body>

        </worldbody>

        <actuator>
          <general joint="R1j"/>
          <general joint="R2j"/>
          <general joint="L1j"/>
          <general joint="L2j"/>
          <damper joint="R1j" ctrlrange="0 100000" kv="1"/>
          <damper joint="R2j" ctrlrange="0 100000" kv="1"/>
          <damper joint="L1j" ctrlrange="0 100000" kv="1"/>
          <damper joint="L2j" ctrlrange="0 100000" kv="1"/>
        </actuator>

    </mujoco>
    '''.format(d, l1, l2, d - 0.02, bs/2, q_limit)
    mj_model = mujoco.MjModel.from_xml_string(xml)
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # mj_model.opt.iterations = 6
    # mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    self.steps_per_control_step = 50
    kwargs['n_frames'] = kwargs.get(
        'n_frames', 1)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self.e_list = jp.zeros((16, 4))
    ct = 0
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):
            self.e_list = self.e_list.at[ct].set( jp.array([ (-1.0)**(i), (-1.0)**(j), (-1.0)**(k), (-1.0)**(l)  ]))
            ct +=1
    self.q_limit = q_limit

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
  
  def transmission(self, d, w, e, f):
    v = d.qvel[0:4]
    ctrl = jp.zeros(self.sys.nu)
    # ctrl.at[:4] = e * f
    # ctrl.at[4:8] = jp.abs(v) * w
    # return ctrl
    return jp.concatenate([e*f, jp.abs(v) * w])
  
  # def controller(self, d, e_idx, q_target):
  def controller(self, d, action):
    idx = jp.argmax(action)
    e = self.e_list[idx].squeeze()

    q = d.qpos[0:4]
    v = d.qvel[0:4]
    kp = 10.0

    f = kp* (jp.dot(e,(e * self.q_limit  - q - 2 * np.sqrt([0.01/kp]) * v)))
    w = 0.1
    return w, e, f

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + 0.0 * jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = 0.0 * jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward': zero,
        'y_reward': zero, 
        'x_reward': zero,
        'theta_reward': zero
    }
    return State(data, obs, reward, done, metrics)

  @property
  def action_size(self) -> int:
    return 16

  @property
  def dt(self) -> float:
    return self.sys.opt.timestep * self.steps_per_control_step

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data = state.pipeline_state

    def f(d, _):
      w, e, f = self.controller(d, action)
      ctrl = self.transmission(d, w, e, f)
      return (self._pipeline.step(self.sys, d, ctrl, self._debug), None)
    # for _ in range(self.steps_per_control_step ):
    #   w, e, f = self.controller(data, action)
    #   ctrl = self.transmission(data, w, e, f)
    #   data = self.pipeline_step(data, ctrl)
    data = jax.lax.scan(f, data, (), self.steps_per_control_step)[0]

    
    obs = self._get_obs(data, action)
    x_reward = -data.qpos[4] * data.qpos[4]
    y_reward = -data.qpos[5] * data.qpos[5]
    theta_reward = -(data.qpos[6] - 1.57) * (data.qpos[6] - 1.57)
    reward =  theta_reward + 10.0 * y_reward# forward_reward + healthy_reward - ctrl_cost
    done = 0.0
    state.metrics.update(
        reward = reward,
        y_reward = y_reward,
        x_reward = x_reward
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    # external_contact_forces are excluded
    return jp.concatenate([
        data.qpos,
        data.qvel
    ])




class TwoArmProp(PipelineEnv):

  def __init__(
      self,
      forward_reward_weight=1.25,
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(1.0, 2.0),
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,
  ):
    d = 0.15
    l1, l2 = 0.14, 0.14
    bs = 0.15
    q_limit_l = jp.array([-jp.pi/2, -2.5, -jp.pi/2, -2.5])
    q_limit_u = jp.array([jp.pi/2, 2.5, jp.pi/2, 2.5])

    xml = '''
    <mujoco>
        <compiler angle="radian" />
        <option  timestep="0.001" gravity="0 0 0"/>
        <default>
        <geom solref="0.002 1" contype="1" conaffinity="1" friction="0.3 0.005 0.0001" />
        <joint  armature="0.01"/>
        </default>
        <visual>
        <map stiffness="10000"/>
        <global offheight="640" offwidth="640"/>
        </visual>
        <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
        <texture name="body" type="cube" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1"/>
        <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="4 4" texuniform="false" reflectance=".0"/>
        </asset>

        <worldbody>
            <geom name="floor" type="plane" size=".4 .4 0.01" material="grid" pos="0 0.2 -0.025"/>
            <camera name="top" pos="0 0 1.0" resolution="640 640"/>
            <body name="robot">
            <geom name="base" type="capsule" size="0.01" fromto="-{3} -0.02 0 {3} -0.02 0" />
            <body name="R1" pos="{0} 0 0.0">
                <geom type="capsule" size="0.01" fromto="0 0 0 0 {1} 0" rgba="0 0 0 1"/>
                <joint name="R1j" type="hinge" pos="0 0 0" axis="0 0 1"  range="{5} {6}" />
                <body name="R2" pos="0 {1} 0">
                    <geom type="capsule" size="0.01" fromto="0 0 0 0 {2} 0" />
                    <joint name="R2j" type="hinge" pos="0 0 0" axis="0 0 1"  range="{7} {8}"/>
                </body>
            </body>
            <body name="L1" pos="-{0} 0 0">
                <geom type="capsule" size="0.01" fromto="0 0 0 0 {1} 0" rgba="0 0 0 1"/>
                <joint name="L1j" type="hinge" pos="0 0 0" axis="0 0 1"  range="{9} {10}"/>
                <body name="L2" pos="0 {1} 0">
                    <geom type="capsule" size="0.01" fromto="0 0 0 0 {2} 0"/>
                    <joint name="L2j" type="hinge" pos="0 0 0" axis="0 0 1"  range="{11} {12}"/>
                </body>
            </body>
            </body>
            <body name="obj" pos="0 {4} 0.0">
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 1 0"/>
            <joint type="hinge" axis="0 0 1"/>
            <!-- <geom type="capsule" size="{4} 0.025" pos="0 0 0"/> -->
            <geom type="box" size="{4} {4} 0.025" pos="0 0 0"/>
            <!-- <geom type="box" size="{4} 0.01 0.025" pos="0 0 0"/> -->
            <!-- <geom type="box" size="0.01 {4} 0.025" pos="0 0 0"/> -->
            <!-- <geom type="box" size="0.02 0.02 0.025" pos="0 0 0"/> -->
            </body>

        </worldbody>

        <actuator>
        <general joint="R1j"/>
        <general joint="R2j"/>
        <general joint="L1j"/>
        <general joint="L2j"/>
        <damper joint="R1j" ctrlrange="0 100000" kv="1"/>
        <damper joint="R2j" ctrlrange="0 100000" kv="1"/>
        <damper joint="L1j" ctrlrange="0 100000" kv="1"/>
        <damper joint="L2j" ctrlrange="0 100000" kv="1"/>
        </actuator>

    </mujoco>
    '''.format(d, l1, l2, d - 0.02, bs/2, q_limit_l[0], q_limit_u[0], q_limit_l[1], q_limit_u[1], q_limit_l[2], q_limit_u[2], q_limit_l[3], q_limit_u[3])
    mj_model = mujoco.MjModel.from_xml_string(xml)
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # mj_model.opt.iterations = 6
    # mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    self.steps_per_control_step = 50
    kwargs['n_frames'] = kwargs.get(
        'n_frames', 1)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self.e_list = jp.zeros((16, 4))
    ct = 0
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):
            self.e_list = self.e_list.at[ct].set( jp.array([ (-1.0)**(i), (-1.0)**(j), (-1.0)**(k), (-1.0)**(l)  ]))
            ct +=1
    self.q_limit_l = q_limit_l
    self.q_limit_u = q_limit_u

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
  
  def transmission(self, d, w, e, f):
    # v = d.qvel[0:4]
    # ctrl = jp.zeros(self.sys.nu)
    # ctrl.at[:4] = e * f
    # ctrl.at[4:8] = jp.abs(v) * w
    # return ctrl
    # return jp.concatenate([e*f, jp.abs(v) * w])
    return jp.concatenate([e*f, w])
  
  # def controller(self, d, e_idx, q_target):
  def controller(self, d, action):
    # idx = jp.argmax(action.at[0:4])
    # e = self.e_list[idx].squeeze()
    e = jax.lax.slice_in_dim(action, 0, 4)
    e = jp.sign(e)

    q_wall = jax.lax.slice_in_dim(action, 4, 8)
    q_wall = q_wall * (self.q_limit_u - self.q_limit_l) * 0.5 + (self.q_limit_l + self.q_limit_u) * 0.5

    q = d.qpos[0:4]
    v = d.qvel[0:4]
    kp = 10.0

    f = kp* (jp.dot(e,(e * (self.q_limit_u - self.q_limit_l) * 0.5 + (self.q_limit_l + self.q_limit_u) * 0.5  - q - 2 * np.sqrt([0.01/kp]) * v)))
    f = jp.clip(f, 0.0, 3.0)
    w_mag = 0.2
    w = w_mag / jp.abs(q_wall - q)
    
    return w, e, f

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + 0.0 * jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = 0.0 * jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward': zero,
        'y_reward': zero, 
        'x_reward': zero,
        'theta_reward': zero
    }
    return State(data, obs, reward, done, metrics)

  @property
  def action_size(self) -> int:
    return 8

  @property
  def dt(self) -> float:
    return self.sys.opt.timestep * self.steps_per_control_step

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data = state.pipeline_state

    def f(d, _):
      w, e, f = self.controller(d, action)
      ctrl = self.transmission(d, w, e, f)
      return (self._pipeline.step(self.sys, d, ctrl, self._debug), None)
    # for _ in range(self.steps_per_control_step ):
    #   w, e, f = self.controller(data, action)
    #   ctrl = self.transmission(data, w, e, f)
    #   data = self.pipeline_step(data, ctrl)
    data = jax.lax.scan(f, data, (), self.steps_per_control_step)[0]

    
    obs = self._get_obs(data, action)
    x_reward = -data.qpos[4] * data.qpos[4]
    y_reward = -data.qpos[5] * data.qpos[5]
    theta_reward = -(data.qpos[6] - 1.57) * (data.qpos[6] - 1.57)
    reward =  theta_reward + 10.0 * y_reward# forward_reward + healthy_reward - ctrl_cost
    done = 0.0
    state.metrics.update(
        reward = reward,
        y_reward = y_reward,
        x_reward = x_reward
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    # external_contact_forces are excluded
    return jp.concatenate([
        data.qpos,
        data.qvel
    ])
