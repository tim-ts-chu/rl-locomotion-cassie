
import numpy as np
from mujoco_py.builder import MujocoException
from gym.envs.mujoco import mujoco_env
from gym import utils

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='',
                 forward_reward_weight=1.25, # 1.25
                 ctrl_cost_weight=0.1, #0.1
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 healthy_reward=5.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.7, 1.5),
                 reset_noise_scale=1e-2, #1e-2
                 exclude_current_positions_from_observation=True,
                 enable_perturb=True):
        utils.EzPickle.__init__(**locals())

        # save reward parameters
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._enable_perturb = enable_perturb
        self._interval_count = 0

        # model path, frame_skip
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(action))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        # qpos is generalized coordinates (x, y, z)
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # TODO print more information out later
        #print('position:\n', len(position))
        #print('velocity:\n', len(velocity))
        #print('com_inertia:\n', len(com_inertia))
        #print('com_velocity:\n', len(com_velocity))
        #print('actuator_forces:\n', len(actuator_forces))
        #print('external_contact_forces:\n', len(external_contact_forces))

        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ))

    def apply_force(self, force):
        # apply force o pelvis: bodyid = 1
        self.sim.data.xfrc_applied[1,:] = force

    def step(self, action):

        # FIXME assume action input range from -1 to 1
        action_range = np.array([
            4.5, 4.5, 12.2, 12.2, 0.9,
            4.5, 4.5, 12.2, 12.2, 0.9])

        if self._enable_perturb:
            self._interval_count += 1
            if self._interval_count % 200 == 0:
                self.apply_force(np.random.uniform(-0,0,6))

        xy_position_before = mass_center(self.model, self.sim)
        try:
            # only use 50% of full power
            self.do_simulation(action*0.5*action_range, self.frame_skip)
        except MujocoException as e:
            print(e)
            # return -5 to represent super unhealthy
            return self._get_obs(), -5, True, {}

        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        shifting_cost_weight = 0
        shifting_cost = shifting_cost_weight*(abs(x_velocity) + abs(y_velocity))

        # forward_reward_weight has been set to zero
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost + shifting_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done

        #print('healthy_reward:', healthy_reward)
        #print('shifting_cost:', shifting_cost)
        #print('control_cost:', ctrl_cost)
        #print('contact_cost:', contact_cost)
        #print('=== final reward ===:', reward)

        info = {
            'reward_linvel': forward_reward,
            'reward_quadctrl': -ctrl_cost,
            'reward_alive': healthy_reward,
            'reward_impact': -contact_cost,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # FIXME This is a tuned better initial pose
        # But this setting should be written in xml file
        self.init_qpos = np.array([
            0, 0, 1.13, 1, 0, 0, 0, 0, 0, 0.25,
            1, 0, 0, 0, -0.79, 0, 1, 0, 0, 0,
            -1.5, 0, 0, 0.25, 1, 0, 0, 0, -0.79, 0,
            1, 0, 0, 0, -1.5])

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        #qpos = self.init_qpos
        #qvel = self.init_qvel
        self.set_state(qpos, qvel)
        #print('qpos:', self.init_qpos)
        #print('qvel:', self.init_qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)



