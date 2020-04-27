#!/usr/bin/python3
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import os.path
import torch
from gym.envs.registration import register
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

import numpy as np
#np.set_printoptions(precision=3)

register(
    id='Cassie-v0',
    entry_point='cassie_gym.cassie_v0:CassieEnv',
    max_episode_steps=1000,
)

def get_full_path(filename):
    # for loading local xml model, it has to be fullpath
    # otherwise gym will search under its own libpath
    fullpath = os.path.abspath(filename)
    return fullpath

def build_and_train(env_id="Cassie-v0", run_ID=0, cuda_idx=None, snapshot_file=None):

    if snapshot_file is None:
        initial_optim_state_dict = None
        initial_model_state_dict = None
    else:
        snapshot = torch.load(snapshot_file)
        initial_optim_state_dict=snapshot['optimizer_state_dict']
        initial_model_state_dict=snapshot['agent_state_dict']

    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id,
            xml_file=get_full_path('resources/cassie.xml')),
        eval_env_kwargs=dict(id=env_id,
            xml_file=get_full_path('resources/cassie.xml')),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(1000),
        eval_max_trajectories=50, # 50
    )
    algo = SAC(
            initial_optim_state_dict=initial_optim_state_dict)
    agent = SacAgent(
            initial_model_state_dict=initial_model_state_dict)
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=5e4, #5e4
        affinity=dict(cuda_idx=cuda_idx),
    )
    other_param = dict(
            env_id=env_id,
            forward_reward_weight=0,
            shift_cost=True,
            cum_steps='1M')
    name = "sac_" + env_id
    log_dir = "Cassie_stand"
    with logger_context(log_dir, run_ID, name, other_param, snapshot_mode='last', use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='Cassie-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--snapshot_file', help='pretrained snapshot file', default=None)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        snapshot_file=args.snapshot_file
    )
