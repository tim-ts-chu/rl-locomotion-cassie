
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import os.path
from rlpyt.samplers.serial.sampler import SerialSampler
#from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from env import env_make

def get_full_path(filename):
    # for loading local xml model, it has to be fullpath
    # otherwise gym will search under its own libpath
    fullpath = os.path.abspath(filename)
    return fullpath

def build_and_train(run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=env_make,
        env_kwargs=dict(xml_file=get_full_path('cassie.xml')),
        eval_env_kwargs=dict(xml_file=get_full_path('cassie.xml')),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(1000),
        eval_max_trajectories=50,
    )
    algo = SAC()  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict()
    name = "sac_cassie"
    log_dir = "cassie"
    with logger_context(log_dir, run_ID, name, config, use_summary_writer=False):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
