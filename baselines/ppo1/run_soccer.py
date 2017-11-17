#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import boolean_flag, set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys

sys.path.insert(0, '../../..')

from core.soccer_env import SoccerEnv

def train(env_id, num_timesteps, seed, save_model):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    rank = MPI.COMM_WORLD.Get_rank()
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = SoccerEnv(rank)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', save_model=save_model
        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    boolean_flag(parser, 'save-model', default=True)
    args = parser.parse_args()
    # logger.configure()

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, save_model=args.save_model)

if __name__ == '__main__':
    main()
