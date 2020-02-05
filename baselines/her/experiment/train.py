import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

import os.path as osp
import tempfile
import datetime
from baselines.her.util import (dumpJson, loadJson, save_video, save_weight, load_weight)
import pickle
import tensorflow as tf


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def sample_skill(num_skills, rollout_batch_size, use_skill_n=None):
    # sample skill z

    z_s = np.random.randint(0, num_skills, rollout_batch_size)
    if use_skill_n:
        use_skill_n = use_skill_n - 1
        z_s.fill(use_skill_n)
        
    z_s_onehot = np.zeros([rollout_batch_size, num_skills])
    z_s = np.array(z_s).reshape(rollout_batch_size, 1)
    for i in range(rollout_batch_size):
        z_s_onehot[i, z_s[i]] = 1
    return z_s, z_s_onehot

def train(logdir, policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval, save_policies, num_cpu, collect_data, collect_video, goal_generation, num_skills, use_skill_n, batch_size, mi_r_scale, mi_end_epoch, sk_r_scale, no_train_mi, **kwargs):

    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    t = 1
    for epoch in range(n_epochs):

        # train
        episodes = []
        rollout_worker.clear_history()
        for cycle in range(n_cycles):

            z_s, z_s_onehot = sample_skill(num_skills, rollout_worker.rollout_batch_size, use_skill_n)

            if goal_generation == 'Zero':
                generated_goal = np.zeros(rollout_worker.g.shape)
            else:
                generated_goal = False

            if collect_video:
                imgs, episode = rollout_worker.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot)
                # log video
                for rollout in range(rollout_worker.rollout_batch_size):
                    filename = logdir + '/video_epoch_{0}_rank_{1}_cycle_{2}_rollout_{3}_skill_{4}.avi'.format(epoch, rank, cycle, rollout, z_s[rollout][0])
                    if rank == 0:
                        save_video(imgs[rollout], filename)
                # #
            else:
                episode = rollout_worker.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot)
            episodes.append(episode)
            policy.store_episode(episode)
            for batch in range(n_batches):
                t = epoch
                policy.train(t)

                # train mutual information estimator
                if epoch >= mi_end_epoch:
                    mi_r_scale = 0
                    policy.mi_r_scale = 0

                if mi_r_scale > 0 and (not no_train_mi):
                    o_s = policy.buffer.buffers['o'][0: policy.buffer.current_size]
                    episode_idxs = np.random.randint(0, policy.buffer.current_size, batch_size)
                    o_s_batch = o_s[episode_idxs]
                    policy.train_mi(o_s_batch)
                # #

                # train skill discriminator
                if sk_r_scale > 0:
                    o_s = policy.buffer.buffers['o'][0: policy.buffer.current_size]
                    z_s = policy.buffer.buffers['z'][0: policy.buffer.current_size]
                    T = z_s.shape[-2]
                    episode_idxs = np.random.randint(0, policy.buffer.current_size, batch_size)
                    t_samples = np.random.randint(T, size=batch_size)
                    o_s_batch = o_s[episode_idxs, t_samples]
                    z_s_batch = z_s[episode_idxs, t_samples]
                    policy.train_sk(o_s_batch, z_s_batch)
                # #

            policy.update_target_net()

        if collect_data and (rank == 0):
            dumpJson(logdir, episodes, epoch, rank)

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            z_s, z_s_onehot = sample_skill(num_skills, evaluator.rollout_batch_size, use_skill_n)
            evaluator.generate_rollouts(generated_goal=False, z_s_onehot=z_s_onehot)

        # record logs
        logger.record_tabular('timestamp', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        logger.record_tabular('best_success_rate', best_success_rate)

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env_name, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, binding, logging, version, n_cycles, note, override_params={}, save_policies=True):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu, binding)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging

    if logging: 
        logdir = 'logs/'+str(env_name)+'-replay_strategy'+str(replay_strategy)+'-n_epochs'+str(n_epochs)+'-num_cpu'+str(num_cpu)+'-seed'+str(seed)+'-n_cycles'+str(n_cycles)+'-version'+str(version)+'-T-'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure() # use temp folder for other rank
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    params['binding'] = binding
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    params['version'] = version
    params['n_cycles'] = n_cycles
    params['num_cpu'] = num_cpu
    params['note'] = note or params['note']
    if note:
        with open('params/'+env_name+'/'+note+'.json', 'r') as file:
            override_params = json.loads(file.read())
            params.update(**override_params)

    if params['load_weight']:
        if type(params['load_weight']) is list:
            params['load_weight'] = params['load_weight'][seed]
        base = os.path.splitext(params['load_weight'])[0]
        policy_weight_file = open(base+'_weight.pkl', 'rb')
        pretrain_weights = pickle.load(policy_weight_file)
        policy_weight_file.close()
    else:
        pretrain_weights = None

    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, pretrain_weights=pretrain_weights, clip_return=clip_return)

    render = False
    if params['collect_video']:
        render = 'rgb_array'

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'render': render,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'], policy_save_interval=policy_save_interval, save_policies=save_policies, num_cpu=num_cpu, collect_data=params['collect_data'], collect_video=params['collect_video'], goal_generation=params['goal_generation'], num_skills=params['num_skills'], use_skill_n=params['use_skill_n'], batch_size=params['_batch_size'], mi_r_scale=params['mi_r_scale'], mi_end_epoch=params['mi_end_epoch'], sk_r_scale=params['sk_r_scale'], no_train_mi=params['no_train_mi'])


@click.command()
@click.option('--env_name', type=click.Choice(['FetchPush-v1','FetchSlide-v1', 'FetchPickAndPlace-v1']), default='FetchPickAndPlace-v1', help='the environment that you want to train on, including: FetchPush-v1, FetchSlide-v1, FetchPickAndPlace-v1')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=1, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--n_cycles', type=int, default=50, help='n_cycles')
@click.option('--replay_strategy', type=click.Choice(['future', 'final', 'none']), default='future', help='replay strategy to be used.')
@click.option('--clip_return', type=int, default=0, help='whether or not returns should be clipped')
@click.option('--binding', type=click.Choice(['none', 'core']), default='core', help='configure mpi using bind-to none or core.')
@click.option('--logging', type=bool, default=False, help='whether or not logging')
@click.option('--version', type=int, default=0, help='version')
@click.option('--note', type=str, default=None, help='unique notes')

def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()
