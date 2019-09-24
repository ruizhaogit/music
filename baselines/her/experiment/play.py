# COPYRIGHT NOTICE: THIS CODE IS ONLY FOR THE PURPOSE OF ICLR 2020 REVIEW.
# PLEASE DO NOT DISTRIBUTE. WE WILL CREATE A NEW CODE REPO AFTER THE REVIEW PROCESS.

import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import save_video
import os
import json


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=20)
@click.option('--render', type=click.Choice(['human', 'rgb_array']), default='rgb_array')
@click.option('--exploit', type=bool, default=True)
@click.option('--compute_q', type=bool, default=True)
@click.option('--collect_data', type=bool, default=True)
@click.option('--goal_generation', type=str, default='Zero')
@click.option('--note', type=str, default=None, help='unique notes')

def main(policy_file, seed, n_test_rollouts, render, exploit, compute_q, collect_data, goal_generation, note):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['note'] = note or params['note']
    if note:
        with open('params/'+env_name+'/'+note+'.json', 'r') as file:
            override_params = json.loads(file.read())
            params.update(**override_params)

    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    goal_generation = params['goal_generation']
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': exploit, # eval: True, train: False
        'use_target_net': params['test_with_polyak'], # eval/train: False
        'compute_Q': compute_q, # eval: True, train: False
        'rollout_batch_size': 1,
        'render': render,
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    num_skills = params['num_skills']

    if goal_generation == 'Zero':
        generated_goal = np.zeros(evaluator.g.shape)
    else:
        generated_goal = False

    for z in range(num_skills):
        assert(evaluator.rollout_batch_size==1)
        z_s_onehot = np.zeros([evaluator.rollout_batch_size, num_skills])
        z_s_onehot[0, z] = 1

        base = os.path.splitext(policy_file)[0]
        for i_test_rollouts in range(n_test_rollouts):
            if render == 'rgb_array' or render == 'human':

                imgs, episode = evaluator.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot)
                end = '_test_{:02d}_exploit_{}_compute_q_{}_skill_{}.avi'.format(i_test_rollouts, exploit, compute_q, z)
                test_filename = base + end
                save_video(imgs[0], test_filename, lib='cv2')
            else:
                episode = evaluator.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot)

            if collect_data:
                end = '_test_{:02d}_exploit_{}_compute_q_{}_skill_{}.txt'.format(i_test_rollouts, exploit, compute_q, z)
                test_filename = base + end
                with open(test_filename, 'w') as file:
                     file.write(json.dumps(episode['o'].tolist()))

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
