# COPYRIGHT NOTICE: THIS CODE IS ONLY FOR THE PURPOSE OF ICLR 2020 REVIEW.
# PLEASE DO NOT DISTRIBUTE. WE WILL CREATE A NEW CODE REPO AFTER THE REVIEW PROCESS.

import numpy as np
import random
from baselines.her.util import split_observation_np
from baselines.common.schedules import PiecewiseSchedule
from scipy.stats import rankdata


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, mi_w_schedule, et_w_schedule, mi_prioritization):

    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0
    mi_w_scheduler = PiecewiseSchedule(endpoints=mi_w_schedule)
    et_w_scheduler = PiecewiseSchedule(endpoints=et_w_schedule)

    def _sample_her_transitions(ddpg, ir, episode_batch, batch_size_in_transitions, mi_r_scale, sk_r_scale, t):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        # calculate intrinsic rewards
        mi_trans = np.zeros([episode_idxs.shape[0], 1])
        sk_trans = np.zeros([episode_idxs.shape[0], 1])
        if ir:
            if mi_prioritization and not (episode_batch['p'].sum() == 0):
                r_traj = rankdata(episode_batch['p'], method='dense')
                r_traj = r_traj - 1
                if not (r_traj.sum()==0):
                    p_traj = r_traj / r_traj.sum()
                    episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, replace=True, p=p_traj.flatten())

            o_curr = episode_batch['o'][episode_idxs, t_samples].copy()
            o_curr = np.reshape(o_curr, (o_curr.shape[0], 1, o_curr.shape[-1]))
            o_next = episode_batch['o'][episode_idxs, t_samples+1].copy()
            o_next = np.reshape(o_next, (o_next.shape[0], 1, o_next.shape[-1]))
            o_s = np.concatenate((o_curr, o_next), axis=1)

            if mi_r_scale > 0:
                neg_l = ddpg.run_mi(o_s)
                mi_trans = (-neg_l).copy()

            o = episode_batch['o'][episode_idxs, t_samples].copy()
            z = episode_batch['z'][episode_idxs, t_samples].copy()
            if sk_r_scale > 0:
                sk_r = ddpg.run_sk(o, z)
                sk_trans = sk_r.copy()
        # #

        transitions = {}
        for key in episode_batch.keys():
            if not (key == 'm' or key == 's' or key == 'p'):
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
            else:
                transitions[key] = episode_batch[key][episode_idxs].copy()
        transitions['m'] = transitions['m'].flatten().copy()
        transitions['s'] = transitions['s'].flatten().copy()

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        if ir:
            transitions['m'] = mi_trans.flatten().copy()
            transitions['s'] = sk_trans.flatten().copy()

        transitions['m_w'] = mi_w_scheduler.value(t)
        transitions['s_w'] = 1.0
        transitions['r_w'] = 1.0
        transitions['e_w'] = et_w_scheduler.value(t)

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
