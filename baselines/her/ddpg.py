from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, split_observation_tf, save_weight, load_weight)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
import json
from collections import deque


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class_actor_critic, network_class_discriminator, polyak, batch_size, Q_lr, pi_lr, mi_lr, sk_lr, r_scale, mi_r_scale, sk_r_scale, et_r_scale, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals ,clip_pos_returns, clip_return, sample_transitions, gamma, env_name, max_timesteps, pretrain_weights, finetune_pi, mi_prioritization, sac, reuse=False, history_len=10000, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class_actor_critic)
        self.create_discriminator = import_function(self.network_class_discriminator)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimz = self.input_dims['z']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        self.env_name = env_name

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['w'] = (None,)
        stage_shapes['m'] = (None,)
        stage_shapes['s'] = (None,)
        stage_shapes['m_w'] = ()
        stage_shapes['s_w'] = ()
        stage_shapes['r_w'] = ()
        stage_shapes['e_w'] = ()
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(pretrain_weights, mi_prioritization, reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, mi_prioritization)

        self.mi_r_history = deque(maxlen=history_len)
        self.gl_r_history = deque(maxlen=history_len)
        self.sk_r_history = deque(maxlen=history_len)
        self.et_r_history = deque(maxlen=history_len)
        self.mi_current = 0
        self.finetune_pi = finetune_pi

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, z, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        if self.sac:
            vals = [policy.mu_tf]
        else:
            vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]

        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.z_tf: z.reshape(-1, self.dimz),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        
        # update the mutual information reward into the episode batch
        episode_batch['m'] = np.empty([episode_batch['o'].shape[0], 1])
        episode_batch['s'] = np.empty([episode_batch['o'].shape[0], 1])
        # #

        self.buffer.store_episode(episode_batch, self)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            
            transitions = self.sample_transitions(self, False, episode_batch, num_normalizing_transitions, 0, 0, 0)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        self.mi_adam.sync()
        self.sk_adam.sync()

    def _grads_mi(self, data):
        mi, mi_grad = self.sess.run([
            self.main_ir.mi_tf,
            self.mi_grad_tf,
        ], feed_dict={self.o_tau_tf: data})
        return mi, mi_grad

    def _grads_sk(self, o_s_batch, z_s_batch):
        sk, sk_grad = self.sess.run([
            self.main_ir.sk_tf,
            self.sk_grad_tf,
        ], feed_dict={self.main_ir.o_tf: o_s_batch, self.main_ir.z_tf: z_s_batch})
        return sk, sk_grad

    def _grads(self):
        critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.main.neg_logp_pi_tf,
            self.e_w_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w

    def _update_mi(self, mi_grad):
        self.mi_adam.update(mi_grad, self.mi_lr)

    def _update_sk(self, sk_grad):
        self.sk_adam.update(sk_grad, self.sk_lr)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self, ir, t):

        transitions = self.buffer.sample(self, ir, self.batch_size, self.mi_r_scale, self.sk_r_scale, t)
        weights = np.ones_like(transitions['r']).copy()
        if ir:
            self.mi_r_history.extend(((np.clip((self.mi_r_scale * transitions['m']), *(0, 1))- (1 if not self.mi_r_scale == 0 else 0) )*transitions['m_w']).tolist())
            self.sk_r_history.extend(((np.clip(self.sk_r_scale * transitions['s'], *(-1, 0)))*1.00).tolist())
            self.gl_r_history.extend(self.r_scale * transitions['r'])

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions['w'] = weights.flatten().copy() # note: ordered dict
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, ir, t, batch=None):
        if batch is None:
            batch = self.sample_batch(ir, t)
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def run_mi(self, o_s):
        feed_dict = {self.o_tau_tf: o_s.copy()}
        neg_l = self.sess.run(self.main_ir.mi_tf, feed_dict=feed_dict)
        return neg_l

    def run_sk(self, o, z):
        feed_dict = {self.main_ir.o_tf: o, self.main_ir.z_tf: z}
        sk_r = self.sess.run(self.main_ir.sk_r_tf, feed_dict=feed_dict)
        return sk_r

    def train_mi(self, data, stage=True):
        mi, mi_grad = self._grads_mi(data)
        self._update_mi(mi_grad)
        self.mi_current = -mi.mean()
        return -mi.mean()

    def train_sk(self, o_s_batch, z_s_batch, stage=True):
        sk, sk_grad = self._grads_sk(o_s_batch, z_s_batch)
        self._update_sk(sk_grad)
        return -sk.mean()

    def train(self, t, stage=True):
        if not self.buffer.current_size==0:
            if stage:
                self.stage_batch(ir=True, t=t)
            critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w = self._grads()
            self._update(Q_grad, pi_grad)
            self.et_r_history.extend((( np.clip((self.et_r_scale * neg_logp_pi), *(-1, 0))) * e_w ).tolist())
            return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, pretrain_weights, mi_prioritization, reuse=False):
        if self.sac:
            logger.info("Creating a SAC agent with action space %d x %s..." % (self.dimu, self.max_u))
        else:
            logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        batch_tf['w'] = tf.reshape(batch_tf['w'], [-1, 1])
        batch_tf['m'] = tf.reshape(batch_tf['m'], [-1, 1])
        batch_tf['s'] = tf.reshape(batch_tf['s'], [-1, 1])

        self.o_tau_tf = tf.placeholder(tf.float32, shape=(None, None, self.dimo))

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # intrinsic reward (ir) network for mutual information
        with tf.variable_scope('ir') as vs:
            if reuse:
                vs.reuse_variables()
            self.main_ir = self.create_discriminator(batch_tf, net_type='ir', **self.__dict__)
            vs.reuse_variables()

        # loss functions

        mi_grads_tf = tf.gradients(tf.reduce_mean(self.main_ir.mi_tf), self._vars('ir/state_mi'))
        assert len(self._vars('ir/state_mi')) == len(mi_grads_tf)
        self.mi_grads_vars_tf = zip(mi_grads_tf, self._vars('ir/state_mi'))
        self.mi_grad_tf = flatten_grads(grads=mi_grads_tf, var_list=self._vars('ir/state_mi'))
        self.mi_adam = MpiAdam(self._vars('ir/state_mi'), scale_grad_by_procs=False)

        sk_grads_tf = tf.gradients(tf.reduce_mean(self.main_ir.sk_tf), self._vars('ir/skill_ds'))
        assert len(self._vars('ir/skill_ds')) == len(sk_grads_tf)
        self.sk_grads_vars_tf = zip(sk_grads_tf, self._vars('ir/skill_ds'))
        self.sk_grad_tf = flatten_grads(grads=sk_grads_tf, var_list=self._vars('ir/skill_ds'))
        self.sk_adam = MpiAdam(self._vars('ir/skill_ds'), scale_grad_by_procs=False)

        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, self.clip_return if self.clip_pos_returns else np.inf)

        self.e_w_tf = batch_tf['e_w']

        if not self.sac:
            self.main.neg_logp_pi_tf = tf.zeros(1)

        target_tf = tf.clip_by_value(self.r_scale * batch_tf['r'] * batch_tf['r_w'] + (tf.clip_by_value( self.mi_r_scale * batch_tf['m'], *(0, 1) ) - (1 if not self.mi_r_scale == 0 else 0)) * batch_tf['m_w'] + (tf.clip_by_value( self.sk_r_scale * batch_tf['s'], *(-1, 0))) * batch_tf['s_w'] + (tf.clip_by_value( self.et_r_scale * self.main.neg_logp_pi_tf, *(-1, 0))) * self.e_w_tf + self.gamma * target_Q_pi_tf, *clip_range)

        self.td_error_tf = tf.stop_gradient(target_tf) - self.main.Q_tf
        self.errors_tf = tf.square(self.td_error_tf)
        self.errors_tf = tf.reduce_mean(batch_tf['w'] * self.errors_tf)
        self.Q_loss_tf = tf.reduce_mean(self.errors_tf)

        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')

        # polyak averaging
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        if pretrain_weights:
            load_weight(self.sess, pretrain_weights, ['state_mi']) 
            if self.finetune_pi:
                load_weight(self.sess, pretrain_weights, ['main'])

        self._sync_optimizers()
        if pretrain_weights and self.finetune_pi:
            load_weight(self.sess, pretrain_weights, ['target'])
        else:
            self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('mi_reward/mean', np.mean(self.mi_r_history))]
        logs += [('mi_reward/std', np.std(self.mi_r_history))]
        logs += [('mi_reward/max', np.max(self.mi_r_history))]
        logs += [('mi_reward/min', np.min(self.mi_r_history))]
        logs += [('mi_train/-neg_l', self.mi_current)]
        logs += [('sk_reward/mean', np.mean(self.sk_r_history))]
        logs += [('sk_reward/std', np.std(self.sk_r_history))]
        logs += [('sk_reward/max', np.max(self.sk_r_history))]
        logs += [('sk_reward/min', np.min(self.sk_r_history))]
        logs += [('et_reward/mean', np.mean(self.et_r_history))]
        logs += [('et_reward/std', np.std(self.et_r_history))]
        logs += [('et_reward/max', np.max(self.et_r_history))]
        logs += [('et_reward/min', np.min(self.et_r_history))]
        logs += [('gl_reward/mean', np.mean(self.gl_r_history))]
        logs += [('gl_reward/std', np.std(self.gl_r_history))]
        logs += [('gl_reward/max', np.max(self.gl_r_history))]
        logs += [('gl_reward/min', np.min(self.gl_r_history))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic', 'create_discriminator', '_history']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        if 'env_name' not in state:
            state['env_name'] = 'FetchPickAndPlace-v1'
        if 'network_class_discriminator' not in state:
            state['network_class_discriminator'] = 'baselines.her.discriminator:Discriminator'
        if 'mi_r_scale' not in state:
            state['mi_r_scale'] = 1
        if 'mi_lr' not in state:
            state['mi_lr'] = 0.001
        if 'sk_r_scale' not in state:
            state['sk_r_scale'] = 1
        if 'sk_lr' not in state:
            state['sk_lr'] = 0.001
        if 'et_r_scale' not in state:
            state['et_r_scale'] = 1
        if 'finetune_pi' not in state:
            state['finetune_pi'] = None
        if 'no_train_mi' not in state:
            state['no_train_mi'] = None
        if 'load_weight' not in state:
            state['load_weight'] = None
        if 'pretrain_weights' not in state:
            state['pretrain_weights'] = None
        if 'mi_prioritization' not in state:
            state['mi_prioritization'] = None
        if 'sac' not in state:
            state['sac'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
