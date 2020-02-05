import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from baselines.her.util import store_args, nn, split_observation_tf


class Discriminator:
    @store_args
    def __init__(self, inputs_tf, dimo, dimz, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env_name, **kwargs):
        """The discriminator network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        self.o_tf = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.z_tf = tf.placeholder(tf.float32, shape=(None, self.dimz))
        self.g_tf = tf.placeholder(tf.float32, shape=(None, self.dimg))

        obs_tau_excludes_goal, obs_tau_achieved_goal = split_observation_tf(self.env_name, self.o_tau_tf)

        obs_excludes_goal, obs_achieved_goal = split_observation_tf(self.env_name, self.o_tf)

        # Discriminator networks

        with tf.variable_scope('state_mi'):
            # Mutual Information Neural Estimation
            # shuffle and concatenate
            x_in = obs_tau_excludes_goal
            y_in = obs_tau_achieved_goal
            y_in_tran = tf.transpose(y_in, perm=[1, 0, 2])
            y_shuffle_tran = tf.random_shuffle(y_in_tran)
            y_shuffle = tf.transpose(y_shuffle_tran, perm=[1, 0, 2])
            x_conc = tf.concat([x_in, x_in], axis=-2)
            y_conc = tf.concat([y_in, y_shuffle], axis=-2)

            # propagate the forward pass
            layerx = tf_layers.linear(x_conc, int(self.hidden/2))
            layery = tf_layers.linear(y_conc, int(self.hidden/2))
            layer2 = tf.nn.relu(layerx + layery)
            output = tf_layers.linear(layer2, 1)
            output = tf.nn.tanh(output)
            
            # split in T_xy and T_x_y predictions
            N_samples = tf.shape(x_in)[-2]
            T_xy = output[:,:N_samples,:]
            T_x_y = output[:,N_samples:,:]
            
            # compute the negative loss (maximise loss == minimise -loss)
            mean_exp_T_x_y = tf.reduce_mean(tf.math.exp(T_x_y), axis=-2)
            neg_loss = -(tf.reduce_mean(T_xy, axis=-2) - tf.math.log(mean_exp_T_x_y))
            neg_loss = tf.check_numerics(neg_loss, 'check_numerics caught bad neg_loss')
            self.mi_tf = neg_loss

        with tf.variable_scope('skill_ds'):
            self.logits_tf = nn(obs_achieved_goal, [int(self.hidden/2)] * self.layers + [self.dimz])
            self.sk_tf = tf.nn.softmax_cross_entropy_with_logits(labels=self.z_tf, logits=self.logits_tf)
            self.sk_r_tf = -1 * self.sk_tf
                        



