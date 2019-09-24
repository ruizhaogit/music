# COPYRIGHT NOTICE: THIS CODE IS ONLY FOR THE PURPOSE OF ICLR 2020 REVIEW.
# PLEASE DO NOT DISTRIBUTE. WE WILL CREATE A NEW CODE REPO AFTER THE REVIEW PROCESS.

import threading
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, mi_prioritization):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # add key for intrinsic rewards
        self.buffers['m'] = np.empty([self.size, 1])
        self.buffers['s'] = np.empty([self.size, 1])
        self.mi_prioritization = mi_prioritization
        if self.mi_prioritization:
            self.buffers['p'] = np.zeros([self.size, 1]) # priority

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, ddpg, ir, batch_size, mi_r_scale, sk_r_scale, t):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(ddpg, ir, buffers, batch_size, mi_r_scale, sk_r_scale, t)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not (key == 'm' or key == 's' or key == 'p'):
                assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch, ddpg):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        if self.mi_prioritization:
            num_steps = episode_batch['o'].shape[-2] - 1
            mi_tran = np.zeros([batch_size, num_steps])
            for t in range(num_steps):
                o_curr = episode_batch['o'][:,t].copy()
                o_curr = np.reshape(o_curr, (o_curr.shape[0], 1, o_curr.shape[-1]))
                o_next = episode_batch['o'][:,t+1].copy()
                o_next = np.reshape(o_next, (o_next.shape[0], 1, o_next.shape[-1]))
                o_s = np.concatenate((o_curr, o_next), axis=1)
                neg_l = ddpg.run_mi(o_s)
                mi_tran[:,t] = (-neg_l).copy().flatten()
                mi_tran[:,t] = np.clip(mi_tran[:,t], a_min=0, a_max=0.0002)
            
            mi_traj = np.sum(mi_tran, axis=1)
            episode_batch['p'] = mi_traj.reshape(-1,1)

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
