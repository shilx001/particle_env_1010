import numpy as np
import tensorflow as tf


# DQN algorithm
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_min=0.1,
            replace_target_iter=500,
            memory_size=10000,
            replay_start=1000,
            batch_size=32,
            e_greedy_decay=0.01,
            name='default'
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.replace_target_iter = replace_target_iter
        self.replay_start = replay_start
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decay = e_greedy_decay
        self.namespace = name

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3 + 1 + 2))  # 存入episode号与step号，方便进行后续计算

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.namespace + 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.namespace + 'eval_net')

        with tf.variable_scope(self.namespace + 'soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features],
                                name=self.namespace + 's')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features],
                                 name=self.namespace + 's_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name=self.namespace + 'r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name=self.namespace + 'a')  # input Action [batch_size, ]
        self.done = tf.placeholder(tf.float32, [None, ],
                                   name=self.namespace + 'done')  # if s_ is the end of episode
        self.importance_ratio = tf.placeholder(tf.float32, [None, ],
                                               name=self.namespace + 'IS_ratio')  # importance ratio
        self.epsilon_factor = tf.placeholder(tf.float32, [None, 1], name=self.namespace + 'epsilon_factor')
        self.episode = tf.placeholder(tf.float32, [None, 1], name=self.namespace + 'episode')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope(self.namespace + 'eval_net'):
            input_all = tf.concat([self.s, self.episode / 100, self.epsilon_factor], axis=1)
            e1 = tf.layers.dense(input_all, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope(self.namespace + 'target_net'):
            input_all = tf.concat([self.s_, self.episode / 100, self.epsilon_factor], axis=1)
            t1 = tf.layers.dense(input_all, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope(self.namespace + 'q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') * (
                1 - self.done)  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)  # q_target只作为值，不计算梯度
        with tf.variable_scope(self.namespace + 'q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope(self.namespace + 'loss'):  # loss function with importance sampling
            self.loss = tf.reduce_mean(tf.multiply(tf.clip_by_value(self.importance_ratio, 0.01, 2),
                                                   tf.squared_difference(self.q_target, self.q_eval_wrt_a,
                                                                         name='TD_error')))
        with tf.variable_scope(self.namespace + 'train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, done, action_probability, episode):
        s = np.reshape(np.array(s), [self.n_features, ])
        s_ = np.reshape(np.array(s_), [self.n_features, ])

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_, done, action_probability, episode, self.epsilon))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, episode):
        # to have batch dimension when feed into tf placeholder
        observation = np.reshape(np.array(observation), [1, self.n_features])

        if np.random.uniform() < 1 - self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation,
                                                     self.epsilon_factor: np.reshape(self.epsilon, [1, 1]),
                                                     self.episode: episode})
            action = np.argmax(actions_value)
            action_probability = 1 - self.epsilon + self.epsilon / self.n_actions
        else:
            action = np.random.randint(0, self.n_actions)
            action_probability = self.epsilon / self.n_actions
        return action, action_probability

    def learn(self, current_probability):
        if self.memory_counter < self.replay_start:
            return
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        importance_ratio = current_probability / batch_memory[:, -3]
        # print('importance ration is:',importance_ratio)
        # feed_dict index要改
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features - 4:-4],
                self.done: batch_memory[:, -4],
                self.importance_ratio: importance_ratio,
                self.episode: np.reshape(batch_memory[:, -2], [-1, 1]),
                self.epsilon_factor: np.reshape(batch_memory[:, -1], [-1, 1])
            })

        self.cost_his.append(cost)
        # print("cost is: " + str(cost))
        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig(self.namespace)
