import tensorflow as tf
from func import cudnn_gru, native_gru,pointer, summ, dropout, ptr_net, dense, softmax_mask, dot_attention_sentence, ptr_net_span
import numpy as np

class critic(object):
    def __init__(self, sess, config, trainable = True, opt=True):
        self.config = config
        self.sess = sess
        self.global_step = tf.get_variable('global_step_critic', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        ques_limit = config.ques_limit if trainable else config.test_ques_limit
        sen_num = config.sen_num if trainable else config.test_sen_num
        self.k = config.k
        d = config.hidden

        N = config.batch_size
        # self.c, self.q, self.ch, self.qh, self.y1, self.y2,  self.qa_id = batch.get_next()
        # self.q, self.cs, self.csh, self.ce,  self.qh, self.y1, self.y2, self.y, self.qa_id = batch.get_next()
        self.input = tf.placeholder(tf.float32, [config.batch_size, 2, 4, 6*d])
        self.reward = tf.placeholder(tf.float32, [config.batch_size])

        self.is_train = tf.get_variable(
            "is_train_cri", shape=[], dtype=tf.bool, trainable=False)

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr_cri", shape=[], dtype=tf.float32, trainable=False)

            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op =self.opt.minimize(self.loss)
            # RL change
            # grads = self.opt.compute_gradients(self.loss)
            # gradients, variables = zip(*grads)
            # capped_grads, _ = tf.clip_by_global_norm(
            #     gradients, config.grad_clip)
            # self.train_op = self.opt.apply_gradients(
            #     zip(capped_grads, variables), global_step=self.global_step)



    def ready(self):
        config = self.config
        N, d= config.batch_size,config.hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("critic"):
            inp = tf.reshape(self.input,[N*2*4,6*d])
            x1 = tf.layers.dense(inp, 1, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              trainable=self.is_train)
            x1 = tf.reshape(x1, [N*2, 4])
            x2 = tf.layers.dense(x1, 1, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              trainable=self.is_train)
            x2 = tf.reshape(x2, [N, 2])
            self.baseline = tf.nn.sigmoid(x2)
        with tf.variable_scope("critic-predict"):
            reward = tf.expand_dims(self.reward, axis=1)
            reward = tf.tile(reward,[1,2])
            self.rewardDiff = self.baseline-reward
            self.loss = tf.reduce_mean(tf.square(self.rewardDiff))
