import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, dense, softmax_mask, dot_attention_sentence, ptr_net_span
import numpy as np

class select_model(object):
    def __init__(self, sess, config, word_mat=None, char_mat=None, trainable = True, opt=True):
        self.config = config
        self.sess = sess
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        ques_limit = config.ques_limit if trainable else config.test_ques_limit
        sen_num = config.sen_num if trainable else config.test_sen_num
        self.k = config.k
        d = config.hidden
        # self.c, self.q, self.ch, self.qh, self.y1, self.y2,  self.qa_id = batch.get_next()
        # self.q, self.cs, self.csh, self.ce,  self.qh, self.y1, self.y2, self.y, self.qa_id = batch.get_next()
        self.qh = tf.placeholder(tf.int32, [config.batch_size, ques_limit, config.char_limit])
        self.csh = tf.placeholder(tf.int32, [config.batch_size, config.sen_num, config.sen_len, config.char_limit])
        self.ce = tf.placeholder(tf.float32, [config.batch_size, sen_num, config.sen_len, 3])
        self.y = tf.placeholder(tf.float32, [config.batch_size, sen_num])
        self.qa_id = tf.placeholder(tf.int32, [config.batch_size])
        self.q = tf.placeholder(tf.int32, [config.batch_size, ques_limit])
        self.cs = tf.placeholder(tf.int32, [None, sen_num, config.sen_len])
        self.qe = tf.placeholder(tf.float32, [config.batch_size, ques_limit, ques_limit])
        self.ct = tf.placeholder(tf.float32, [config.batch_size, sen_num, config.sen_len, 19])
        self.qt = tf.placeholder(tf.float32, [config.batch_size, ques_limit, 19])
        # self.a = tf.placeholder(tf.int32, None, "act")

        #paragraph encode
        self.reward = tf.placeholder(tf.float32, [None])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.baseline = tf.placeholder(tf.float32, [None])
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
        # self.c_mask = tf.cast(self.c, tf.bool)

        self.c_s_mask = tf.cast(self.cs, tf.bool)
        self.c_s_len = tf.reduce_sum(tf.cast(self.c_s_mask, tf.int32),axis=2)

        self.q_mask = tf.cast(self.q, tf.bool)
        # self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c_s_maxlen = tf.reduce_max(tf.reduce_max(self.c_s_len))
            self.c_p_mask = tf.cast(self.c_s_len, tf.bool)
            self.c_p_len = tf.reduce_sum(tf.cast(self.c_p_mask, tf.int32),axis=1)
            self.c_s_maxnum = tf.reduce_max(tf.reduce_sum(tf.cast(self.c_p_mask, tf.int32), axis=1))
            #self.c_s_maxnum = tf.cast(tf.constant([config.sen_len]), tf.int32)
            self.q_slice = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.cs_slice = tf.slice(self.cs, [0, 0, 0], [N, self.c_s_maxnum, self.c_s_maxlen])
            self.ce_slice = tf.slice(self.ce, [0,0,0,0],[N,self.c_s_maxnum,self.c_s_maxlen,3])
            self.qe_slice = tf.slice(self.qe, [0,0,0],[N,self.q_maxlen,self.q_maxlen])
            self.c_s_mask = tf.slice(self.c_s_mask, [0, 0, 0], [N, self.c_s_maxnum, self.c_s_maxlen])
            self.c_p_mask = tf.slice(self.c_p_mask, [0, 0], [N, self.c_s_maxnum])
            self.csh_slice = tf.slice(self.csh, [0, 0, 0, 0], [N, self.c_s_maxnum, self.c_s_maxlen, CL])
            self.y_slice = tf.slice(self.y, [0, 0], [N, self.c_s_maxnum])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            # self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh_slice = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.ct_slice = tf.slice(self.ct, [0, 0, 0, 0], [N, self.c_s_maxnum, self.c_s_maxlen, 19])
            self.qt_slice = tf.slice(self.qt, [0, 0, 0], [N, self.q_maxlen, 19])


        else:
            self.c_s_maxnum, self.c_s_maxlen, self.q_maxlen = config.sen_num, config.sen_len, config.ques_limit
        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
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
        N, QL, CL, d, dc, dg = config.batch_size, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru
        SN, SL = self.c_s_maxnum, self.c_s_maxlen
        W = config.glove_dim
        print('embedding part')
        with tf.variable_scope("emb"):
            # with tf.variable_scope("char"):
            #         ch_emb = tf.reshape(tf.nn.embedding_lookup(
            #             self.char_mat, self.csh_slice), [N, SN * SL, CL, dc], name='char_reshape')
            #         qh_emb = tf.reshape(tf.nn.embedding_lookup(
            #             self.char_mat, self.qh_slice), [N, QL, CL, dc])
            #         ch_emb = dropout(
            #             ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #         qh_emb = dropout(
            #             qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            # ch_emb_char = tf.unstack(ch_emb, axis=0)
            # qh_emb_char = tf.unstack(qh_emb, axis=0)
            '''

            filter_size = [3, 4, 5]
            att_char = []
            merge_char = []
            q_merge_char = []
            for filter in filter_size:
                with tf.variable_scope("char-cnnencoder-%s" % filter):
                    step_merge_char = []
                    step_att_char = []
                    q_step_merge_char = []
                    q_step_att_char = []
                    for i in range(2):
                        if i==0:
                            input_char=ch_emb
                        else:
                            input_char=qh_emb
                        conv_branch_char = tf.layers.conv2d(
                            inputs=input_char,
                            # use as many filters as the hidden size
                            filters=50,
                            kernel_size=filter,
                            use_bias=True,
                            activation=tf.nn.relu,
                            trainable=True,
                            padding='SAME',
                            name = 'conv_char_' + str(filter),
                            reuse = tf.AUTO_REUSE,
                            data_format='channels_last'
                        )
                        if i ==0:
                            step_att_char.append(conv_branch_char)
                            # pool over the words to obtain: [first_dim x 1* hidden_size]
                            pool_branch_char = tf.reduce_max(conv_branch_char, axis=2)
                            merge_char.append(pool_branch_char)
                        else:
                            q_step_att_char.append(conv_branch_char)
                            # pool over the words to obtain: [first_dim x 1* hidden_size]
                            q_pool_branch_char = tf.reduce_max(conv_branch_char, axis=2)
                            q_merge_char.append(q_pool_branch_char)
                    # batch_merge = tf.stack(step_merge_char, axis=0)
                    # merge_char.append(batch_merge)
                    # batch_merge_q = tf.stack(q_step_merge_char, axis=0)
                    # q_merge_char.append(batch_merge_q)
            ch_con = tf.concat(merge_char, axis=-1)
            ch_con = tf.reshape(ch_con,[N,SN,SL,150])
            qh_con = tf.concat(q_merge_char,axis=-1)
            '''
            # if(use_char):
            #     with tf.variable_scope("char"):
            #         ch_emb = tf.reshape(tf.nn.embedding_lookup(
            #             self.char_mat, self.csh), [N * SN * SL, CL, dc], name='char_reshape')
            #         qh_emb = tf.reshape(tf.nn.embedding_lookup(
            #             self.char_mat, self.qh), [N * QL, CL, dc])
            #         ch_emb = dropout(
            #             ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #         qh_emb = dropout(
            #             qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #         cell_fw = tf.contrib.rnn.GRUCell(dg)
            #         cell_bw = tf.contrib.rnn.GRUCell(dg)
            #         _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #             cell_fw, cell_bw, ch_emb, self.csh_len, dtype=tf.float32)
            #         ch_emb = tf.concat([state_fw, state_bw], axis=1)
            #         _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #             cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
            #         qh_emb = tf.concat([state_fw, state_bw], axis=1)
            #         qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
            #         ch_emb = tf.reshape(ch_emb, [N, SN, SL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.cs_slice)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q_slice)

            with tf.name_scope("softemb"):
                c_emb_linear = tf.nn.relu(
                    dense(c_emb, d, use_bias=True, scope="c_emb_linear"))
                q_emb_linear = tf.nn.relu(
                    dense(q_emb, d, use_bias=True, scope="q_emb_linear"))
                c_emb_linear = tf.reshape(c_emb_linear,[N,self.c_s_maxnum*self.c_s_maxlen,d])
                align_cq = tf.matmul(c_emb_linear,tf.transpose(q_emb_linear,[0,2,1]))

                cq_mask = tf.tile(tf.expand_dims(self.q_mask, axis=1), [1, self.c_s_maxnum*self.c_s_maxlen, 1])
                self.align_cq = tf.nn.softmax(softmax_mask(align_cq, cq_mask))
                align_c_emb = tf.matmul(self.align_cq, q_emb_linear)
                align_c_emb = tf.reshape(align_c_emb,[N, self.c_s_maxnum,self.c_s_maxlen, d])
            c_emb = tf.concat([c_emb,align_c_emb,self.ce_slice,self.ct_slice], axis=3)
            c_emb = tf.reshape(c_emb,[N,self.c_s_maxnum,self.c_s_maxlen,W+d+3+19],name='c_emb_reshape')

            q_emb = tf.concat([q_emb, self.qt_slice], axis=2)
            self.c_emb = c_emb
            self.q_emb = q_emb
            # c_emb = tf.reshape(c_emb, [N,self.c_s_maxnum,self.c_s_maxlen,W+self.q_maxlen])

        print('encode-part')
        # c_s_len = tf.unstack(self.c_s_len, axis=1)


        cnn_out = []
        c_s_emb = tf.unstack(c_emb, axis = 0)
        # q_s_emb = tf.expand_dims(q_emb, axis=1)
        # q_sample_emb = tf.unstack(q_s_emb, axis = 0)

        filter_size = [3, 4, 5]
        att = []
        merge = []
        q_merge = []
        with tf.variable_scope("cnnencoder"):
         for filter in filter_size:
            step_merge = []
            step_att = []
            q_step_merge = []
            q_step_att = []
            with tf.variable_scope("cnnencoder-%s" % filter):
                for i in range(N):
                    conv_branch = tf.layers.conv1d(
                        inputs=c_s_emb[i],
                        # use as many filters as the hidden size
                        filters=100,
                        kernel_size=[filter],
                        use_bias=True,
                        activation=tf.nn.relu,
                        trainable=True,
                        padding='SAME',
                        name='conv_' + str(filter),
                        reuse=tf.AUTO_REUSE
                    )
                    # tf.get_variable_scope().reuse_variables()
                    step_att.append(conv_branch)
                    # pool over the words to obtain: [first_dim x 1* hidden_size]
                    pool_branch = tf.reduce_max(conv_branch, axis=1)
                    pool_branch = dropout(pool_branch,keep_prob=config.keep_prob, is_train=self.is_train)
                    step_merge.append(pool_branch)

            batch_merge = tf.stack(step_merge, axis=0)
            merge.append(batch_merge)
            # batch_merge_q = tf.stack(q_step_merge, axis = 0)
            # q_merge.append(batch_merge_q)

            con = tf.concat(merge, axis=-1)
            # q_con = tf.concat(q_merge, axis = -1)
            #
            # attention_vis = tf.stack(att, axis=0)
            # attention_vis = tf.reduce_mean(attention_vis, axis=0)
            # cnn_out.append(con)
            # c_sen_emb = tf.concat(con, axis = 0)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=con.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            print('passage-encoder')
            c_s = rnn(con, seq_len=self.c_p_len)
            # q = rnn(q_emb, seq_len=self.q_len)
        with tf.variable_scope("qencode"):
            with tf.variable_scope("encoding"):
                rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=q_emb.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)

                q = rnn(q_emb, seq_len=self.q_len)
        self.q_enc = q
        print('qc_att')

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c_s, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)

            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            self.att_s = rnn(qc_att, seq_len=self.c_p_len)


        # print('pointer')
        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train,is_sentence=True)

            logits1 = pointer(init, self.att_s, d, self.c_p_mask)
            self.lo = logits1
        with tf.variable_scope("predict"):
            self.outer = tf.nn.softmax(logits1)
            self.yp = tf.argmax(self.outer, axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits1, labels=tf.stop_gradient(self.y_slice))
            self.out1 = tf.nn.top_k(self.outer, config.k).values
            self.policy = tf.nn.top_k(self.outer, 1).values
            self.policy = tf.reduce_sum(tf.nn.top_k(self.outer, config.k).values, axis=-1, keepdims=True)
            self.policy_log_part = tf.log(self.policy)
            #self.loss = tf.reduce_mean(-1 * self.policy_log_part * self.reward)
            reward = self.advantage
            reward_mean, reward_var = tf.nn.moments(reward, axes=[0])

            reward_std = tf.sqrt(reward_var)+1e-6
            self.reward_mean = reward_mean
            self.reward_var = reward_std
            reward = tf.div(reward-reward_mean, reward_std)

            self.final_reward = reward-self.baseline
            self.loss = tf.reduce_mean(-1 * self.policy_log_part * self.advantage)
            #self.loss = tf.reduce_mean(losses*self.final_reward)


    def learn(self,qa_id,q,cs,y,qh,csh,ce,qe,ct,qt):
        feed_dict = {self.qa_id: qa_id, self.q: q, self.cs:cs, self.y: y, self.qh: qh,self.csh : csh,
                    self.ce: ce, self.qe: qe, self.ct: ct, self.qt: qt}

        loss, train_op, qaid, outer, att_s = self.sess.run([self.loss, self.train_op, self.qa_id, self.outer, self.att_s],feed_dict = feed_dict)

        return loss, qa_id, outer, att_s


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
