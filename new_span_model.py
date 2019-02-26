# xun lian hao de c_emb he q_emb

import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, dense, softmax_mask, \
    dot_attention_sentence, ptr_net_span
from func import linear, Highway, dense_summ
import numpy as np



class span_model(object):

    def __init__(self, sess, config, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.sess = sess
        self.global_step = tf.get_variable('global_step_span', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        ques_limit = config.ques_limit if trainable else config.test_ques_limit
        self.k = config.k
        N = config.batch_size
        SN = self.k
        W = config.glove_dim
        d = config.hidden
        self.qa_id = tf.placeholder(tf.int32, [config.batch_size])
        self.q = tf.placeholder(tf.float32, [config.batch_size, ques_limit, W + 19]) # change given

        self.qe = tf.placeholder(tf.float32, [config.batch_size, ques_limit, ques_limit])
        self.qt = tf.placeholder(tf.float32, [config.batch_size, ques_limit, 19])
        self.qh = tf.placeholder(tf.int32, [config.batch_size, ques_limit, config.char_limit])

        self.sentence = tf.placeholder(tf.float32, [None, self.k, config.sen_len, W + d + 3 + 19])
        self.sentence_index = tf.placeholder(tf.int32, [None, self.k, config.sen_len])

        self.sentence_h = tf.placeholder(tf.int32, [config.batch_size, SN, config.sen_len, config.char_limit])
        self.sentence_e = tf.placeholder(tf.float32, [config.batch_size, SN, config.sen_len, 3])
        self.sentence_s = tf.placeholder(tf.int32, [None, SN, config.sen_len])
        self.sentence_t = tf.placeholder(tf.float32, [config.batch_size, SN, config.sen_len, 19])
        # self.sentence_len = tf.placeholder(tf.int32, [None, self.k])

        self.outer = tf.placeholder(tf.float32, [N, None])  # the probability of sentence

        # self.sentence_h_len = tf.reshape(tf.reduce_sum(
        #      tf.cast(tf.cast(tf.reshape(self.sentence_h_slice, [N, SN*self.c_s_maxlen, CL]), tf.bool), tf.int32), axis=2), [-1])
        # self.qh_len = tf.reshape(tf.reduce_sum(
        #      tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.para = tf.placeholder(tf.int32, [config.batch_size, config.para_limit])
        self.parah = tf.placeholder(tf.int32, [config.batch_size, config.para_limit, config.char_limit])
        self.para_e = tf.placeholder(tf.float32, [config.batch_size, config.para_limit, 3])
        self.para_t = tf.placeholder(tf.float32, [config.batch_size, config.para_limit, 19])

        self.y1 = tf.placeholder(tf.float32, [config.batch_size, config.para_limit])
        self.y2 = tf.placeholder(tf.float32, [config.batch_size, config.para_limit])

        self.para_enc = tf.placeholder(tf.float32, [config.batch_size, None, None])
        self.para_enc_mask = tf.placeholder(tf.float32, [config.batch_size, None])
        self.reward_Diff = tf.placeholder(tf.float32, [None, 2])
        # self.att_s_mask = tf.cast(tf.reduce_sum(self.att_s, axis=2), tf.bool)
        # input other sentence-level embedding to caculate probability
        self.sentence_mask = tf.cast(tf.reduce_sum(tf.cast(tf.cast(self.sentence,tf.bool),tf.int32),axis=3), tf.bool)#change
        #self.sentence_mask = tf.cast(self.sentence_s, tf.bool)
        self.sentence_len = tf.reduce_sum(tf.cast(self.sentence_mask, tf.int32), axis=2)
        self.sentence_maxlen = tf.reduce_max(tf.reduce_max(self.sentence_len))

        self.sentence_index_slice = tf.slice(self.sentence_index, [0, 0, 0], [N, self.k, self.sentence_maxlen])
        self.sentence_slice = tf.slice(self.sentence, [0, 0, 0, 0],
                                       [N, SN, self.sentence_maxlen, W + d + 3 + 19])  # CHANGE
        self.sentence_mask = tf.slice(self.sentence_mask, [0, 0, 0], [N, SN, self.sentence_maxlen])

        #self.q_mask = tf.cast(self.q, tf.bool)
        self.q_mask = tf.cast(tf.reduce_sum(tf.cast(tf.cast(self.q,tf.bool),tf.int32),axis=2), tf.bool)#change
        # self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.para_mask = tf.cast(self.para, tf.bool)
        self.para_len = tf.reduce_sum(tf.cast(self.para_mask, tf.int32), axis=1)
        self.para_maxlen = tf.reduce_max(self.para_len)

        self.is_train = tf.get_variable(
            "is_train_span", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat_span", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat_span", initializer=tf.constant(char_mat, dtype=tf.float32))

        # self.c_mask = tf.cast(self.c, tf.bool)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.q_slice = tf.slice(self.q, [0, 0, 0], [N, self.q_maxlen,W+19])  # CHANGE
            self.sentence_s_slice = tf.slice(self.sentence_s, [0, 0, 0], [N, SN, self.sentence_maxlen])
            self.sentence_e_slice = tf.slice(self.sentence_e, [0, 0, 0, 0], [N, SN, self.sentence_maxlen, 3])
            self.sentence_t_slice = tf.slice(self.sentence_t, [0, 0, 0, 0], [N, SN, self.sentence_maxlen, 19])
            self.qe_slice = tf.slice(self.qe, [0, 0, 0], [N, self.q_maxlen, self.q_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            # self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh_slice = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])

            self.qt_slice = tf.slice(self.qt, [0, 0, 0], [N, self.q_maxlen, 19])

            self.para_slice = tf.slice(self.para, [0, 0], [N, self.para_maxlen])
            self.para_mask = tf.slice(self.para_mask, [0, 0], [N, self.para_maxlen])

            self.para_e_slice = tf.slice(self.para_e, [0, 0, 0], [N, self.para_maxlen, 3])
            self.para_t_slice = tf.slice(self.para_t, [0, 0, 0], [N, self.para_maxlen, 19])

            self.y1_slice = tf.slice(self.y1, [0, 0], [N, self.para_maxlen])
            self.y2_slice = tf.slice(self.y2, [0, 0], [N, self.para_maxlen])

            self.p_maxnum = tf.reduce_max(tf.reduce_sum(tf.cast(self.para_enc_mask, tf.int32), axis=1))
            self.para_enc_slice = tf.slice(self.para_enc, [0, 0, 0], [N, self.p_maxnum, 2 * config.hidden])
            self.para_enc_mask_slice = tf.slice(self.para_enc_mask, [0, 0], [N, self.p_maxnum])

            self.parah_slice = tf.slice(self.parah, [0, 0, 0], [N, self.para_maxlen, CL])
            self.parah_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.parah_slice, tf.bool), tf.int32), axis=2),
                                        [-1])

            align_q_emb = tf.ones([N, ques_limit, d], tf.float32)
            q_e_emb = tf.ones([N, ques_limit, 3], tf.float32)
            self.align_q_emb_slice = tf.slice(align_q_emb, [0, 0, 0], [N, self.q_maxlen, d])
            self.q_e_slice = tf.slice(q_e_emb, [0, 0, 0], [N, self.q_maxlen, 3])
        # else:
        # self.c_s_maxlen, self.q_maxlen = config.para_len, config.ques_limit

        self.ptrspan()

        if trainable:
            self.lr = tf.get_variable(
                "lr_span", shape=[], dtype=tf.float32, trainable=False)
            # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss_span)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ptrspan(self):
        config = self.config
        N, QL, CL, d, dc, dg = config.batch_size, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden

        gru = cudnn_gru if config.use_cudnn else native_gru
        SN = self.k
        W = config.glove_dim
        d = config.hidden

        print('embedding part')

        with tf.name_scope("word"):
            para_emb = tf.nn.embedding_lookup(self.word_mat, self.para_slice)
            c_emb = self.sentence_slice
            q_emb = self.q_slice

        with tf.name_scope("para_encode"):

            para_emb_linear = tf.layers.dense(para_emb, d, use_bias=False, kernel_initializer=tf.ones_initializer(),
                                              trainable=self.is_train, name='para_emb_line')
            q_emb_linear = tf.layers.dense(q_emb, d, use_bias=False, kernel_initializer=tf.ones_initializer(),
                                           trainable=self.is_train, name='q_emb_line')
            align_pq = tf.matmul(para_emb_linear, tf.transpose(q_emb_linear, [0, 2, 1]))
            pq_mask = tf.tile(tf.expand_dims(self.q_mask, axis=1), [1, self.para_maxlen, 1])
            align_pq = tf.nn.softmax(softmax_mask(align_pq, pq_mask))
            align_para_emb = tf.matmul(align_pq, q_emb_linear)
            para_emb_concat = tf.concat([para_emb, align_para_emb, self.para_e_slice, self.para_t_slice], axis=2)
            self.para_emb = para_emb_concat


        print('encode-part')
        # c_emb = self.sentence_slice

        c_emb_sen = tf.unstack(c_emb, axis=1)
        sentence_len = tf.unstack(self.sentence_len, axis=1)
        c_s = []
        with tf.variable_scope("sentence_encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb_sen[0].get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)

            print('passage-encoder')
            for i in range(SN):
                c_s_emb = rnn(c_emb_sen[i], seq_len=sentence_len[i], concat_layers=False)

                c_s.append(c_s_emb)
            para_gru = rnn(para_emb_concat, seq_len=self.para_len, concat_layers=False)

        with tf.variable_scope("q_encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=q_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            q = rnn(q_emb, seq_len=self.q_len, concat_layers=False)

        # c_s_h = []
        # with tf.variable_scope("highway_encoding",reuse = tf.AUTO_REUSE):
        #     highway = Highway(hidden_size=2*d,is_train=self.is_train)
        #     for i in range(SN):
        #         c_s_highway = highway(c_s[i])
        #         c_s_h.append(c_s_highway)
        #     para_gru = highway(para_gru)
        #     q = highway(q)
        # c_s = c_s_h

        print('qc_att')
        self.c_s = c_s
        self.para_gru = para_gru
        qc_att = []
        sen_mask = tf.unstack(self.sentence_mask, axis=1)
        with tf.variable_scope("sentence_attention", reuse=tf.AUTO_REUSE):
            for i in range(SN):
                qc_att_sample = dot_attention(c_s[i], q, mask=self.q_mask, hidden=d,
                                              keep_prob=config.keep_prob, is_train=self.is_train)
                qc_att.append(qc_att_sample)

            para_att = dot_attention(para_gru, q, mask=self.q_mask, hidden=d,
                                     keep_prob=config.keep_prob, is_train=self.is_train)

        att_s = []
        with tf.variable_scope("sentence_qcatt_rnn"):
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att[0].get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            for i in range(SN):
                att_s_single = rnn(qc_att[i], seq_len=sentence_len[i])
                att_s.append(att_s_single)
            para_s = rnn(para_att, seq_len=self.para_len)

        self.sentence_att = qc_att
        self.para_att = para_att

        self_att = []

        with tf.variable_scope("sentence_cpattention", reuse=tf.AUTO_REUSE):
            for i in range(SN):
                self_att_single = dot_attention(
                    att_s[i], para_s, mask=self.para_mask, hidden=d, keep_prob=config.keep_prob,
                    is_train=self.is_train)
                self_att.append(self_att_single)

        with tf.variable_scope("para_selfattn"):
            # self.para_enc_slice, mask = self.para_enc_mask_slice,
            para_self_att = dot_attention(para_s, para_s, mask=self.para_mask, hidden=d,
                                          keep_prob=config.keep_prob, is_train=self.is_train)

        self.sentence_selfatt = self_att
        self.para_selfatt = para_self_att

        match = []
        with tf.variable_scope("sentence_cp_rnn"):
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att[0].get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            for i in range(SN):
                match_single = rnn(self_att[i], seq_len=sentence_len[i])
                match.append(match_single)
            para_match = rnn(para_self_att, seq_len=self.para_len)
        self.match = match

        dense_prob = []
        dense_con = []
        with tf.variable_scope("dense_prob", reuse=tf.AUTO_REUSE):
            for i in range(SN):
                sentence_con = tf.concat([c_s[i], att_s[i], match[i]], axis=2)
                prob = dense_summ(sentence_con, d, mask=sen_mask[i], keep_prob=config.keep_prob, is_train=self.is_train)
                dense_prob.append(prob)
                dense_con.append(sentence_con)
            # with tf.variable_scope("para_prob"):
            para_con = tf.concat([para_gru, para_s, para_match], axis=2)
            para_prob = dense_summ(para_con, d, mask=self.para_mask, keep_prob=config.keep_prob, is_train=self.is_train)
            dense_prob.append(para_prob)
            dense_prob = tf.concat(dense_prob, axis=1)
            self.topk = tf.nn.softmax(dense_prob)

        batch_nums = tf.range(0, limit=N)
        batch_nums = tf.expand_dims(batch_nums, 1)
        batch_nums = tf.tile(batch_nums, [1, self.sentence_maxlen])
        lo_shape = tf.constant([N, config.para_limit])

        sentence_index_slice = tf.unstack(self.sentence_index_slice, axis=1)
        # how to ensure the probability
        # sentence1,sentence2,setence3,q,para =?*4

        lo1 = []
        lo2 = []
        with tf.variable_scope("sentence_pointer", reuse=tf.AUTO_REUSE):

            self.init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                             keep_prob=config.keep_prob, is_train=self.is_train)
            pointer = ptr_net_span(batch=N, hidden=self.init.get_shape().as_list(
            )[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            indice_test = []
            lo1_test = []
            lo2_test = []
            present = []
            present_inp = []

            for i in range(SN):
                logits1, logits2, inp1, inp2 = pointer(self.init, dense_con[i], d, sen_mask[i])
                logits1 = logits1 * tf.cast(sen_mask[i], tf.float32)
                logits2 = logits2 * tf.cast(sen_mask[i], tf.float32)
                indice = tf.stack([batch_nums, sentence_index_slice[i]], axis=2)
                inp = tf.stack([inp1, inp2], axis=1)
                present.append(inp)
                present_inp.append(inp2)
                lo1_test.append(logits1)
                lo2_test.append(logits2)
                indice_test.append(indice)

            self.lo1 = lo1_test[0]
            self.lo2 = lo1_test[1]
            self.lo3 = lo1_test[2]

            lo1 = [tf.slice(tf.scatter_nd(in1, in2, lo_shape), [0, 0], [N, self.para_maxlen])
                   for (in1, in2) in zip(indice_test, lo1_test)]
            lo2 = [tf.slice(tf.scatter_nd(in1, in2, lo_shape), [0, 0], [N, self.para_maxlen])
                   for (in1, in2) in zip(indice_test, lo2_test)]

            with tf.variable_scope("para_pointer"):
                para_pointer = ptr_net_span(batch=N, hidden=self.init.get_shape().as_list(
                )[-1], keep_prob=config.keep_prob, is_train=self.is_train)
                para_lo1, para_lo2, inp1, inp2 = para_pointer(self.init, para_match, d, self.para_mask)
                present_para = tf.stack([inp1, inp2], axis=1)
                para_lo1 = softmax_mask(para_lo1, self.para_mask)
                para_lo2 = softmax_mask(para_lo2, self.para_mask)
            present.append(tf.tile(present_para,[1,1,3]))
            present_inp.append(inp2)
            lo1.append(para_lo1)
            lo2.append(para_lo2)
            self.lo4 = para_lo2
            self.present = tf.stack(present,axis=2)
            out_lo1 = tf.stack(lo1, axis=1)
            out_lo2 = tf.stack(lo2, axis=1)
            out_lo1 = (tf.expand_dims(self.topk, axis=2)) * out_lo1
            out_logits1 = tf.reduce_sum(out_lo1, axis=1)
            # out_logits1 = tf.slice(out_logits1, [0, 0], [N, self.para_maxlen])
            # out_logits1 = softmax_mask(out_logits1, self.para_mask)
            out_lo2 = (tf.expand_dims(self.topk, axis=2)) * out_lo2
            out_logits2 = tf.reduce_sum(out_lo2, axis=1)
            # out_logits2 = tf.slice(out_logits2, [0, 0], [N, self.para_maxlen])
            # out_logits2 = softmax_mask(out_logits2, self.para_mask)

            self.out_lo1 = out_lo1
            self.out_lo2 = out_logits1

            # out_logits1 = tf.nn.softmax(out_logits1)
            # out_logits2 = tf.nn.softmax(out_logits2)
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(out_logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(out_logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)

        with tf.variable_scope("predict"):

            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=out_logits1, labels=tf.stop_gradient(self.y1_slice))
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=out_logits2, labels=tf.stop_gradient(self.y2_slice))
            prob_y1 = tf.expand_dims(tf.reduce_max(tf.reduce_max(outer, axis=2), axis=1),axis=1)
            prob_y2 = tf.expand_dims(tf.reduce_max(tf.reduce_max(outer, axis=1), axis=1),axis=1)
            prob = tf.concat([prob_y1, prob_y2], axis=1)
            lossRL = -tf.log(prob)*self.reward_Diff
            self.out1 = losses

            self.out2 = losses2
            loss = tf.concat([tf.expand_dims(losses,axis=1), tf.expand_dims(losses2,axis=1)],axis=1)
            final_reward = loss*self.reward_Diff
            self.loss3 = tf.reduce_mean((losses + losses2))
            lam = config.lam
            self.loss_span = tf.reduce_mean(final_reward)
            # self.loss_span = lam*self.loss3 +(1-lam)*(tf.reduce_mean(lossRL))

