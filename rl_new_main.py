import tensorflow as tf

import ujson as json
import numpy as np
from tqdm import tqdm
import os
from model import select_model
from new_span_model import span_model
from evalute_func import convert_tokens,evaluate, compute_reward, convert_sentence,compute_question_aware_reward
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from prepro import DataProcessor
from new_baseline import critic
def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    dp_train = DataProcessor('train', config)
    dp_dev = DataProcessor('dev', config)
    #dp_dev = dp_train
    dev_total = dp_dev.get_data_size()

    print("Building model...")

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr
    lr_span = 0.1
    lr_critic = 0.1
    train_batch_num = int(np.floor(dp_train.num_samples / config.batch_size)) - 1
    batch_no = 0
    SN = config.k
    with tf.Session(config=sess_config) as sess:
        selector = select_model(sess, config, word_mat, char_mat)
        spaner = span_model(sess, config, word_mat, char_mat)
        criticer = critic(sess,config)

        writer = tf.summary.FileWriter(config.new_RL_log_dir)
        sess.run(tf.global_variables_initializer())

        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_restore = variables[0:334]
        saver_model = tf.train.Saver(variables_to_restore)
        saver_model.restore(sess, tf.train.latest_checkpoint(config.span_save_dir_trained))
        saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint(config.new_RL_save_dir))
        sess.run(tf.assign(selector.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(spaner.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(selector.lr, tf.constant(lr, dtype=tf.float32)))
        sess.run(tf.assign(selector.global_step, tf.constant(0, dtype=tf.int32)))
        sess.run(tf.assign(spaner.global_step, tf.constant(0, dtype=tf.int32)))
        sess.run(tf.assign(spaner.lr, tf.constant(lr_span, dtype=tf.float32)))
        sess.run(tf.assign(criticer.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(criticer.lr, tf.constant(lr_critic, dtype=tf.float32)))
        sen_len_limit = config.sen_len
        question_type_num = 19
        for batch_time in tqdm(range(1, config.num_steps + 1)):

            tensor_dict, _ = dp_train.get_train_batch(batch_no%train_batch_num)
            sess.run(tf.assign(selector.is_train, tf.constant(False, dtype=tf.bool)))
            sess.run(tf.assign(spaner.is_train, tf.constant(False, dtype=tf.bool)))
            qa_id, yp, outer, para_encode, para_enc_mask, q_emb, sen_emb, lo, q_enc, para_enc,policy,out1= sess.run(
                [selector.qa_id, selector.yp, selector.outer, selector.att_s, selector.c_p_mask,
                 selector.q_emb, selector.c_emb,selector.lo, selector.q_enc, selector.att_s,selector.policy,
                 selector.out1],
                                              feed_dict={   selector.qa_id: tensor_dict['ids'],
                                                            selector.q: tensor_dict['ques_idxs'],
                                                            selector.cs: tensor_dict['context_s_idxs'],
                                                            selector.y: tensor_dict['y'],
                                                            selector.ce: tensor_dict['context_s_exist_tag'],
                                                            selector.ct: tensor_dict['context_type_tag'],
                                                            selector.qt: tensor_dict['ques_type_tag']
                                                            })
            np.savetxt("policy.txt", policy)
            np.savetxt("out.txt",out1)
            select_sentence = []
            sentences_len = []
            q = []

            sentences_cs = []
            sentences_ce = []
            sentences_ct = []
            sentence_index = []
            for i in range(config.batch_size):
                ques = np.zeros([config.ques_limit, q_emb.shape[-1]], np.float32)
                ques[:q_emb.shape[-2]] = q_emb[i]
                q.append(ques)
                sentences = []
                sentence_len = []
                sum = tensor_dict['sentence_num'][i]
                indexs = np.argsort(-outer[i])
                sentence_index.append(indexs[:config.k])
                #RL change
                # indexs = np.random.choice(a=outer.shape[0], size=config.k, replace=False, p=outer)

                for j in range(config.k):
                    top_index = indexs[j]
                    sentence = np.zeros([config.sen_len, sen_emb.shape[-1]], np.float32)
                    sentence[:sen_emb.shape[-2]] = sen_emb[i][top_index]
                    sentences.append(sentence)
                    sentence_length = np.arange(sum[indexs[j]-1],sum[indexs[j]-1]+config.sen_len,1)
                    sentence_len.append(sentence_length)
                sentence_len = np.array(sentence_len)
                sentences = np.array(sentences)
                select_sentence.append(sentences)
                sentences_len.append(sentence_len)
            select_sentences = np.array(select_sentence)
            sentences_lens = np.array(sentences_len)
            q =np.array(q)

            global_step = sess.run(spaner.global_step) + 1
            qa_id, yp1, yp2, present  = sess.run([spaner.qa_id, spaner.yp1, spaner.yp2,spaner.present],
                                                            feed_dict={spaner.qa_id: tensor_dict['ids'],

                                                                       spaner.para: tensor_dict['para_idxs'],
                                                                       spaner.para_e: tensor_dict['para_exist_tag'],
                                                                       spaner.para_t: tensor_dict['para_type_tag'],

                                                                       spaner.q: q,
                                                                       spaner.sentence: select_sentences,
                                                                       spaner.sentence_index: sentences_lens,
                                                                       spaner.outer: outer,
                                                                       spaner.para_enc: para_encode,
                                                                       spaner.para_enc_mask: para_enc_mask,
                                                                       spaner.y1: tensor_dict['y1'],
                                                                       spaner.y2: tensor_dict['y2']
                                                                       })

            answer_dict, _ ,sentence_dict= convert_tokens(train_eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())

            reward = compute_reward(train_eval_file, answer_dict, sentence_dict)
            critic_train_op, critic_loss, reward_Diff = sess.run([criticer.train_op,criticer.loss, criticer.rewardDiff],
                                              feed_dict={   criticer.input: present,
                                                            criticer.reward: reward
                                                            })

            sess.run(tf.assign(spaner.is_train, tf.constant(True, dtype=tf.bool)))
            qa_id, train_op, loss, yp1, yp2,loss3  = sess.run([spaner.qa_id,spaner.train_op, spaner.loss_span, spaner.yp1, spaner.yp2,spaner.loss3],
                                                            feed_dict={spaner.qa_id: tensor_dict['ids'],

                                                                       spaner.para: tensor_dict['para_idxs'],
                                                                       spaner.para_e: tensor_dict['para_exist_tag'],
                                                                       spaner.para_t: tensor_dict['para_type_tag'],

                                                                       spaner.q: q,
                                                                       spaner.sentence: select_sentences,
                                                                       spaner.sentence_index: sentences_lens,
                                                                       spaner.outer: outer,
                                                                       spaner.para_enc: para_encode,
                                                                       spaner.para_enc_mask: para_enc_mask,
                                                                       spaner.y1: tensor_dict['y1'],
                                                                       spaner.y2: tensor_dict['y2'],
                                                                       spaner.reward_Diff: reward_Diff
                                                                       })

            np.savetxt("reward.txt", reward)

            #np.savetxt("advantage1.txt", advantage)

            # print(reward_var)
            # print(reward_mean)
            print(loss3)
            print("spanloss"+str(loss))
            # print("critic_loss"+str(critic_loss))
            batch_no = batch_no + 1
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            if batch_time % config.checkpoint == 0:
                sess.run(tf.assign(selector.is_train, tf.constant(False, dtype=tf.bool)))
                sess.run(tf.assign(spaner.is_train, tf.constant(False, dtype=tf.bool)))

                metrics,metrics_sele, summ = evaluate_span_batch(
                    config, selector, spaner, dev_total // config.batch_size, dev_eval_file, sess, "dev", dp_dev)
                for s in summ:
                    writer.add_summary(s, global_step)
                print('epoch'+str(global_step))
                print('dev_loss:'+str(metrics["loss"]))
                print('F1:'+str(metrics["f1"]))
                print('em:' + str(metrics["exact_match"]))
                print('F:' + str(metrics_sele["f"]))

                sess.run(tf.assign(spaner.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(spaner.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.new_RL_save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, dp):
    answer_dict = {}
    losses = []
    qa_ids = []
    yps = []

    for batch_num in tqdm(range(0, num_batches)):
        tensor_dict, _ = dp.get_train_batch(batch_num, is_test=True)
        qa_id, loss, yp, = sess.run(
            [model.qa_id, model.loss, model.yp], feed_dict={model.qa_id: tensor_dict['ids'],
                                                            model.q: tensor_dict['ques_idxs'],
                                                            model.cs: tensor_dict['context_s_idxs'],
                                                            model.y: tensor_dict['y'],
                                                            model.qh: tensor_dict['ques_char_idxs'],
                                                            model.csh: tensor_dict['context_s_char_idxs'],
                                                            model.ce: tensor_dict['context_s_exist_tag'],
                                                            model.qe: tensor_dict['ques_s_exist_tag'],
                                                            model.ct: tensor_dict['context_type_tag'],
                                                            model.qt: tensor_dict['ques_type_tag']
                                                            })
        qa_ids.append(qa_id)
        yps.append(yp)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = exact_match_sentence(eval_file, qa_ids, yps)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])

    f_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f".format(data_type), simple_value=metrics["f"]), ])
    return metrics, [loss_sum, f_sum]



def evaluate_span_batch(config,selector, spaner, num_batches, eval_file, sess, data_type, dp):
    answer_dict = {}
    losses = []
    qa_ids = []
    yps = []
    SN = config.k
    sen_len_limit = config.sen_len
    question_type_num = 19
    reward = np.zeros([config.batch_size], np.float32)
    reward_Diff = np.zeros([config.batch_size,2],np.float32)
    for batch_num in tqdm(range(0, num_batches)):
        tensor_dict, _ = dp.get_train_batch(batch_num, is_test=True)
        qa_id, yp, outer, para_encode, para_enc_mask, q_emb, sen_emb, lo = sess.run(
            [selector.qa_id, selector.yp, selector.outer, selector.att_s, selector.c_p_mask, selector.q_emb,
             selector.c_emb, selector.lo],
            feed_dict={selector.qa_id: tensor_dict['ids'],
                       selector.q: tensor_dict['ques_idxs'],
                       selector.cs: tensor_dict['context_s_idxs'],
                       selector.y: tensor_dict['y'],
                       selector.ce: tensor_dict['context_s_exist_tag'],
                       selector.ct: tensor_dict['context_type_tag'],
                       selector.qt: tensor_dict['ques_type_tag'],
                       selector.advantage: reward,
                       selector.baseline: reward
                       })
        np.savetxt("lo.txt", lo)
        qa_ids.append(qa_id)
        yps.append(yp)

        select_sentence = []
        sentences_len = []
        q = []

        sentences_cs = []
        sentences_ce = []
        sentences_ct = []

        for i in range(config.batch_size):
            ques = np.zeros([config.ques_limit, q_emb.shape[-1]], np.float32)
            ques[:q_emb.shape[-2]] = q_emb[i]
            q.append(ques)
            sentences = []
            sentence_len = []
            sum = tensor_dict['sentence_num'][i]
            indexs = np.argsort(-outer[i])
            for j in range(config.k):
                top_index = indexs[j]
                sentence = np.zeros([config.sen_len, sen_emb.shape[-1]], np.float32)
                sentence[:sen_emb.shape[-2]] = sen_emb[i][top_index]
                sentences.append(sentence)
                sentence_length = np.arange(sum[indexs[j] - 1], sum[indexs[j] - 1] + config.sen_len, 1)
                sentence_len.append(sentence_length)
            sentence_len = np.array(sentence_len)
            sentences = np.array(sentences)
            select_sentence.append(sentences)
            sentences_len.append(sentence_len)
        select_sentences = np.array(select_sentence)
        sentences_lens = np.array(sentences_len)
        q = np.array(q)

        qa_id, loss, train_op_span, yp1, yp2 = sess.run(
            [spaner.qa_id, spaner.loss_span, spaner.train_op, spaner.yp1, spaner.yp2],
            feed_dict={spaner.qa_id: tensor_dict['ids'],
                       spaner.para: tensor_dict['para_idxs'],
                       spaner.para_e: tensor_dict['para_exist_tag'],
                       spaner.para_t: tensor_dict['para_type_tag'],
                       spaner.q: q,
                       spaner.sentence: select_sentences,
                       spaner.sentence_index: sentences_lens,
                       spaner.outer: outer,
                       spaner.para_enc: para_encode,
                       spaner.para_enc_mask: para_enc_mask,
                       spaner.y1: tensor_dict['y1'],
                       spaner.y2: tensor_dict['y2'],
                       spaner.reward_Diff:reward_Diff
                       })

        answer_dict_, _,_ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    metrics_sele = exact_match_sentence(eval_file, qa_ids, yps)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)

    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    f_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f".format(data_type), simple_value=metrics_sele["f"]), ])
    return metrics,metrics_sele, [loss_sum, f1_sum, em_sum,f_sum]




def exact_match_sentence(eval_file, qa_id, yp):
    exact = 0
    total = 0
    for id, y in zip(qa_id, yp):

        for id1, y1 in zip(id.tolist(), y.tolist()):
            ground_index = eval_file[str(id1)]["answer_sen_id"]
            total = total+1
            if y1 == int(ground_index):
                exact = exact+1
    f = 100*exact/total
    return {'f': f}



def exact_match_sentence_topk(eval_file, qa_id, yp, outer,save_file):
    exact = 0
    total = 0
    k = 3
    with open(save_file, "w") as fh:
        predict_index ={}
        for id, y, out in zip(qa_id, yp,outer):

            for id1, y1, out1 in zip(id.tolist(), y.tolist(), out.tolist()):
                ground_index = eval_file[str(id1)]["answer_sen_id"]
                total = total+1
                predict = np.argsort(-np.array(out1))
                predict_index[str(id1)] = {
                    "context":eval_file[str(id1)]["context"], "answer_sen": out1, "uuid": eval_file[str(id1)]["uuid"]}

                for i in range(k):
                    if(predict[i] == ground_index):
                        exact = exact+1
                        break
        json.dump(predict_index, fh)
    f = 100*exact/total
    print(exact)
    print(total)
    return {'f': f}



def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
            eval_file = json.load(fh)
    with open(config.dev_example, "r") as fh:
            example_file = json.load(fh)
    # # with open(config.test_eval_file, "r") as fh:
    # #     eval_file = json.load(fh)
    #
    # # dp_test = DataProcessor('test', config)
    # # total = dp_test.get_data_size()
    dp_test = DataProcessor('test', config, is_test=True)
    total = dp_test.get_data_size()
    print("Loading model...")



    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        selector = select_model(sess, config, word_mat, char_mat)
        spaner = span_model(sess, config, word_mat, char_mat)
        sess.run(tf.global_variables_initializer())
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_restore = variables
        #variables_to_restore = [v for v in variables if v.name.split('/')[0] != 'output']

        saver = tf.train.Saver()
        # saver = tf.train.Saver()
        reward_diff = np.ones([config.batch_size,2],np.float32)
        saver.restore(sess, tf.train.latest_checkpoint(config.new_RL_save_dir))

        sess.run(tf.assign(selector.is_train, tf.constant(False, dtype=tf.bool)))
        sess.run(tf.assign(spaner.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // config.batch_size )):

            tensor_dict, _ = dp_test.get_train_batch(step, is_test=True)

            qa_id, yp, outer, para_encode, para_enc_mask, q_emb, sen_emb, lo = sess.run(
                [selector.qa_id, selector.yp, selector.outer, selector.att_s, selector.c_p_mask, selector.q_emb, selector.c_emb,selector.lo],
                                              feed_dict={   selector.qa_id: tensor_dict['ids'],
                                                            selector.q: tensor_dict['ques_idxs'],
                                                            selector.cs: tensor_dict['context_s_idxs'],
                                                            selector.y: tensor_dict['y'],
                                                            selector.ce: tensor_dict['context_s_exist_tag'],
                                                            selector.ct: tensor_dict['context_type_tag'],
                                                            selector.qt: tensor_dict['ques_type_tag']
                                                            })
            select_sentence = []
            sentences_len = []
            q = []
            for i in range(config.batch_size):
                ques = np.zeros([config.ques_limit, q_emb.shape[-1]], np.float32)
                ques[:q_emb.shape[-2]] = q_emb[i]
                q.append(ques)
                sentences = []
                sentence_len = []
                sum = tensor_dict['sentence_num'][i]
                indexs = np.argsort(-outer[i])
                for j in range(config.k):
                    top_index = indexs[j]
                    sentence = np.zeros([config.sen_len, sen_emb.shape[-1]], np.float32)
                    sentence[:sen_emb.shape[-2]] = sen_emb[i][top_index]
                    sentences.append(sentence)
                    sentence_length = np.arange(sum[indexs[j]-1],sum[indexs[j]-1]+config.sen_len,1)
                    sentence_len.append(sentence_length)
                sentence_len = np.array(sentence_len)
                sentences = np.array(sentences)
                select_sentence.append(sentences)
                sentences_len.append(sentence_len)
            select_sentences = np.array(select_sentence)
            sentences_lens = np.array(sentences_len)
            q =np.array(q)
            qa_id, loss, train_op_span, yp1, yp2= sess.run([spaner.qa_id, spaner.loss_span, spaner.train_op,spaner.yp1, spaner.yp2],
                                                            feed_dict={spaner.qa_id: tensor_dict['ids'],

                                                                       spaner.para: tensor_dict['para_idxs'],
                                                                       spaner.para_e: tensor_dict['para_exist_tag'],
                                                                       spaner.para_t: tensor_dict['para_type_tag'],
                                                                        spaner.reward_Diff:reward_diff,
                                                                       spaner.q: q,
                                                                       spaner.sentence: select_sentences,
                                                                       spaner.sentence_index: sentences_lens,
                                                                       spaner.outer: outer,
                                                                       spaner.para_enc: para_encode,
                                                                       spaner.para_enc_mask: para_enc_mask,
                                                                       spaner.y1: tensor_dict['y1'],
                                                                       spaner.y2: tensor_dict['y2']
                                                                       })

            answer_dict_, remapped_dict_ ,_= convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open(config.answer_file, "w") as fh:
            json.dump(remapped_dict, fh)
        print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))


