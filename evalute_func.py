import tensorflow as tf
import numpy as np
import re
from collections import Counter
import string
beta = 0.5
alpha = 0.5
import nltk
from nltk.tokenize import sent_tokenize
from prepro import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def convert_tokens(eval_file, qa_id, pp1, pp2 ):
    answer_dict = {}
    remapped_dict = {}
    sentence_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]

        start_idx = spans[p1][0]
        end_idx = spans[p2-1][1]
        p11 = p1-7
        p22 = p2-1+7
        if(p11<0):
            p11 = 0
        if(p22>=len(spans)):
            p22 = len(spans)-1
        startsen_idx = spans[p11][0]
        endsen_idx = spans[p22][1]

        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
        sentence_dict[str(qid)] = context[startsen_idx: endsen_idx]
    return answer_dict, remapped_dict,sentence_dict


def convert_sentence(eval_file, qa_id, indexs):
    sentence_dict = {}

    for qid, index in zip(qa_id, indexs):
        sentence = []
        context = sent_tokenize(eval_file[str(qid)]["context"])
        uuid = eval_file[str(qid)]["uuid"]
        ques = eval_file[str(qid)]["question"]
        for i in range(len(index)):
            try:
                sentence.append(context[index[i]])
            except IndexError:
                break
        sentence_dict[str(qid)] = sentence

    return sentence_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        # print(ground_truths)
        # print(prediction)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    print("total"+str(total))
    print("em" + str(exact_match))
    print("f1" + str(f1))
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def compute_question_aware_reward(eval_file, answer_dict, sentence_dict):
    beta = 0.5
    reward = []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"]
        sentence = sentence_dict[key]
        question = eval_file[key]["question"]
        prediction = value
        r_qc = compute_bleu(question, sentence)
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        if (f1 == 0) and (r_qc ==0):
           r_all = 0
        else:
            if (f1 ==1):
                r_all = 1
            else:
                r_all = (1 + pow(beta, 2)) * r_qc * f1 / ((pow(beta,2)*r_qc)+f1)
                if(f1==0):
                    r_all = pow(beta,2)*r_qc
        reward.append(r_all)
    f1s = np.array(reward)
    return f1s


def compute_reward(eval_file, answer_dict, sentence_dict):

    reward = []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"]
        sentence = sentence_dict[key]
        question = eval_file[key]["question"]

        prediction = value
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        reward.append(f1)

    f1s = np.array(reward)
    return f1s


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth.lower())
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def bleu_v1(candidate_token, reference_token):

    count = 0
    for token in candidate_token:
        if token in reference_token:
            count += 1
    a = count
    b = len(candidate_token)
    return a/b

def compute_bleu(candidate, reference):

    max_count = 0
    candidate_token = word_tokenize(candidate)[:-1]
    count_array = []
    filtered_sentence = [w for w in candidate_token if not w in stop_words]
    count = 0
    reference_token = word_tokenize(reference)
    for token in filtered_sentence:
            if token in reference_token:
                count += 1

    a = count
    b = len(filtered_sentence)
    if (b==0):
        bleu = 0
    else:
        bleu = a/b
    return bleu