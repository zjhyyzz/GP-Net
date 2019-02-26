import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('wordnet')
from nltk import WordNetLemmatizer,PorterStemmer
wnl = WordNetLemmatizer()
import spacy
tag_token = spacy.load('en_core_web_sm')
porter = PorterStemmer()
nlp = spacy.blank("en")
flags = tf.flags
config = flags.FLAGS

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_sentence_tokens = [word_tokenize(sen) for sen in sent_tokenize(context)]
                context_pos_tag = []
                for sen in sent_tokenize(context):
                    tokens = tag_token.tokenizer(sen)
                    tag_token.tagger(tokens)
                    tag_token.entity(tokens)
                    sample = [[tokens[i].text, tokens[i].tag_, tokens[i].lemma_, tokens[i].ent_type_] for i in
                                   range(len(tokens))]
                    context_pos_tag.append(sample)
                context_s_chars = [[list(token) for token in sentokens] for sentokens in context_sentence_tokens]

                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    ys = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer['answer_end']
                        answer_sen_index = answer['answer_sen_index']
                        y1, y2 = answer_start, answer_end
                        y = answer_sen_index
                        # y = int(y1/config.sen_len)
                        # answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        # answer_span = []
                        # for idx, span in enumerate(spans):
                        #     if not (answer_end <= span[0] or answer_start >= span[1]):
                        #         answer_span.append(idx)
                        # y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                        ys.append(y)
                    example = {"context":context_tokens, "context_ch":context_chars, "context_tokens": context_sentence_tokens, "context_chars": context_s_chars, "ques_tokens": ques_tokens,
                               "context_pos_tag":context_pos_tag, "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s,"ys":ys, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "question":ques, "answer_sen_id": y,"answer_start":y1,"answer_end":y2, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
       config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word2idx_dict = None
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim, token2idx_dict=word2idx_dict)

    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim, token2idx_dict=char2idx_dict)

    # build_features(config, train_examples, "train",
    #                config.train_record_file, word2idx_dict, char2idx_dict)
    # dev_meta = build_features(config, dev_examples, "dev",
    #                           config.dev_record_file, word2idx_dict, char2idx_dict)
    #
    # test_meta = build_features(config, test_examples, "test",
    #                           config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    # save(config.dev_meta, dev_meta, message="dev meta")
    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.char2idx_file, char2idx_dict, message="char2idx")
    save(config.train_example, train_examples, message="train example")
    save(config.dev_example, dev_examples, message="dev example")
    save(config.test_example, test_examples, message="test example")
    # save(config.test_meta, test_meta, message="test meta")


from nltk.tag import StanfordNERTagger, StanfordPOSTagger
NER_tag = StanfordNERTagger('/home/cide/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','/home/cide/stanford-ner/stanford-ner.jar')
POS_tag = StanfordPOSTagger('/home/cide/stanford-pos/models/english-bidirectional-distsim.tagger','/home/cide/stanford-pos/stanford-postagger.jar')

#question_type: who, what...
#question_type_dict: 0-18
#question_types: numpy

question_type = ['who', 'what', 'where', 'when', 'which', 'how many', 'other']
question_type_small = ['who', 'what', 'where', 'when', 'which','how many']
 #+other
all_type = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT' ,'EVENT','WORK_OF_ART',
            'LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
question_type_num = len(all_type)+1
ques_type_dict = {}
for i in range(len(all_type)):
    ques_type_dict[all_type[i]] = i
ques_type_dict[''] = len(all_type)
who_type = ['PERSON']
what_type = ['NORP','FAC','ORG','GPE','LOC','PRODUCT' ,'EVENT','WORK_OF_ART',
            'LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
where_type = ['NORP', 'FAC', 'ORG', 'LOC', 'GPE']
when_type = ['EVENT', 'DATE', 'TIME']
how_many_type = ['PERCENT','MONEY','QUANTITY','CARDINAL']
which_type = all_type
other_type = all_type
question_type_spec = [who_type, what_type, where_type, when_type, which_type, how_many_type, other_type]
question_types = {}
for i in range(len(question_type)):
    ques_type = np.zeros([len(ques_type_dict)], dtype = np.float32)
    for j in range(len(question_type_spec[i])):
        ques_type[[ques_type_dict[question_type_spec[i][j]]]] = 1.0

    question_types[question_type[i]] = ques_type
question_types[''] = np.ones([len(ques_type_dict)])




class DataProcessor:
    def __init__(self, data_type, config, is_test=False):
        self.data_type = data_type
        self.config = config
        if data_type =='train':
            data_path = config.train_example
        else:
            data_path = config.dev_example
        #data_path = os.path.join('data', "{}_example.json".format(data_type))
        self.data = self.load_data(data_path)

        if os.path.isfile(config.word2idx_file):
            with open(config.word2idx_file, "r") as fh:
                self.word2idx_dict = json.load(fh)

        if os.path.isfile(config.char2idx_file):
            with open(config.char2idx_file, "r") as fh:
                self.char2idx_dict = json.load(fh)

        self.para_limit = config.para_limit
        self.ques_limit = config.ques_limit

        # paragraph length filter: (train only)
        if not is_test:
            self.data = [example for example in self.data if
                         len([x for j in example["context_tokens"] for x in j]) <= self.para_limit and
                         len(example["ques_tokens"]) <= self.ques_limit]
        else:
            newdata = []
            for example in self.data:
                if len([x for j in example["context_tokens"] for x in j])>self.para_limit:
                    longsum = 0
                    para = []
                    for sentence in example["context_tokens"] :
                        if(longsum<self.para_limit):
                            longsum = len(sentence) + longsum
                            if(longsum<self.para_limit):
                                para.append(sentence)
                            else:
                                para.append(sentence[:self.para_limit-longsum])
                        else:
                            break
                    example["context_tokens"] = para
                newdata.append(example)
            self.data = newdata
        self.num_samples = self.get_data_size()
        print("Loaded {} examples from {}".format(self.num_samples, data_type))

    def load_data(self, path):
        with open(path, 'r') as fh:
            data = json.load(fh)
        return data

    def get_data_size(self):
        return len(self.data)

    def _get_word(self,word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in self.word2idx_dict:
                return self.word2idx_dict[each]
        return 1

    def _get_char(self,char):
        if char in self.char2idx_dict:
            return self.char2idx_dict[char]
        return 1

    def get_train_batch(self, batch_no, is_test=False):
        config = self.config

        si = (batch_no * config.batch_size)
        ei = min(self.num_samples, si + config.batch_size)
        n = ei - si
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        sen_num_limit = config.test_sen_num if is_test else config.sen_num
        sen_len_limit = config.sen_len
        para_limit = config.para_limit
        sentence_num = np.zeros([n,sen_num_limit],dtype = np.int32)
        context_s_exist_tag = np.zeros([n, sen_num_limit, sen_len_limit, 3], dtype=np.float32)
        context_s_idxs = np.zeros([n, sen_num_limit, sen_len_limit], dtype=np.int32)
        context_s_char_idxs = np.zeros([n, sen_num_limit, sen_len_limit, char_limit], dtype=np.int32)
        context_type_tag = np.zeros([n, sen_num_limit, sen_len_limit, question_type_num], dtype=np.float32)
        ques_idxs = np.zeros([n, ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([n, ques_limit, char_limit], dtype=np.int32)
        ques_type_tag = np.zeros([n,ques_limit,question_type_num], dtype = np.float32)
        ques_s_exist_tag = np.zeros([n, ques_limit, ques_limit], dtype=np.float32)

        para_idxs = np.zeros([n, para_limit], dtype=np.int32)
        para_char_idxs = np.zeros([n, para_limit, char_limit], dtype=np.int32)
        para_type_tag = np.zeros([n, para_limit, question_type_num], dtype = np.float32)
        para_exist_tag = np.zeros([n, para_limit, 3], dtype=np.float32)

        y = np.zeros([n, sen_num_limit], dtype=np.float32)
        y1 = np.zeros([n, para_limit], dtype=np.float32)
        y2 = np.zeros([n, para_limit], dtype=np.float32)
        ids = np.zeros([n], dtype=np.int32)
        idxs = []
        count = 0
        tensor_dict = {}

        for data_index in range(si, ei):
            idxs.append(data_index)
            example = self.data[data_index]
            q_lemma1 = [wnl.lemmatize(word) for word in example["ques_tokens"]]
            q_lemma2 = [porter.stem(word) for word in example["ques_tokens"]]
            context_pos_tag = example["context_pos_tag"]
            sentence_sum = 0
            for i in range(len(example["context_tokens"])):
                if i == sen_num_limit:
                    break
                sentence_num[count][i] = sentence_sum+len(example["context_tokens"][i])
                sentence_sum = sentence_num[count][i]


            m = 0
            for i, sen_tokens in enumerate(example["context_tokens"]):
                if m == para_limit:
                    break
                for j, token in enumerate(sen_tokens):
                    para_idxs[count, m] = self._get_word(token)
                    if token in example["ques_tokens"]:
                        para_exist_tag[count, m, 0] = 1.0
                    if token in q_lemma1:
                        para_exist_tag[count, m, 1] = 1.0
                    if token in q_lemma2:
                        para_exist_tag[count, m, 2] = 1.0
                    try:
                        pos_index = ques_type_dict[context_pos_tag[i][j][3]]
                    except IndexError:
                        print(context_pos_tag[i][j][3])
                        print("k")
                    para_type_tag[count, m, pos_index] = 1.0
                    m = m+1

            # for m, token in enumerate(example["context"]):
            #
            #     if m == para_limit:
            #         break
            #
            #     para_idxs[count, m] = self._get_word(token)
            #     if token in example["ques_tokens"]:
            #             para_exist_tag[count, m, 0] = 1.0
            #     if token in q_lemma1:
            #             para_exist_tag[count, m, 1] = 1.0
            #     if token in q_lemma2:
            #             para_exist_tag[count, m, 2] = 1.0
            #     pos_index = ques_type_dict[context_pos_tag[i][j][3]]
            #     para_type_tag[count, m, pos_index] = 1.0


            for i, sen_tokens in enumerate(example["context_tokens"]):
                if i == sen_num_limit:
                    break

                for j, token in enumerate(sen_tokens):
                    if j == sen_len_limit:
                        break
                    context_s_idxs[count, i, j] = self._get_word(token)
                    if token in example["ques_tokens"]:
                        context_s_exist_tag[count, i, j, 0] = 1.0
                    if token in q_lemma1:
                        context_s_exist_tag[count, i, j, 1] = 1.0
                    if token in q_lemma2:
                        context_s_exist_tag[count, i, j, 2] = 1.0
                    pos_index = ques_type_dict[context_pos_tag[i][j][3]]
                    context_type_tag[count, i, j, pos_index] = 1.0

            for i, sen_tokens in enumerate(example["context_chars"]):
                if i == sen_num_limit:
                    break
                for j, token in enumerate(sen_tokens):
                    if j == sen_len_limit:
                        break
                    for m, char in enumerate(token):
                        if m == char_limit:
                            break
                        context_s_char_idxs[count, i, j, m] = self._get_char(char)

            for m, ques_type_token in enumerate(question_type_small):
                question = ' '.join(example["ques_tokens"])
                if ques_type_token in question:
                    for i in range(len(example["ques_tokens"])):
                        ques_type_tag[count][i] = question_types[ques_type_token]
                else:
                    for i in range(len(example["ques_tokens"])):
                        ques_type_tag[count][i] = question_types['']
            for i, token in enumerate(example["ques_tokens"]):
                if i == ques_limit:
                    break
                ques_idxs[count, i] = self._get_word(token)
                ques_s_exist_tag[count, i, i] = 1.0


            for i, token in enumerate(example["ques_chars"]):
                if i == ques_limit:
                    break
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[count, i, j] = self._get_char(char)

            answer_sen = example["ys"] #if example["ys"] < sen_num_limit else sen_num_limit-1
            y[count][answer_sen] = 1.0
            if not is_test:
                y1[count][example["y1s"]] = 1.0
                y2[count][example["y2s"]] = 1.0
            else:
                if example["y1s"][0] >=self.para_limit:
                    y1[count][0] = 1.0
                else:
                    y1[count][example["y1s"]] = 1.0
                if example["y2s"][0] >=self.para_limit:
                    y2[count][0] = 1.0
                else:
                    y2[count][example["y2s"]] = 1.0

            ids[count] = example["id"]

            count = count + 1

            tensor_dict['ids'] = ids
            tensor_dict['para_idxs'] = para_idxs
            tensor_dict['para_exist_tag'] = para_exist_tag
            tensor_dict['para_type_tag'] = para_type_tag

            tensor_dict['context_s_idxs'] = context_s_idxs
            tensor_dict['context_s_char_idxs'] = context_s_char_idxs
            tensor_dict['context_s_exist_tag'] = context_s_exist_tag
            tensor_dict['ques_s_exist_tag'] = ques_s_exist_tag
            tensor_dict['ques_idxs'] = ques_idxs
            tensor_dict['ques_char_idxs'] = ques_char_idxs
            tensor_dict['y'] = y
            tensor_dict['y1'] = y1
            tensor_dict['y2'] = y2
            tensor_dict['context_type_tag'] = context_type_tag
            tensor_dict['ques_type_tag'] = ques_type_tag
            tensor_dict['sentence_num'] = sentence_num

        return tensor_dict, idxs
