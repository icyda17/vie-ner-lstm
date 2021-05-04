import codecs
import numpy as np
import pickle5 as pickle
from underthesea import pos_tag
import re
from tensorflow import keras
from alphabet import Alphabet
from utils import read_conll_format

model = keras.models.load_model('model')
embedd_vectors = np.load(r'embedding/vectors.npy') #load pre-trained vector. shape (#words, dim)
with open(r'embedding/words.pl', 'rb') as handle: #list words. len(#words)
    embedd_words = pickle.load(handle)
alphabet_pos = Alphabet(name = 'pos', keep_growing=False)
alphabet_pos.load('model')
alphabet_tag = Alphabet(name = 'tag')
alphabet_tag.load('model')

def read_format(input:str):
    word_list = []
    pos_list = []
    pos_extract = pos_tag(input)
    for ele in pos_extract:
        word_list.append(map_number_and_punct(re.sub(' ','_',ele[0]).lower()))
        pos_list.append(ele[1])
    return [word_list], [pos_list]


def map_number_and_punct(word):
    if any(char.isdigit() for char in word):
        word = u'<number>'
    elif word in [u',', u'<', u'.', u'>', u'/', u'?', u'..', u'...', u'....', u':', u';', u'"', u"'", u'[', u'{', u']',
                  u'}', u'|', u'\\', u'`', u'~', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'+',
                  u'=']:
        word = u'<punct>'
    return word


def map_string_2_id_close(string_list, alphabet_string):
    string_id_list = []
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    return string_id_list


def map_string_2_id(pos_list_test):
    pos_id_list_test = map_string_2_id_close(pos_list_test, alphabet_pos)
    return pos_id_list_test

def construct_tensor_word(word_sentences, unknown_embedd, embedd_words, embedd_vectors, embedd_dim, max_length):
    X = np.empty([len(word_sentences), max_length, embedd_dim]) # shape: (#sentence, max_length of a sentence, dim)
    for i in range(len(word_sentences)):
        words = word_sentences[i] # a sentence
        length = len(words)
        for j in range(length):
            word = words[j].lower()
            try:
                embedd = embedd_vectors[embedd_words.index(word)]
            except:
                embedd = unknown_embedd
            X[i, j, :] = embedd
        # Zero out X after the end of the sequence
        X[i, length:] = np.zeros([1, embedd_dim])
    return X


def construct_tensor_onehot(feature_sentences, max_length, dim):
    X = np.zeros([len(feature_sentences), max_length, dim])
    for i in range(len(feature_sentences)):
        for j in range(len(feature_sentences[i])):
            if feature_sentences[i][j] > 0:
                X[i, j, feature_sentences[i][j]] = 1
    return X


def create_vector_data(word_list_test, pos_id_list_test, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                       max_length, dim_pos):
    word_test = construct_tensor_word(word_list_test, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                                      max_length)
    pos_test = construct_tensor_onehot(pos_id_list_test, max_length, dim_pos)
    input_test = word_test
    input_test = np.concatenate((input_test, pos_test), axis=2)
    return input_test


def create_data(test_input):
    embedd_dim = np.shape(embedd_vectors)[1]
    unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
    word_list_test, pos_list_test = read_format(test_input)
    pos_id_list_test = map_string_2_id(pos_list_test)
    max_length = 130 # modify according to train set
    input_test = \
        create_vector_data(word_list_test, pos_id_list_test, unknown_embedd, embedd_words,
                           embedd_vectors, embedd_dim, max_length, alphabet_pos.size())
    return input_test, word_list_test


def infer_string(test_input):
    input_test, word_list_test = create_data(test_input)
    predicts = model.predict_classes(input_test, batch_size=50)
    result = []
    tmp = {}
    for i in range(len(word_list_test)):
            for j in range(len(word_list_test[i])):
                predict = alphabet_tag.get_instance(predicts[i][j])
                if predict == None:
                    predict = alphabet_tag.get_instance(predicts[i][j] + 1)
                tmp[word_list_test[i][j]] = predict
                result.append(tmp)
                tmp = {}
    return result


# def infer_to_file(test_dir, output_file):
#     word_list, pos_list, tag_list, _, _ = read_conll_format(test_dir)
#     input_test, word_list_test = create_data(test_dir)
#     predicts = model.predict_classes(input_test, batch_size=50)
#     with codecs.open(output_file, 'w', 'utf-8') as f:
#         for i in range(len(word_list_test)):
#             for j in range(len(word_list_test[i])):
#                 predict = alphabet_tag.get_instance(predicts[i][j])
#                 if predict == None:
#                     predict = alphabet_tag.get_instance(predicts[i][j] + 1)
#                 f.write(word_list_test[i][j] + ' ' + predict + '\n')
#             f.write('\n')

if __name__ == "__main__":
    infer_string('Bên cạnh đó, Shark Đỗ Thị Kim Liên, Nhà sáng lập Ứng dụng bảo hiểm công nghệ LIAN cam kết tiếp tục hành trình “bà đỡ” cho start-up Việt.')
