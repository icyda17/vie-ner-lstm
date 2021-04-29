import codecs
import numpy as np
import pickle5 as pickle
import json
from tensorflow import keras
from alphabet import Alphabet

model = keras.models.load_model('model')

def read_conll_format(input_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        word_list = []
        chunk_list = []
        pos_list = []
        words = []
        chunks = []
        poss = []
        for line in f:
            line = line.split()
            if len(line) > 0:
                words.append(map_number_and_punct(line[0].lower()))
                poss.append(line[1])
                chunks.append(line[2])
            else:
                word_list.append(words)
                pos_list.append(poss)
                chunk_list.append(chunks)
                words = []
                chunks = []
                poss = []
    return word_list, pos_list, chunk_list


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


def map_string_2_id(pos_list_test, chunk_list_test):
    alphabet_pos = Alphabet(name = 'pos', keep_growing=False)
    alphabet_pos.load('model')
    alphabet_chunk = Alphabet(name = 'chunk', keep_growing=False)
    alphabet_chunk.load('model')
    pos_id_list_test = map_string_2_id_close(pos_list_test, alphabet_pos)
    chunk_id_list_test = map_string_2_id_close(chunk_list_test, alphabet_chunk)
    return pos_id_list_test, chunk_id_list_test, alphabet_pos, alphabet_chunk


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


def create_vector_data(word_list_test, pos_id_list_test,chunk_id_list_test,unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                       max_length, dim_pos, dim_chunk):
    word_test = construct_tensor_word(word_list_test, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                                      max_length)
    pos_test = construct_tensor_onehot(pos_id_list_test, max_length, dim_pos)
    chunk_test = construct_tensor_onehot(chunk_id_list_test, max_length, dim_chunk)
    input_test = word_test
    input_test = np.concatenate((input_test, pos_test), axis=2)
    input_test = np.concatenate((input_test, chunk_test), axis=2)
    return input_test


def create_data(word_dir, vector_dir, test_dir):
    embedd_vectors = np.load(vector_dir) #load pre-trained vector. shape (#words, dim)
    with open(word_dir, 'rb') as handle: #list words. len(#words)
        embedd_words = pickle.load(handle)
    embedd_dim = np.shape(embedd_vectors)[1]
    unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
    word_list_test, pos_list_test, chunk_list_test = \
        read_conll_format(test_dir)
    pos_id_list_test, chunk_id_list_test, alphabet_pos, alphabet_chunk = \
        map_string_2_id(pos_list_test, chunk_list_test)
    max_length = 43 # modify according to train set
    input_test = \
        create_vector_data(word_list_test, pos_id_list_test,chunk_id_list_test,unknown_embedd, embedd_words,
                           embedd_vectors, embedd_dim, max_length, alphabet_pos.size(), alphabet_chunk.size())
    return input_test, max_length, word_list_test


def infer_to_file(word_dir, vector_dir, test_dir, output_file):
    input_test, _ , word_list_test = create_data(word_dir, vector_dir, test_dir)
    predicts = model.predict_classes(input_test, batch_size=50)
    alphabet_tag = Alphabet(name = 'tag')
    alphabet_tag.load('model')
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for i in range(len(word_list_test)):
            for j in range(len(word_list_test[i])):
                predict = alphabet_tag.get_instance(predicts[i][j])
                if predict == None:
                    predict = alphabet_tag.get_instance(predicts[i][j] + 1)
                f.write(word_list_test[i][j] + ' ' + predict + '\n')
            f.write('\n')


if __name__ == "__main__":
    infer_to_file('embedding/words.pl', 'embedding/vectors.npy', 'test.txt', 'out3.txt')
