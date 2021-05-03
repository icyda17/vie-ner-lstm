import sys
sys.path.append(r"D:\OneDrive - Hanoi University of Science and Technology\Intern\Training\ner_lstm\vie-ner-lstm")

from utils import *

train_dir = r'data/data_pos/train_pos2.txt'
dev_dir = r'data\data_pos\dev_pos.txt'
test_dir = r'data\data_pos\test_pos.txt'

# alphabet_tag during training
word_list_train, pos_list_train, tag_list_train, num_sent_train, max_length_train = \
    read_conll_format(train_dir)
word_list_dev, pos_list_dev, tag_list_dev, num_sent_dev, max_length_dev = \
read_conll_format(dev_dir)
word_list_test, pos_list_test, tag_list_test, num_sent_test, max_length_test = \
read_conll_format(test_dir)
pos_id_list_train, pos_id_list_dev, pos_id_list_test,\
tag_id_list_train, tag_id_list_dev, tag_id_list_test, alphabet_pos, alphabet_tag = \
    map_string_2_id(pos_list_train, pos_list_dev, pos_list_test, tag_list_train, tag_list_dev, \
        tag_list_test)
# alphabet_tag during infer
alphabet_pos2 = Alphabet(name = 'pos', keep_growing=False)
alphabet_pos2.load('model')
alphabet_tag2 = Alphabet(name = 'tag')
alphabet_tag2.load('model')

def test_alphabet(alphabet_tag1, alphabet_tag2, tag_list):
        """Check if index while training = during infer
        """
        out1 = [alphabet_tag1.get_index(i) for i in tag_list]
        out2 = [alphabet_tag2.get_index(i) for i in tag_list]
        assert out1 == out2, "Mismatch"

if __name__ == "__main__":
    test_alphabet(alphabet_tag, alphabet_tag2, tag_list_train[0])