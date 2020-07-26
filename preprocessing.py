from utils import *
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
import os


class TextPreprocessing:
    """
    Clean input sentence to predict proper response
    """
    def __init__(self):

        settings_dir = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
        self.DATA_FOLDER = os.path.join(PROJECT_ROOT, 'learning/data')
        self.stop_words = None
        self.tokenizer = None

    def load_data(self):
        """
        load stop words and saved tokenizer from training
        """
        with open(os.path.join(self.DATA_FOLDER, 'french_stopwords.txt'), 'rb') as fp:
            self.stop_words = pickle.load(fp)

        with open(os.path.join(self.DATA_FOLDER, 'tokenizer.pickle'), 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def clean_up_sentence(self, sentence):
        """
        clean sentences from stop words and correct word according to word dictionary used in training
        (saliar --> salaire, credi --> crédit)
        """

        num = []
        pattern = sentence.lower()
        tokens = word_tokenize(pattern)
        tokens, num = self.tag_number(tokens)
        filtered_words = [words_correction(w) for w in tokens if w not in self.stop_words]
        sentence = TreebankWordDetokenizer().detokenize(filtered_words)
        return sentence, num

    def transform_test_numeric(self, sentence, maxlen=10):
        """
        transform sentence to sequence of integers according to trained tokenizer
        """
        sequence = self.tokenizer.texts_to_sequences([sentence])
        sequences = pad_sequences(sequence, maxlen=maxlen)
        return sequences

    def text_preprocessing(self, sentence):
        """
        sentence cleaning
        """
        clean_sentence, numbers = self.clean_up_sentence(sentence)
        sequence = self.transform_test_numeric(clean_sentence)

    @staticmethod
    def tag_number(tokens):
        """
        transform numbers to numbercode so that the model can recognize it as a number
        """
        num = []
        for token in tokens:
            if is_number(token):
                num.append(token)
                tokens[tokens.index(token)] = "numbercode"
        return tokens, num
