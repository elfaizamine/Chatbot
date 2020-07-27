from learning.moduleSingleton import Singleton  # import trained models as singleton but script not included
from preprocessing import TextPreprocessing

import json
import os
import pickle


class Response:
    """
    Find proper response to sentence by model classification
    """

    def __init__(self):

        settings_dir = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
        self.DATA_FOLDER = os.path.join(PROJECT_ROOT, 'learning/data')
        self.embedding_matrix = None
        self.data = None
        self.words = None
        self.classes = None

    def load_data(self):
        """
        load data, training words dictionary, output categories, embedding matrix of training words
        """
        with open(os.path.join(self.DATA_FOLDER, 'input_data.json'), encoding='utf-8') as json_data:
            self.data = json.load(json_data)

        with open(os.path.join(self.DATA_FOLDER, 'WORDS.pickle'), 'rb') as f:
            self.words = pickle.load(f)

        with open(os.path.join(self.DATA_FOLDER, 'classes.pickle'), 'rb') as handle:
            self.classes = pickle.load(handle)

        with open(os.path.join(self.DATA_FOLDER, 'embedding_matrix.pickle'), 'rb') as handle:
            self.embedding_matrix = pickle.load(handle)

    def classify(self, sentence, maxlen=10, embedding_dim=300, error_thresh=0.005):
        """
        classify sentence into proper response by grouping multiple predictions
        if sentence contains number then the input must be user information and output is given accordingly

        :return  list of tuples of tag predictions sorted by probability : [('SumLoan', 0.125')]
        probability of proper response with tag SumLoan is 12.5%
        """

        cleaner = TextPreprocessing()
        cleaner.load_data()  # load needed data for input cleaning

        result = []  # list of predictions

        # loading trained models
        allModel = Singleton(cleaner.tokenizer.word_index, self.embedding_matrix, maxlen, embedding_dim)
        model0 = allModel[0]
        model1 = allModel[1]
        model2 = allModel[2]
        model3 = allModel[3]
        model4 = allModel[4]
        model5 = allModel[5]

        sequence, num = cleaner.text_preprocessing(sentence)  # clean sentence input

        # predict classes probability
        results0 = model0.predict(sequence)[0]
        results1 = model1.predict(sequence)[0]
        results2 = model2.predict(sequence)[0]
        results3 = model3.predict(sequence)[0]
        results4 = model4.predict(sequence)[0]
        results5 = model5.predict(sequence)[0]
        results = (results0 + results1 + results2 + results3 + results4 + results5) / 6

        # remove small probability predictions
        results = [[i, r] for i, r in enumerate(results) if r > error_thresh]

        # sort prediction classes by probability
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            result.append((self.classes[r[0]], r[1]))

        # if sentence contains number, it must be on of these classes
        if len(num) != 0:
            result = [('SumLoan', 0.125), ('LengthCE', 0.125), ('Rate', 0.125),
                      ('Length', 0.125), ('RateCE', 0.125), ('Diff', 0.125),
                      ('Amount', 0.125), ('Salary', 0.125)]

        return result



