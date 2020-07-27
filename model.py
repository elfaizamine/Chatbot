from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import WordPunctTokenizer
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Reshape
from keras.layers import *
from collections import Counter
import pickle
import numpy as np
import os
import json
import string


class Models:
    """
    Preprocess text inputs, train model, save model, save data
    """
    
    def __init__(self):
        settings_dir = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
        self.DATA_FOLDER = os.path.join(PROJECT_ROOT, 'learning/data')
        self.embedding_matrix = None
        self.stop_words = None
        self.tokenizer = None
        self.classes = None
        self.output = None
        self.words = None
        self.input = None
        self.model = None
        self.data = None
        

    def load_data(self):
        """
        load data, pretrained words embeddings, and stop words
        """
        with open(os.path.join(self.DATA_FOLDER, 'input_data.json'), encoding='utf-8') as json_data:
            self.data = json.load(json_data)

        # pretrained embedding matrix matrix for french words
        path_embedding_matrix = os.path.join(self.DATA_FOLDER, 'wiki.fr.vec')
        self.embedding_matrix = KeyedVectors.load_word2vec_format(path_embedding_matrix)

        french_stops = (stopwords.words('french'))  # nltk.download('stopwords')
        french_stops.extend(list(string.punctuation))
        french_stops.extend(['??', 'a', 'si', 'être', 'avoir', 'quel', 'quelle', 'quels', 'quoi'])
        self.stop_words = french_stops
        

    def preprocess_data(self):
        """
        clean sentences and normalize them, transform sentences to sequence of integers
        """
        # eliminate stop words and punctuation from data
        words, classes, documents = [], [], []
        # loop through each sentence in our intents patterns
        for intent in self.data['intents']:
            if 'patterns' in intent:
                for pattern in intent['patterns']:
                    # tokenize each word in the sentence
                    pattern = pattern.lower()
                    tokens = WordPunctTokenizer().tokenize(pattern)
                    filtered_words = [w for w in tokens if w not in self.stop_words]
                    words.extend(filtered_words)
                    sentence = TreebankWordDetokenizer().detokenize(filtered_words)
                    # add to documents in our corpus
                    documents.append((sentence, intent['tag']))
                    # add to our classes list
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

        # save dictionary of words
        repository = words
        for i in words:
            if not i.isalpha():
                repository.remove(i)

        # separate sentences (input) and labels (output)
        text, output = [], []
        for doc in documents:
            text.append(doc[0])
            output.append(classes.index(doc[1]))

        self.input = text  # save input
        self.output = np.asarray(output)  # save output
        self.classes = classes  # save target classes
        self.words = Counter(repository)  # save words dictionary

        self.transform_text_to_numeric()  # transform sentences to sequence of integers
        self.transform_numeric_with_embeddings()  # create embedding matrix with words in dictionary
        

    def transform_text_to_numeric(self):
        """
        transform sentences to sequence of integers with maximum of maxlen=10 or padding them
        """
        # transforming sequences of words to sequences of integers
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.input)
        sequences = tokenizer.texts_to_sequences(self.input)

        self.input = pad_sequences(sequences, maxlen=10)  # parameters of sequence length maxlen = 10
        self.tokenizer = tokenizer
        

    def transform_numeric_with_embeddings(self):
        """
        create matrix of embeddings that link word index number with embedding vector
        """

        word_index = self.tokenizer.word_index
        embedding_matrix_input = np.random.uniform(-0.3, 0.3, (len(word_index) + 1, 300))  # embedding dimensions is 300
        for word, index in word_index.items():
            if word == 'nnuumm':
                embedding_vector = self.embedding_matrix['numéro']
            if word in self.embedding_matrix:
                embedding_vector = self.embedding_matrix[word]
            if embedding_vector is not None:
                embedding_matrix_input[index] = embedding_vector

        self.embedding_matrix = embedding_matrix_input
        

    def train(self, maxlen=10, embedding_dim=300):
        """
        train multiple models architecture
        """
        # Models Architecture and training
        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model = Conv1D(filters=128, kernel_size=2, strides=1, activation='relu', padding='same')(embedded)
        model = Dropout(0.4)(model)
        model = MaxPooling1D(2)(model)
        model = Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same')(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(0.4)(model)
        model = Dense(34, activation='softmax')(model)
        model = Model(input=_input, output=model)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        model.fit(self.input, self.output, epochs=30, batch_size=16)

        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model1 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(embedded)
        model1 = PReLU()(model1)
        model1 = Dropout(0.7)(model1)
        model1 = Dense(34, activation='softmax')(model1)
        model1 = Model(input=_input, output=model1)
        model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
        model1.fit(self.input, self.output, epochs=25, batch_size=16)

        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model2 = Bidirectional(GRU(128, return_sequences=True, activity_regularizer=regularizers.l1(0.0001)),
                               merge_mode='concat')(embedded)
        model2 = PReLU()(model2)
        model2 = Dense(34, activation='softmax')(model2)
        model2 = Model(input=_input, output=model2)
        model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
        model2.fit(self.input, self.output, epochs=50, batch_size=16)

        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model3 = Conv1D(filters=128, kernel_size=2, strides=1, activation='relu', padding='same')(embedded)
        model3 = Dropout(0.5)(model3)
        model3 = Dense(34, activation='softmax')(model3)
        model3 = Model(input=_input, output=model3)
        model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
        model3.fit(self.input, self.output, epochs=25, batch_size=16)

        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model4 = Conv1D(filters=128, kernel_size=2, strides=1, activation='relu', padding='same',
                        activity_regularizer=regularizers.l1(0.0001))(embedded)
        model4 = Dense(34, activation='softmax')(model4)
        model4 = Model(input=_input, output=model4)
        model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
        model4.fit(self.input, self.output, epochs=35, batch_size=16)

        _input = Input(shape=[maxlen], dtype='int32')
        embedded = Embedding(len(self.tokenizer.word_index) + 1, embedding_dim, input_length=maxlen,
                             weights=[self.embedding_matrix], trainable=True)(_input)
        model5 = Reshape((10, 300, 1), input_shape=(10, 300,))(embedded)
        model5 = TimeDistributed(Conv1D(filters=128, kernel_size=5, strides=3, activation='relu', padding='valid',
                                        activity_regularizer=regularizers.l1(0.0001)))(model5)
        model5 = Reshape((10, 99 * 128), input_shape=(10, 99, 128,))(model5)
        model5 = Dense(34, activation='softmax')(model5)
        model5 = Model(input=_input, output=model5)
        model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
        model5.fit(self.input, self.output, epochs=45, batch_size=16)
        self.model = model, model1, model2, model3, model4, model5
        

    def save_all(self):
        """
        save models and data
        """
        self.model[0].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model.h5'))
        self.model[1].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model1.h5'))
        self.model[2].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model2.h5'))
        self.model[3].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model3.h5'))
        self.model[4].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model4.h5'))
        self.model[5].save_weights(os.path.join(self.DATA_FOLDER, 'weights_model5.h5'))

        with open(os.path.join(self.DATA_FOLDER, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.DATA_FOLDER, 'classes.pickle'), 'wb') as handle:
            pickle.dump(self.classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.DATA_FOLDER, 'WORDS.pickle'), 'wb') as f:
            pickle.dump(self.words_dictionnary, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.DATA_FOLDER, 'french_stopwords.txt'), 'wb') as fp:
            pickle.dump(self.stop_words, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.DATA_FOLDER, 'embedding_matrix.pickle'), 'wb') as handle:
            pickle.dump(self.embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

