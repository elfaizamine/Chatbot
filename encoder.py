from nltk.corpus import stopwords
import os
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from collections import Counter
from sklearn.utils import class_weight
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import WordPunctTokenizer
from gensim.models import KeyedVectors
from keras.layers.core import Reshape
from keras.layers import Embedding, Dense, GRU, Input, LSTM, Dropout,Flatten,MaxPooling1D,GlobalMaxPooling1D,Bidirectional, Activation, PReLU ,Bidirectional,Conv1D, GlobalMaxPool1D,TimeDistributed
from keras.models import Model, Sequential
import json
import string
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from learning.attention import Attention
import nltk

settings_dir = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'learning/data')

def encoder():

    with open(os.path.join(DATA_FOLDER, 'input_data.json'), encoding='utf-8') as json_data:
        intents = json.load(json_data)

    # set stop words and caracters
    nltk.download('stopwords')
    french_stops = (stopwords.words('french'))
    french_stops.remove('pas')
    french_stops.extend(list(string.punctuation))
    french_stops.extend(['??','a','si','être','avoir','quel','quelle','quels','quoi'])
    words = []
    classes = []
    documents = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        if 'patterns' in intent:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                pattern = pattern.lower()
                tokens = WordPunctTokenizer().tokenize(pattern)
                filtered_words = [w for w in tokens if not w in french_stops]
                words.extend(filtered_words)
                sentence = TreebankWordDetokenizer().detokenize(filtered_words)
                # add to documents in our corpus
                documents.append((sentence, intent['tag']))
                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

    # seperate sentences and labels
    text = []
    output = []
    for doc in documents:
        text.append(doc[0])
        num = classes.index(doc[1])
        output.append(num)

    # parameters of sequence length
    maxlen = 10
    # tokenisation and transforming to sequences of integers
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=maxlen)
    Y = np.asarray(output)

    # Pretrained embedding matrix matrix for french words
    pathembMatrix = r'C:\Users\u958336\Desktop\wiki.fr.vec'
    embMatrix = KeyedVectors.load_word2vec_format(pathembMatrix)

    embedding_dim = 300
    embedding_matrix = np.random.uniform(-0.3,0.3,(len(word_index)+1, embedding_dim))

    for word, i in word_index.items():
        if word == 'nnuumm':
            embedding_vector = embMatrix['numéro']
        if word in embMatrix:
            embedding_vector = embMatrix[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    class_weights1 = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
    
    # diminish the weight for the classes where there is a context
    class_weights = {}
    for i in range(34):
        if i in (25,26,27,28,29,30,31,32):
            class_weights[i] = 0.4
        else:
            class_weights[i] = class_weights1[i]
       
    
    ## Models
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True)(_input)
    model = Conv1D(filters = 128, kernel_size = 2,strides = 1,activation='relu',padding = 'same')(embedded)
    model = Dropout(0.4)(model)
    model = MaxPooling1D(2)(model)
    model = Conv1D(filters = 64, kernel_size = 2,strides = 1, activation='relu',padding ='same')(model)
    model = GlobalMaxPooling1D()(model)
    model = Dropout(0.4)(model)
    model = Dense(34, activation='softmax')(model)
    model = Model(input=_input, output=model)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model.fit(X,Y,epochs=30,batch_size=16, class_weight = class_weights)
 
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True )(_input)
    model1 = Bidirectional(GRU(128,return_sequences = True),merge_mode = 'concat')(embedded) 
    model1 = PReLU()(model1)
    model1 = Attention(10)(model1)
    model1 = Dropout(0.7)(model1)
    model1 = Dense(34, activation='softmax')(model1)
    model1 = Model(input=_input, output=model1)
    model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model1.fit(X,Y,epochs=25,batch_size=16, class_weight = class_weights)
    
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True )(_input)
    model2 = Bidirectional(GRU(128,return_sequences = True,activity_regularizer=regularizers.l1(0.0001)),merge_mode = 'concat')(embedded) 
    model2 = PReLU()(model2)
    model2 = Attention(10)(model2)
    model2 = Dense(34, activation='softmax')(model2)
    model2 = Model(input=_input, output=model2)
    model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model2.fit(X,Y,epochs=50,batch_size=16, class_weight = class_weights)
    
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True )(_input)
    model3 = Conv1D(filters = 128, kernel_size = 2,strides = 1,activation='relu',padding = 'same')(embedded)
    model3 = Attention(10)(model3)
    model3 = Dropout(0.5)(model3)
    model3 = Dense(34, activation='softmax')(model3)
    model3 = Model(input=_input, output=model3)
    model3.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model3.fit(X,Y,epochs=25,batch_size=16, class_weight = class_weights)
    
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True )(_input)
    model4 = Conv1D(filters = 128, kernel_size = 2,strides = 1,activation='relu',padding = 'same',activity_regularizer=regularizers.l1(0.0001))(embedded)
    model4 = Attention(10)(model4)
    model4 = Dense(34, activation='softmax')(model4)
    model4 = Model(input=_input, output=model4)
    model4.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model4.fit(X,Y,epochs=35,batch_size=16, class_weight = class_weights)
    
    _input = Input(shape=[maxlen], dtype='int32')
    embedded = Embedding(len(word_index)+1, embedding_dim,input_length=maxlen,weights = [embedding_matrix],trainable = True)(_input)
    model5 = Reshape((10,300,1), input_shape=(10,300,))(embedded)
    model5 =  TimeDistributed(Conv1D(filters = 128, kernel_size = 5,strides =3,activation='relu',padding = 'valid',activity_regularizer=regularizers.l1(0.0001)))(model5) 
    model5 = Reshape((10,99*128), input_shape=(10,99,128,))(model5)
    model5 = Attention(10)(model5)
    model5 = Dense(34, activation='softmax')(model5)
    model5 = Model(input=_input, output=model5)
    model5.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    model5.fit(X,Y,epochs=45,batch_size=16, class_weight = class_weights)

    # Create dictionnary for saving correct words and frequences

    repository = []
    for intent in intents['intents']:
        if 'patterns' in intent:
            for pattern in intent['patterns']:
                pattern = pattern.lower()
                tokens = WordPunctTokenizer().tokenize(pattern)
                filtered_words = [w for w in tokens if not w in french_stops]
                repository.extend(filtered_words)
                
    for i in repository:
        if not i.isalpha():
            repository.remove(i)
    WORDS = Counter(repository)

    model.save_weights(os.path.join(DATA_FOLDER, 'weights_model.h5'))
    model1.save_weights(os.path.join(DATA_FOLDER, 'weights_model1.h5'))
    model2.save_weights(os.path.join(DATA_FOLDER, 'weights_model2.h5'))
    model3.save_weights(os.path.join(DATA_FOLDER, 'weights_model3.h5'))
    model4.save_weights(os.path.join(DATA_FOLDER, 'weights_model4.h5'))
    model5.save_weights(os.path.join(DATA_FOLDER, 'weights_model5.h5'))


    with open(os.path.join(DATA_FOLDER, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(DATA_FOLDER, 'classes.pickle'), 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(DATA_FOLDER, 'WORDS.pickle'), 'wb') as f:
        pickle.dump(WORDS, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(DATA_FOLDER, 'french_stopwords.txt'), 'wb') as fp:
        pickle.dump(french_stops, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(DATA_FOLDER, 'embedding_matrix.pickle'), 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(DATA_FOLDER, 'class_weights.pickle'), 'wb') as handle:
        pickle.dump(class_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)


