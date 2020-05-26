import json
import os
import pickle
import random
import re

from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from learning.moduleSingleton import Singleton

settings_dir = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.dirname(settings_dir))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'learning/data')

# Load data
with open(os.path.join(DATA_FOLDER, 'input_data.json'), encoding='utf-8') as json_data:
    intents = json.load(json_data)

with open(os.path.join(DATA_FOLDER, 'french_stopwords.txt'), 'rb') as fp:
    french_stops = pickle.load(fp)

with open(os.path.join(DATA_FOLDER, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(os.path.join(DATA_FOLDER, 'WORDS.pickle'), 'rb') as f:
    WORDS = pickle.load(f)

with open(os.path.join(DATA_FOLDER, 'classes.pickle'), 'rb') as handle:
    classes = pickle.load(handle)

with open(os.path.join(DATA_FOLDER, 'embedding_matrix.pickle'), 'rb') as handle:
    embedding_matrix = pickle.load(handle)

with open(os.path.join(DATA_FOLDER, 'class_weights.pickle'), 'rb') as handle:
    class_weights = pickle.load(handle)

maxlen = 10
word_index = tokenizer.word_index
embedding_dim = 300


x = 0
for i in WORDS.values():
    x = x + int(i)



def P(word, N=548):
    return int(WORDS[word]) / N


def correction(word):
    return max(candidates(word), key=P)


def candidates(word):
    if word.isalpha():
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    else:
        return [word]


def known(words):
    return set(w for w in words if w in WORDS)

# Correction of words
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyzéàèçêùâ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def decontracted(phrase):
    pattern = [r"j'",r"c'",r"y'",r"qu'",r"n'",r"t'",r"s'",r"m'",r"l'",r"d'"]
    replace = ["j ","c ","y ","qui ","n ","t ","s ","m ","l ","d "]
    
    for i in range(len(pattern)) : 
        phrase = re.sub(pattern[i], replace[i], phrase)

    return phrase

def is_number(s):
    try:
        float(s)
        return True
    except ValueError :
        return False


def tag_number(tokens):
    num = []
    for token in tokens:
        if is_number(token):
            num.append(token)
            tokens[tokens.index(token)] = "nnuumm"
    return tokens, num


def clean_up_sentence(sentence):
    num = []
    pattern = sentence.lower()
    pattern = decontracted(pattern)
    tokens = word_tokenize(pattern)
    num = tag_number(tokens)[1]
    tokens = tag_number(tokens)[0]
    filtered_words = [correction(w) for w in tokens if not w in french_stops]
    sentence = TreebankWordDetokenizer().detokenize(filtered_words)
    return sentence, num


def preprocessing(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    sequences = pad_sequences(sequence, maxlen=maxlen)
    return sequences


context = {}
simulateurCredit = {}
simulateurCreditCE = {}
messageBlocked = {}

ERROR_THRESHOLD = 0.005

# Ensemble learning to group different predictions and classify
def classify(sentence):
    allModel = Singleton(word_index, maxlen, embedding_dim, embedding_matrix)
    model0 = allModel[0]
    model1 = allModel[1]
    model2 = allModel[2]
    model3 = allModel[3]
    model4 = allModel[4]
    model5 = allModel[5]
    clean_sentence = clean_up_sentence(sentence)[0]
    num = clean_up_sentence(sentence)[1]
    sequence = preprocessing(clean_sentence)
    results0 = model0.predict(sequence)[0]
    results1 = model1.predict(sequence)[0]
    results2 = model2.predict(sequence)[0]
    results3 = model3.predict(sequence)[0]
    results4 = model4.predict(sequence)[0]
    results5 = model5.predict(sequence)[0]
    results = (results0 + results1 + results2 + results3 + results4 + results5) / 6
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if len(num) != 0:
        results_num = [('giveMeSumCredit', 0.125), ('giveMeDureeCE', 0.125), ('giveMeTaux', 0.125),
                       ('giveMeDuree', 0.125), ('giveMeTauxCE', 0.125), ('giveMeDiff', 0.125),
                       ('giveMeAmount', 0.125), ('giveMeSalary', 0.125)]
        return results_num
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


