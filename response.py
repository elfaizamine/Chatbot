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

# load architecture


# Correction des mots
x = 0
for i in WORDS.values():
    x = x + int(i)


# Correction of words


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


def getKeyByValue(mydictionary, value):
    for key in mydictionary:
        if mydictionary[key] == value:
            return key


def getFromDic(test_list, value):
    for x in test_list:
        if x[1] == value:
            return x[0]
            break


def getFromDict(test_list, value, val):
    for x in test_list:
        if x[1] == value or x[1] == val:
            return x[0]
            break


def response(sentence, userID='123'):
    num = clean_up_sentence(sentence)[1]
    clean_sentence = clean_up_sentence(sentence)[0]
    results = classify(sentence)
    response = {}
    response['response'] = "stp soit plus précis"
    # if we have a classification then find the matching intent tag
    while results:
        for i in intents['intents']:
            # find a tag matching the first result
            if i['tag'] == results[0][0]:
                # Check if i has context_set
                if 'context_filter' in i and userID not in context and 'context_filter_mondatory' not in i:
                    break
                if 'context_filter' in i and userID in context and not verify_filter(i,
                                                                                     userID) and 'context_filter_mondatory' not in i:
                    break
                if ('applayfopYes' == i['tag']) & (clean_sentence in ['oui', 'yes']):
                    if check_approval_context(userID):
                        approvalcontext = get_approval_context(userID)
                        update_context_Oui(userID, get_approval_context(userID))
                        if 'positiveIntent' in get_intent_by_context_set(approvalcontext):
                            provision_intent = get_intent_by_context_filter(get_oui_context(userID))
                            i = provision_intent
                        if 'positiveAnswer' in get_intent_by_context_set(approvalcontext):
                            response['response'] = get_answer_for_choice('Oui', approvalcontext)
                            return response
                if ('applayfopNo' == i['tag']) & (clean_sentence in ['non', 'no']):
                    if check_approval_context(userID):
                        approvalcontext = get_approval_context(userID)
                        update_context_Non(userID, get_approval_context(userID))
                        if 'negativeIntent' in get_intent_by_context_set(approvalcontext):
                            provision_intent = get_intent_by_context_filter(get_non_context(userID))
                            i = provision_intent
                        if 'negativeAnswer' in get_intent_by_context_set(approvalcontext):
                            response['response'] = get_answer_for_choice('Non', approvalcontext)
                            return response
                if check_approval_context(userID):
                    i = get_intent_by_context_set(get_approval_context(userID))
                if "giveMeAmount" == i['tag']:
                    if verify_filter(i, userID):
                        if int(num[0]) > 25000000:
                            response['response'] = [
                                "Le montant souhaité ne peut pas dépasser 25 MDH.Veuillez saisir un montant inférieur à 25 MDH"]
                            return response
                if i['tag'] in ["giveMeDuree", "giveMeDureeCE"]:
                    if verify_filter(i, userID):
                        if int(num[0]) > 300:
                            response['response'] = ["Veuillez entrer une durée inférieur à 300 mois"]
                            return response
                if 'context_filter' in i:
                    if verify_filter(i, userID):
                        context[userID].pop(getKeyByValue(context[userID], right_filter(i, userID)))
                if 'choice' in i:
                    response['choice'] = i['choice']
                if 'context_set' in i:
                    set_context(userID, i)
                    set_simulator_data(userID, i, num)
                    set_simulator_data_CE(userID, i, num)
                    simulator_data = get_simulator_data(userID)
                    simulator_data_CE = get_simulator_data_CE(userID)
                    if simulator_data:
                        clean_simulator_data(userID)
                        response['bodysimulation'] = simulator_data
                        response['response'] = random.choice(i['responses'])
                        return response
                    if simulator_data_CE:
                        clean_simulator_data_CE(userID)
                        response['bodysimulation'] = simulator_data_CE
                        response['response'] = random.choice(i['responses'])
                        return response
                response['response'] = random.choice(i['responses'])
                return response
                break

        results.pop(0)

    return response


def set_context(user_id, intent):
    # Check if user_id and intent are not null or empty
    if user_id in context:
        if not intent['context_set'] in context[user_id].values():
            size = len(context[user_id]) + 1
            key_context = "c" + str(size)
            context[user_id].update({key_context: intent['context_set']})
    else:
        context[user_id] = {'c1': intent['context_set']}


def get_intent_by_context_set(context_set):
    for i in intents['intents']:
        if 'context_set' in i:
            if i['context_set'] == context_set:
                return i


def get_intent_by_context_filter(context_filter):
    for i in intents['intents']:
        if 'context_filter' in i and not isinstance(i['context_filter'], list):
            if i['context_filter'] == context_filter:
                return i
        if 'context_filter' in i and isinstance(i['context_filter'], list):
            if context_filter in i['context_filter']:
                return i


def get_answer_for_choice(choice, contextset):
    for i in intents['intents']:
        if 'context_set' in i:
            if contextset == i['context_set']:
                if choice == 'Oui':
                    return i['positiveAnswer']
                if choice == 'Non':
                    return i['negativeAnswer']


def update_context_Oui(user_id, value):
    context[user_id].update({getKeyByValue(context[user_id], value): str(value.replace("_NeedApproval", "_Oui"))})


def update_context_Non(user_id, value):
    context[user_id].update({getKeyByValue(context[user_id], value): str(value.replace("_NeedApproval", "_Non"))})


def verify_filter(intent, user_id):
    if isinstance(intent['context_filter'], list):
        for i in intent['context_filter']:
            if user_id in context and i in context[user_id].values():
                return True
        return False
    if not isinstance(intent['context_filter'], list):
        if user_id in context and intent['context_filter'] in context[user_id].values():
            return True
    return False


def right_filter(intent, user_id):
    if isinstance(intent['context_filter'], list):
        for i in intent['context_filter']:
            if i in context[user_id].values():
                return i
    if not isinstance(intent['context_filter'], list):
        if user_id in context and intent['context_filter'] in context[user_id].values():
            return intent['context_filter']


def get_approval_context(user_id):
    if user_id in context:
        for value in context[user_id].values():
            if value.endswith("_NeedApproval"):
                return value
    return


def check_approval_context(user_id):
    if user_id in context:
        for value in context[user_id].values():
            if value.endswith("_NeedApproval"):
                return True
    return False


def get_oui_context(user_id):
    if user_id in context:
        for value in context[user_id].values():
            if value.endswith("_Oui"):
                return value
    return


def get_non_context(user_id):
    if user_id in context:
        for value in context[user_id].values():
            if value.endswith("_Non"):
                return value
    return


def set_block(user_id, intent):
    messageBlocked[user_id] = {intent['context_set']: intent['message_block']}


def set_simulator_data(user_id, intent, num):
    montan_key = user_id + "SCM"
    duree_key = user_id + "SCD"
    taux_key = user_id + "SCT"
    diff_key = user_id + "SCDF"

    if intent['context_set'] == "montantGiven" and not montan_key in simulateurCredit:
        simulateurCredit.update({montan_key: num[0]})

    if intent['context_set'] == "dureeGiven_NeedApproval" and not duree_key in simulateurCredit:
        simulateurCredit.update({duree_key: num[0]})

    if intent['context_set'] == "Simmulation_given_NeedApproval" and not taux_key in simulateurCredit:
        simulateurCredit.update({taux_key: num[0]})

    if intent['context_set'] == "diffGiven" and not diff_key in simulateurCredit:
        simulateurCredit.update({diff_key: num[0]})

    if intent['context_set'] == "Simmulation_given_NeedApproval" and not diff_key in simulateurCredit:
        simulateurCredit.update({diff_key: '0'})


def get_simulator_data(user_id):
    montan_key = user_id + "SCM"
    duree_key = user_id + "SCD"
    taux_key = user_id + "SCT"
    diff_key = user_id + "SCDF"
    simulation_data = False
    if all(k in simulateurCredit for k in (montan_key, duree_key, taux_key, diff_key)):
        simulationJSON = {}
        simulationJSON['montant'] = str(simulateurCredit[montan_key])
        simulationJSON['duree'] = str(simulateurCredit[duree_key])
        simulationJSON['taux'] = str(simulateurCredit[taux_key])
        simulationJSON['differe'] = str(simulateurCredit[diff_key])
        return simulationJSON
    return simulation_data


def clean_simulator_data(user_id):
    montan_key = user_id + "SCM"
    duree_key = user_id + "SCD"
    taux_key = user_id + "SCT"
    diff_key = user_id + "SCDF"

    if all(k in simulateurCredit for k in (montan_key, duree_key, taux_key, diff_key)):
        del simulateurCredit[montan_key]
        del simulateurCredit[duree_key]
        del simulateurCredit[taux_key]
        del simulateurCredit[diff_key]


def set_simulator_data_CE(user_id, intent, num):
    salary_key = user_id + "CES"
    duree_key = user_id + "CED"
    taux_key = user_id + "CET"
    encour_key = user_id + "CEC"

    if intent['context_set'] == "salaryGiven_NeedApproval" and not salary_key in simulateurCreditCE:
        simulateurCreditCE.update({salary_key: num[0]})

    if intent['context_set'] == "dureeGivenCE" and not duree_key in simulateurCreditCE:
        simulateurCreditCE.update({duree_key: num[0]})

    if intent['context_set'] == "SimmulationCE_given_NeedApproval" and not taux_key in simulateurCreditCE:
        simulateurCreditCE.update({taux_key: num[0]})

    if intent['context_set'] == "sumGiven" and not encour_key in simulateurCreditCE:
        simulateurCreditCE.update({encour_key: num[0]})

    if intent['context_set'] == "SimmulationCE_given_NeedApproval" and not encour_key in simulateurCreditCE:
        simulateurCreditCE.update({encour_key: '0'})


def get_simulator_data_CE(user_id):
    salary_key = user_id + "CES"
    duree_key = user_id + "CED"
    taux_key = user_id + "CET"
    encour_key = user_id + "CEC"
    simulation_data = False
    if all(k in simulateurCreditCE for k in (salary_key, duree_key, taux_key, encour_key)):
        simulationJSON = {}
        simulationJSON['salary'] = str(simulateurCreditCE[salary_key])
        simulationJSON['duree'] = str(simulateurCreditCE[duree_key])
        simulationJSON['taux'] = str(simulateurCreditCE[taux_key])
        simulationJSON['autreech'] = str(simulateurCreditCE[encour_key])
        return simulationJSON
    return simulation_data


def clean_simulator_data_CE(user_id):
    salary_key = user_id + "CES"
    duree_key = user_id + "CED"
    taux_key = user_id + "CET"
    encour_key = user_id + "CEC"

    if all(k in simulateurCreditCE for k in (salary_key, duree_key, taux_key, encour_key)):
        del simulateurCreditCE[salary_key]
        del simulateurCreditCE[duree_key]
        del simulateurCreditCE[taux_key]
        del simulateurCreditCE[encour_key]


def search(values, searchFor):
    for k in values:
        for v in values[k]:
            if searchFor in v:
                return k
    return None
