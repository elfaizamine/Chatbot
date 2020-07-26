# Correction of wrong words ; ex: credi --> crédit , saliar --> salaire
def known(words):
    return set(w for w in words if w in WORDS)

def first_degree_combination(word):
    letters = 'abcdefghijklmnopqrstuvwxyzéàèçêùâ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def second_degree_combination(word):
    return (e2 for e1 in first_degree_combination(word) for e2 in first_degree_combination(e1))

def candidates(word):
    if word.isalpha():
        return known([word]) or known(first_degree_combination(word)) or known(second_degree_combination(word)) or [word]
    else:
        return [word]
 
def words_correction(word, unique_words):
    return max(candidates(word), key=lambda x: unique_words[x])
    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError :
        return False
