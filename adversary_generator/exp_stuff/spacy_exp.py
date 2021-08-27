import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')

if nlp.vocab.lookups_extra.has_table("lexeme_prob"):
    nlp.vocab.lookups_extra.remove_table("lexeme_prob")

def most_similar(word, topn=5):
    word = nlp.vocab[str(word)]
    queries = [
        w for w in word.vocab 
        if w.is_lower == word.is_lower and w.prob >= -15 and np.count_nonzero(w.vector)
    ]

    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return [(w.lower_,w.similarity(word)) for w in by_similarity[:topn+1] if w.lower_ != word.lower_]

print(most_similar("dog", topn=3))
