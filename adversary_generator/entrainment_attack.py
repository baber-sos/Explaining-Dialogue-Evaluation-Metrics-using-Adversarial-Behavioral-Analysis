import random
from nltk.corpus.reader import wordlist
from numpy.core.einsumfunc import _can_dot
import spacy
from nltk.corpus import wordnet as wn
from spacy.tokenizer import Tokenizer
from lemminflect import getInflection
# from paraphrase_attack import get_similar

nlp = spacy.load("en_core_web_lg")

def get_similar_from_nltk(word, pos=None, topk=5):
    #pos is coarse-grained pos tag
    if pos not in ['n', 'v', 'a', 'r']:
        synsets = wn.synsets(word)
    else:
        synsets = wn.synsets(word, pos=pos)
    
    word_synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            word_synonyms.add(lemma.name())
    word_synonyms = list(filter(lambda syn_name: syn_name != word, word_synonyms))
    word_lex = nlp.vocab[word]
    similar_words = list()
    for candidate in word_synonyms:
        # print(candidate, type(candidate))
        cand_lex = nlp.vocab[candidate]
        # print("this is the candidate token: {}".format(candidate))
        if cand_lex.is_oov:
            # print('This candidate token is out of vocab')
            continue
        similar_words.append((cand_lex.text, cand_lex.similarity(word_lex)))
    similar_words.sort(key=lambda x: x[1], reverse=True)
    if topk:
        similar_words = similar_words[:topk]
    print("Number of similar words found: {} and the topk is: {}".format(len(similar_words), topk))
    if len(similar_words) == 0:
        return [], []
    return tuple(zip(*similar_words))

def modify_entrainment(conversation):
    #add an entity filter 
    pos_map = {"NOUN" : 'n', "VERB" : 'v', "ADJ" : 'a', "ADV" : 'r', "PROPN" : 'n'}
    tokenizer = Tokenizer(nlp.vocab)
    conv_tokens = nlp(conversation)
    cur_speaker = 0
    sp1_tokens = set()
    sp2_tokens = set()
    occurr_pos = dict()

    for i, token in enumerate(conv_tokens):
        if token.text == '\n':
            cur_speaker = (cur_speaker + 1) % 2
            continue
        if token.is_stop or token.is_punct or len(token.text) == 1:
            continue
        if cur_speaker == 0:
            sp1_tokens.add(token.text)
        else:
            sp2_tokens.add(token.text)
        occurr_pos.setdefault(token.text, [])
        occurr_pos[token.text].append(i)

    entr_cands = list(sp1_tokens & sp2_tokens)
    # entr_cands = ['use']
    print(entr_cands)
    if len(entr_cands) == 0:
        return None, None

    similar_ones = ([], [])
    while similar_ones[0] == [] and len(entr_cands):
        entr_token = random.choice(entr_cands)
        to_modify = random.choice(occurr_pos[entr_token][1:])
        sel_tok = conv_tokens[to_modify].text
        sel_tok_start = conv_tokens[to_modify].idx

        wn_pos = pos_map.get(conv_tokens[to_modify].pos_, None)
        similar_ones = get_similar_from_nltk(sel_tok, pos=wn_pos)
        # print(similar_ones[0])
        entr_cands.remove(entr_token)

    if len(similar_ones[0]) == 0:
        return None, None
    sampled_one = random.choice(similar_ones[0])

    try:
        new_line_index = conv_tokens.text.index('\n', sel_tok_start)
    except ValueError as e:
        new_line_index = len(conv_tokens.text)
    orgnl_conv = conv_tokens.text[:new_line_index]

    orgn_pos = conv_tokens[to_modify].tag_
    inflections = getInflection(sampled_one, tag=orgn_pos)
    print("The sampled one: {} and the original pos tag: {}".format(sampled_one, orgn_pos))
    print("the possible inflections:", inflections)
    if len(inflections) >= 1:
        sampled_one = inflections[0]
    mod_conv = orgnl_conv[:sel_tok_start] + orgnl_conv[sel_tok_start:].replace(sel_tok, sampled_one, 1)
    
    return orgnl_conv.split('\n'), mod_conv.split('\n')

def entrainment_changes(conversation):
    orgnl_conv, mod_conv = modify_entrainment(conversation)
    return [orgnl_conv], [mod_conv]

# conversation = "Excuse me , can I use your computer to type my paper ?\nNo problem .\nI am afraid I can't finish typing it this afternoon.When will you use it tonight ?\nOh , Never mind , I finished my paper.So you can use it tonight ."
# conversation = "Welcome to IBA . Which service do you require ?\n" + \
# "I hope you can help me . I've been told about something called ' Financing Link ' ?\n" + \
# "Yes , that is our Personal Wealth Management Service .\n" + \
# "Could you tell me more ?\n" + \
# "Of course . Financing Link is a value-added service , and can be tailored to suit your requirements ."
# from sample import conversation_sampler
# # random.seed(10)
# count = 15
# sampler = conversation_sampler()
# for i in range(count):
#     conversation = sampler.sample().strip()
#     # print("The original conversation: {}".format(conversation))
# orgnl_conv, mod_conv = modify_entrainment(conversation)
# print('The original conversation:', orgnl_conv)
# print('------------------------')
# print('The modified conversation:', mod_conv)
# print('########################')