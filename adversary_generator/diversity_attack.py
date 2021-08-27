import random
from numpy.core.numeric import extend_all
import spacy
from spacy.tokenizer import Tokenizer
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.text import TokenSearcher
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from tests.explainers.test_tree import test_skopt_rf_et
import string

stopwords = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_lg')

#replace all occurrences of tokens[i] with tokens[j] in tokens
def replace(tokens, i, j, start, end):
    for t in range(start, end):
        if tokens[t] == tokens[i]:
            tokens[t] = tokens[j]
    return tokens

def get_tokens(conversation):
    tokens = []
    for utt in conversation.split('\n'):
        tokens.extend(word_tokenize(utt) + ['\n'])
    return tokens

def get_conv_from_tokens(tokens, start=-1):
    if start != -1:
        try:
            tokens = tokens[:tokens.index('\n', start)]
        except Exception as e:
            pass #change is in the last utterance
    dtknzr = TreebankWordDetokenizer()
    modified_conv = []
    for utt in dtknzr.detokenize(tokens).split('\n'):
        modified_conv.append(utt.strip())
    return modified_conv

def vocabulary_diversity(conversation, topk=5):
    tokens = get_tokens(conversation)
    tkn_pairs = []
    tkn_counts = dict()
    for i in range(len(tokens)):
        tkn_i = tokens[i]
        if tkn_i in stopwords:
            continue
        tkn_counts.setdefault(tkn_i, 0)
        tkn_counts[tkn_i] += 1
        for j in range(i + 1, len(tokens)):
            tkn_j = tokens[j]
            if tkn_i == tkn_j:
                continue
            syns_i = []
            syns_j = []
            for x in wn.synsets(tkn_i):
                syns_i.extend(x.lemma_names())
            for x in wn.synsets(tkn_j):
                syns_j.extend(x.lemma_names())

            #(i, j) => i in synset j
            # print(tkn_i, syns_j[0] if len(syns_j) >= 1 else None)
            if (tkn_i in syns_j):
                tkn_pairs.append((i, j))
            if (tkn_j in syns_i):
                tkn_pairs.append((j, i))
    # random.choice(tkn_pairs)
    tkn_pairs.sort(key=lambda x: -tkn_counts.get(tokens[x[0]], 0))
    if len(tkn_pairs) == 0:
        return None, None

    tkn_pair = random.choice(tkn_pairs[:topk])
    try:
        utt_end = tkn_pair[0] + tokens[tkn_pair[0]:].index('\n')
    except Exception as _:
        utt_end = len(tokens)
    utt_start = tkn_pair[0]
    while utt_start >= 0 and tokens[utt_start] != '\n':
        utt_start -= 1

    utt_pairs = []
    for i, j in tkn_pairs:
        if utt_start <= i < utt_end:
            utt_pairs.append((i, j))
    print(f"Start and end tokens: {tokens[utt_start + 1]}, {tokens[utt_end - 1]}")
    for i, j in utt_pairs:
        tokens = replace(tokens, i, j, utt_start, utt_end)
    
    return get_conv_from_tokens(get_tokens(conversation)[:utt_end]), get_conv_from_tokens(tokens[:utt_end])
    
def word_repetition(conversation, max_rept=5):
    tokens = get_tokens(conversation)
    tkn_ix = random.randint(0, len(tokens) - 1)
    
    max_tries = 10
    tries = 0
    while tokens[tkn_ix] == '\n' or tokens[tkn_ix] in string.punctuation and tries < max_tries:
        tkn_ix = random.randint(0, len(tokens) - 1)
        tries += 1
    
    if tries == max_tries:
        return None, None

    rep_count = random.randint(1, max_rept)
    orgnl_conv = get_conv_from_tokens(tokens, start=tkn_ix)
    
    print("Repeating '{}' {} times".format(tokens[tkn_ix], rep_count))
    for _ in range(rep_count):
        tokens.insert(tkn_ix, tokens[tkn_ix])
    
    return orgnl_conv, get_conv_from_tokens(tokens, start=tkn_ix)

def phrase_repeat(conversation, max_rept=3):
    doc = nlp(conversation)
    noun_chunk_bounds = []
    for chunk in doc.noun_chunks:
        if chunk.end - chunk.start == 1:
            continue
        start_char, end_char, start, end = chunk.start_char, chunk.end_char, chunk.start, chunk.end
        noun_chunk_bounds.append((start_char, end_char, start, end))
    if len(noun_chunk_bounds) == 0:
        return None, None

    mod_conv = ''
    start_char, end_char, start, end = random.choice(noun_chunk_bounds)
    print('Noun phrases:', conversation[start_char:end_char])
    rept_count = random.randint(1, max_rept)
    mod_conv = mod_conv + conversation[:start_char]  + conversation[start_char : end_char] + ' '
    mod_conv += ' '.join([conversation[start_char : end_char]] * rept_count)

    try:
        conv_end = start_char + conversation[start_char:].index('\n')
    except Exception as _:
        conv_end = len(conversation)

    mod_conv = mod_conv + conversation[end_char:conv_end]
    return conversation[:conv_end].split('\n'), mod_conv.split('\n')

def change_diversity(conversation, topk=5, max_rept=5, attack_type=-1):
    assert attack_type in [-1, 0, 1, 2], "Please specify an attack type less than 3."
    if attack_type == -1:
        # vocabulary_diversity(conversation, topk=topk), \
        to_return = [word_repetition(conversation, max_rept=max_rept), \
            phrase_repeat(conversation, max_rept=max_rept)]
    elif attack_type == 0:
        to_return = [vocabulary_diversity(conversation, topk=topk)]
    elif attack_type == 1:
        to_return = [word_repetition(conversation, max_rept=max_rept)]
    else:
        to_return = [phrase_repeat(conversation, max_rept=max_rept)]
    return list(zip(*to_return))

# from sample import conversation_sampler
# count = 15
# sampler = conversation_sampler()
# for i in range(count):
#     print("Attack number: {}".format(i + 1))
#     conversation = sampler.sample().strip()
# #     for i, (orgnl, mod) in enumerate(change_diversity(conversation)):
# #         if not mod:
# #             continue
# #         print("Case {}".format(i + 1))
# #         print("-----------------")
# #         print('\n'.join(orgnl))
# #         print("-----------------")
# #         print('\n'.join(mod))
# #         print("*****************")
#     # paraphrase_using_pretrained(conversation)
#     orgnl_conv, mod_conv = vocabulary_diversity(conversation)
#     if mod_conv:
#         print('The original conversation:', '\n'.join(orgnl_conv))
#         print('------------------------')
#         print('The modified conversation:', '\n'.join(mod_conv))
#         print('------------------------')