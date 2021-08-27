import random
import spacy
from spacy.tokens import Doc
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lemminflect import getInflection
from entrainment_attack import get_similar_from_nltk

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model.to("cuda:0")
nlp = spacy.load("en_core_web_lg")

#some spacy 2.3 fixups for probability lookup tables
if nlp.vocab.lookups_extra.has_table("lexeme_prob"):
    nlp.vocab.lookups_extra.remove_table("lexeme_prob")

nlp.vocab["apple"].prob

similarity_index = dict()

def get_names_and_probs(lex_list):
    words = []
    probs = []
    for lex in lex_list:
        words.append(lex.text)
        probs.append(lex.prob)
    return words, probs

def get_similar(word, topk=5):
    if word in similarity_index:
        return get_names_and_probs(similarity_index[word][:topk])
    word_lex = nlp.vocab[word]
    
    filtered = []
    for lex in [nlp.vocab[orth] for orth in word_lex.vocab.vectors]:
        if lex.is_lower == word_lex.is_lower and lex.prob >= -15 and np.count_nonzero(lex.vector) and word_lex.text != lex.text:
            filtered.append(lex)
    
    filtered.sort(key=lambda x: -x.similarity(word_lex))
    similarity_index[word] = filtered
    return get_names_and_probs(filtered[:topk])

def paraphrase(conversation, topk=5, min_words=3, randomize_stop=False, token_prob=0.25):
    pos_map = {"NOUN" : 'n', "VERB" : 'v', "ADJ" : 'a', "ADV" : 'r', "PROPN" : 'n'}
    utterances = conversation.split('\n')
    if len(utterances) < 3:
        return None, None
    to_randomize = random.randint(2, len(utterances) - 1)
    orgnl_conv = '\n'.join(utterances[:to_randomize + 1])
    utt = utterances[to_randomize]
    utt_doc = nlp(utt)
    tokens = [token.text for token in utt_doc]
    if not randomize_stop:
        token_ixs = [token.i for token in utt_doc if not \
                        (token.is_stop or token.is_punct or len(token.text) == 1 or token.whitespace_ == '')]
    else:
        token_ixs = [token.i for token in utt_doc if not token.is_punct]

    if len(token_ixs) < min_words:
        return None, None

    # num_words = random.randint(1, max_words)
    # num_words = min(num_words, len(token_ixs))

    print('Candidate tokens:', [utt_doc[i] for i in token_ixs])
    # tokens_to_change = random.sample(token_ixs, k=num_words)
    # print(num_words, utt)
    change_flag = False
    for tok_i in token_ixs:
        #later replace with weighted sampling
        wn_pos = pos_map.get(utt_doc[tok_i].pos_, None)
        # similar_words, log_probs = get_similar(tokens[tok_i], topk=topk)
        similar_words, log_probs = get_similar_from_nltk(tokens[tok_i], pos=wn_pos, topk=topk)
        print(tokens[tok_i], similar_words, [nlp.vocab[x].prob for x in similar_words])
        if len(similar_words) == 0:
            continue
        change_flag = True
        if random.random() < token_prob:
            min_ix = np.argmin([nlp.vocab[x].prob for x in similar_words])
            sampled_one = similar_words[min_ix]
        else:
            sampled_one = random.choice(similar_words)
        # print(tokens[tok_i, utt_doc[tok_i]])
        inflections = getInflection(sampled_one, tag=utt_doc[tok_i].tag_)
        print(tokens[tok_i], utt_doc[tok_i], sampled_one, utt_doc[tok_i].tag_, inflections, wn_pos)
        if len(inflections) > 0:
            sampled_one = inflections[0]
        tokens[tok_i] = sampled_one

    if not change_flag:
        return None, None

    new_utt = Doc(utt_doc.vocab, words=tokens).text
    utterances[to_randomize] = new_utt

    # mod_conv = '\n'.join(utterances[:to_randomize + 1])
    mod_conv = utterances[:to_randomize + 1]
    return orgnl_conv.split('\n'), mod_conv

def paraphrase_using_pretrained(conversation, topk=10):
    utterances = conversation.split('\n')
    if len(utterances) < 3:
        return None, None
    to_paraphase = random.randint(2, len(utterances) - 1)
    text =  "paraphrase: " + utterances[to_paraphase] + " </s>"
    
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda:0"), encoding["attention_mask"].to("cuda:0")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=1024,
        do_sample=True,
        top_k=256,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=topk,
    )
    print("Original Sentence")
    print(utterances[to_paraphase])
    print("Modified Sentence")
    print(tokenizer.decode(outputs[1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    print("-------------")
    orgnl_conv = '\n'.join(utterances[:to_paraphase + 1])
    utterances[to_paraphase] = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    mod_conv = utterances[:to_paraphase + 1]
    return orgnl_conv.split('\n'), mod_conv
        
def generate_paraphrases(conversation, topk=5, min_words=10, randomize_stop=False, attack_type=0):
    if attack_type == 0:
        orgnl_conv, mttd_conv = paraphrase(conversation, topk=topk, min_words=min_words, randomize_stop=randomize_stop)
    else:
        orgnl_conv, mttd_conv = paraphrase_using_pretrained(conversation, topk=topk)
    return [orgnl_conv], [mttd_conv]
    
# from sample import conversation_sampler
# # random.seed(10)
# count = 30
# sampler = conversation_sampler()
# for i in range(count):
#     conversation = sampler.sample().strip()
#     orgnl_conv, mod_conv = paraphrase(conversation)
#     # paraphrase_using_pretrained(conversation)
#     print('The original conversation:', orgnl_conv)
#     print('------------------------')
#     print('The modified conversation:', mod_conv)
#     print('------------------------')