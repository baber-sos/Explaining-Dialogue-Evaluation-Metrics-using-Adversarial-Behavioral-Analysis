from os import replace
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize
import neuralcoref
import random
from ner_mutations import ner_cats, ner_categories, build_dataset_information
from nltk.tokenize.treebank import TreebankWordDetokenizer

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp, greedyness=0.5, max_dist=100)

stopwords = set(stopwords.words('english'))

global build_ner_info
build_ner_info = False

#' d replacing this with would might help
#I see.
#Nice to see you.
#Let's.

def get_antonyms(word):
    to_return = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            to_return.extend([ant.name() for ant in lemma.antonyms()])
    return to_return

def contradiction_through_synonyms(utterances):
    synonym_to_utterance = dict()
    syn_utt_ix = dict()
    for i, utt in enumerate(utterances):
        for j, word in enumerate(word_tokenize(utt.lower())):
            if word in stopwords or not word.isalnum():
                # print(f"Removing this: {word}")
                continue
            synonym_to_utterance.setdefault(word, set())
            synonym_to_utterance[word].add(i)

            syn_utt_ix.setdefault(word, dict())
            syn_utt_ix[word].setdefault(i, -1)
            syn_utt_ix[word][i] = j
            for synonym in wn.synsets(word):
                synonym_to_utterance.setdefault(synonym.name(), set())
                synonym_to_utterance[synonym.name()].add(i)

                syn_utt_ix.setdefault(synonym.name(), dict())
                syn_utt_ix[synonym.name()].setdefault(i, -1)
                syn_utt_ix[synonym.name()][i] = j
    
    all_utt_pairs = set()
    pair_to_ants = dict()
    for syn_word in synonym_to_utterance.keys():
        if len(synonym_to_utterance[syn_word]) <= 1:
            continue
        ants = get_antonyms(syn_word)
        if len(ants) < 1:
            continue
        utt_ixs = list(synonym_to_utterance[syn_word])
        for i in range(len(utt_ixs)):
            for j in range(i + 1, len(utt_ixs)):
                utt_pair = [utt_ixs[i], utt_ixs[j]]
                utt_pair.sort()
                i_, j_ = utt_pair 
                utt_pair_tokens = [syn_utt_ix[syn_word][i_], syn_utt_ix[syn_word][j_]]
                all_utt_pairs.add((tuple(utt_pair), tuple(utt_pair_tokens)))
                pair_to_ants[tuple([*utt_pair, *utt_pair_tokens])] = ants
    dtknzr = TreebankWordDetokenizer()
    if len(all_utt_pairs):
        contra_cands = random.choice(list(all_utt_pairs))
        utt_token_pairs = list(zip(*contra_cands))
        # print(utterances[])
        for utt_i, word_i in utt_token_pairs:
            print(f"The utterance: {utterances[utt_i]}, The candidate token: {word_tokenize(utterances[utt_i])[word_i]}")
        sec_utt_ix, sec_utt_tkn_ix  = utt_token_pairs[1]
        second_utt_tokens = word_tokenize(utterances[sec_utt_ix])
        print('These are the antonyms to choose from:', pair_to_ants[tuple([*contra_cands[0], *contra_cands[1]])])
        print(second_utt_tokens)
        second_utt_tokens[sec_utt_tkn_ix] = random.choice(pair_to_ants[tuple([*contra_cands[0], *contra_cands[1]])])
        print(second_utt_tokens)
        utterances[sec_utt_ix] = dtknzr.detokenize(second_utt_tokens)
        print(utterances[sec_utt_ix])
        return utterances[:sec_utt_ix + 1]
    else:
        return []

def get_speaker(char_ix, newlines):
    # speaker = 1
    # for i in range(len(new_lines)):
    #     speaker = (speaker + 1) % 2
    transformed = [i >= char_ix for i in newlines]
    try:
        return transformed.index(1) % 2
    except Exception:
        return len(transformed)
    # return speaker
    

def contradiction_through_repeated_entities(sampler, conversation):
    global build_ner_info
    if not build_ner_info:
        print('Building NER information for the given dataset')
        build_dataset_information(sampler, True)
        build_ner_info = True
    # print(ner_categories)
    new_lines = [i for i in range(len(conversation)) if conversation[i] == '\n']
    doc = nlp(conversation)
    replace_pairs = []
    entity_counts = dict()
    for ent in doc.ents:
        if ent.text == "Ve":
            continue
        print('Associated coref cluster:', ent, ent._.coref_cluster)
        entity_counts.setdefault(ent.text, [])
        entity_counts[ent.text].append(ent)
    
    rand_ent_pairs = []
    for entity in entity_counts:
        selected_pair = []
        if len(entity_counts[entity]) > 1:
            mentions = entity_counts[entity]
            entity_speakers = [get_speaker(ent.start_char, new_lines) for ent in mentions]
            print(entity_speakers)
            entity_speakers.sort()
            try:
                speaker_change_ix = entity_speakers.index(1)
            except Exception as e:
                speaker_change_ix = len(entity_speakers)
            speaker_pairs = []
            if len(mentions[:speaker_change_ix]) > 1:
                speaker_pairs.append(random.sample(mentions[:speaker_change_ix], k=2))
            if len(mentions[speaker_change_ix:]) > 1:
                speaker_pairs.append(random.sample(mentions[speaker_change_ix:], k=2))
            if len(speaker_pairs) > 0:
                selected_pair = random.choice(speaker_pairs)
                selected_pair.sort(key=lambda x: x.start_char)
                rand_ent_pairs.append(selected_pair)
        if len(selected_pair):
            print('The selected entity pair:', [(ent.start_char, ent.text) for ent in selected_pair])
    if len(rand_ent_pairs):
        selected_pair = random.choice(rand_ent_pairs)
        print('The selected entity pair:', [(ent.start_char, ent.text) for ent in selected_pair])
        conversation[selected_pair[1].start_char:]
        try:
            end_ix = conversation[selected_pair[1].start_char:].index('\n') + selected_pair[1].start_char
        except Exception:
            end_ix = len(conversation)
        mutation = random.choice(ner_categories[selected_pair[1].label_])
        print('This is the mutation:', mutation)
        print("This is the end index:", end_ix)
        mutated_conversation = conversation[:selected_pair[1].start_char] + mutation + conversation[selected_pair[1].end_char:end_ix]
        print(mutated_conversation)
        return conversation[:end_ix].split('\n'), mutated_conversation.split('\n')
    else:
        return None, None
        # if ent._.coref_cluster:
        #     cluster = ent._.coref_cluster
        #     main_mention = cluster.main
        #     for mention in cluster.mentions:
        #         if mention.start_char == main_start:
        #             continue
        #         replace_pairs.append((main_mention.start_char, main_mention.end_char), (mention.start_char, mention.end_char))

def contradiction_through_coreference(sampler, conversation):
    global build_ner_info
    if not build_ner_info:
        print('Building NER information for the given dataset')
        build_dataset_information(sampler, True)
        build_ner_info = True
    
    replace_pairs = []
    for ent in nlp(conversation).ents:
        if ent._.coref_cluster:
            cluster = ent._.coref_cluster
            main_mention = cluster.main
            main_start = ent.start_char
            print("This is the detected cluster:", cluster)
            print("This is the entity name:", ent.text)
            print("This is the main mention in the cluster:", cluster.main.text)
            for mention in cluster.mentions:
                if mention.start_char == main_start:
                    continue
                replace_pairs.append(((mention.start_char, mention.end_char), ent.label_))
    if len(replace_pairs):
        mention_ent, ent_label = random.choice(replace_pairs)
        print('This is the mention to be replaced:', conversation[mention_ent[0]:mention_ent[1]])
        try:
            offset = conversation[mention_ent[0]:].index('\n')
        except ValueError as e:
            offset = len(conversation[mention_ent[0]:])
        end_ix = offset + mention_ent[0]
        mutation = random.choice(ner_categories[ent_label])
        print('This is the mutation:', mutation)
        mutated_conversation = conversation[:mention_ent[0]] + mutation  + conversation[mention_ent[1]:end_ix]
        return conversation[:end_ix].split('\n'), mutated_conversation.split('\n')
    return None, None


def check_negation(verb):
    print(verb.text)
    negations = ["no", "not", "n't", "Don"]
    print([tok.text for tok in verb.lefts])
    print([tok.text for tok in verb.rights])
    for token in verb.lefts:
        if token.text in negations:
            return True
    for token in verb.rights:
        if token.text in negations:
            return True
    return False

def contradiction_through_verb_negation(utterances):
    #auxiliary verb => append a not
    #present tense => do/does not depending on if the verb is singular or plural
    #past tense => did not and change verb from to present form
    #All other tenses have auxiliaries. Ignore the questions.
    candidates = []
    for i, utt in enumerate(utterances[:-2]):
        if '?' in utt or "n't" in utt or "n ' t" in utt:
            continue
        for token in nlp(utt):
            if token.pos_ == "VERB":
                if check_negation(token):
                    continue
                candidates.append(i)

    if len(candidates):
        to_modify = random.choice(candidates)
        cur_utt = utterances[to_modify]
        print("This is the utterance before modification:", cur_utt)
        doc = nlp(cur_utt)
        for i, token in enumerate(doc):
            new_utterance = ""
            start_char = token.idx
            end_char = token.idx + len(token.text)
            if len(token.text) == 1 or token.whitespace_ == '':
                continue
            if token.pos_ == "AUX":
                new_utterance = cur_utt[:end_char] + " not" + cur_utt[end_char:]
                print("This is the new utterance:", new_utterance)
                break
            if token.pos_ == "VERB":
                if token.text == "Let" or token.text == "let":
                    if i + 1 < len(doc) and doc[i + 1].text == "'":
                        new_utterance = cur_utt[:end_char] + f"{doc[i + 1].text} {doc[i + 2].text}" + \
                            " not " + cur_utt[doc[i + 2].idx + 1:]
                    elif i + 1 < len(doc) and doc[i + 1].text == "'s":
                        new_utterance = cur_utt[:end_char] + f"{doc[i + 1].text}" + \
                            " not " + cur_utt[doc[i + 1].idx + 2:]
                    else:
                        new_utterance = cur_utt[:start_char] + "Do not " + cur_utt[start_char:]
                elif token.tag_ == "VBD":
                    new_utterance = cur_utt[:start_char] + " did not {}".format(token.lemma_) + cur_utt[end_char:]
                elif token.tag_ == "VBZ":
                    new_utterance = cur_utt[:start_char] + " does not {}".format(token.lemma_) + cur_utt[end_char:]
                elif token.tag_ == "VBP" or token.tag_ == "VB":
                    if token.text == "to":
                        new_utterance = cur_utt[:start_char] + " not {}".format(token.lemma_) + cur_utt[end_char:]
                    else:
                        new_utterance = cur_utt[:start_char] + " do not {}".format(token.lemma_) + cur_utt[end_char:]
                print(token.tag_)
                print("This is the new utterance:", new_utterance)
                break
        if new_utterance == "":
            return None, None
        return utterances[:to_modify + 3], utterances[:to_modify + 2] + [new_utterance]
    return None, None

def generate_contradictions(sampler, conversation):
    utterances = conversation.split('\n')
    orgnl_conv1, mttd_conv1 = contradiction_through_repeated_entities(sampler, conversation)
    orgnl_conv2, mttd_conv2 = contradiction_through_coreference(sampler, conversation)
    orgnl_conv3, mttd_conv3 = contradiction_through_verb_negation(utterances)
    # print(orgnl_conv1, mttd_conv1)
    # print(orgnl_conv3, mttd_conv3)
    return [orgnl_conv1, orgnl_conv2, orgnl_conv3], [mttd_conv1, mttd_conv2, mttd_conv3]

# from sample import conversation_sampler

# generate_contradictions(conversation_sampler(), "Would you like to take a look at the menu , sir ?\n" + \
# "Yes . Thank you .\n" + \
# "Would you care for a drink before you order ?\n" + \
# "A glass of Qingdao beer .\n" + \
# "Yes , sir . I'll bring it over . Have you decided what you'd like , sir ?\n" + \
# "Will you make some recommendation ?")

# x = "Can you manage chopsticks ?\n" + \
# "Why not ? See .\n" + \
# "Good mastery . How do you like our Chinese food ?\n" + \
# "Oh , great ! It's delicious . You see , I am already putting on weight . There is one thing I don't like however , MSG .\n" + \
# "What's wrong with MSG ? It helps to bring out the taste of the food .\n" + \
# "According to some studies it may cause cancer .\n" + \
# "Oh , don't let that worry you . If that were true , China wouldn't have such a large population .\n" + \
# "I just happen to have a question for you guys . Why do the Chinese cook the vegetables ? You see what I mean is that most vitamin are destroyed when heated .\n" + \
# "I don't know exactly . It's a tradition . Maybe it's for sanitary reasons ."

# x = "May I help you find something , sir ?\n" + \
# "I'm looking for an engagement ring for my girlfriend . I have an idea of what she likes , but I want to surprise her with something special , too .\n" + \
# "We have all shapes , sizes , qualities and price ranges , do you know about the four Cs of picking a diamond ?\n" + \
# "I think so . Aren't the four Cs , cut , clarity , carat and color .\n" + \
# "You've got it . Tell me a little bit about what you might be wanted .\n" + \
# "Well , my price range is a 5,000 dollars to 7,000 dollars , I'm looking for a marquise cut on the wide band .\n" + \
# "You have good taste . Let my show you what I have ."

# x = "Good afternoon . How can I help ?\n" + \
# "Hi there . I need to change some dollars into local currency . Can I do that here ?\n" + \
# "Yes , you can . Is that US dollars or HK dollars ? Both are available for exchange to RIB . How much would you like to exchange ?\n" + \
# "Well , that depends . What's the rate like today ?\n" + \
# "Today's rate is 821.32 USD to 100 RIB , which isn't bad .\n" + \
# "That sounds pretty good . OK , I'll change 500 USD today , thanks ."

# sampler = conversation_sampler()
# # random.seed(10)
# # for i in range(50):
# #     x = sampler.sample()
# #     print(x)
# generate_contradictions(sampler, x)
# #     print('******************************')