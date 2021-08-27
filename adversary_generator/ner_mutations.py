from nltk.corpus.reader.comparative_sents import ENTITIES_FEATS
from nltk.data import find
from numpy.core.numerictypes import maximum_sctype
import spacy
import neuralcoref
import random
from tqdm import tqdm

spacy.prefer_gpu()

ner_cats = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', \
    'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
ner_categories = {x : list() for x in ner_cats}

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

#Ve detected as wrong entity
#Names changed in the conversation which are never used again.

def build_dataset_information(sampler, flag, path=None):
    if not flag:
        return
    if path:
        return
    conversations = 20000
    print(f'Building dataset NER stats for max number of conversations: {20000}.')
    
    count = 1
    sampler.randomize()
    for conv in tqdm(sampler.get_next()):
        # print('Conversation Number:', count)
        doc = nlp(conv)
        for ent in doc.ents:
            ner_categories[ent.label_].append(ent.text)
        count += 1
        if count > conversations:
            break
    return ner_categories
    
def get_ner_stats(conversation, ner_stats):
    for ent in nlp(conversation).ents:
        ner_stats.setdefault(ent.label_, 0)
        ner_stats[ent.label_] += 1
    return ner_stats

def mutate_same_entity_type(entities_to_mutate):
    mutated_entities = []
    max_tolerance = 20
    for ent in entities_to_mutate:
        tolerance = 0
        entity_text, ent_label, start_char, end_char = ent.text, ent.label_, ent.start_char, ent.end_char
        if len(ner_categories[ent_label]) == 0:
            continue
        mutated_text = random.choice(ner_categories[ent_label])
        while mutated_text == entity_text and tolerance < max_tolerance:
            mutated_text = random.choice(ner_categories[ent_label])
            tolerance += 1
        
        if tolerance == max_tolerance:
            continue
        mutated_entities.append((mutated_text, ent_label, start_char, end_char))
    return mutated_entities

def mutate_different_entity_type(entities_to_mutate):
    mutated_entities = []
    max_tolerance = 20
    for ent in entities_to_mutate:
        entity_text, ent_label, start_char, end_char = ent.text, ent.label_, ent.start_char, ent.end_char
        mutated_label = random.choice(ner_cats)
        tolerance = 0
        while mutated_label == ent_label or len(ner_categories[mutated_label]) == 0 and tolerance < max_tolerance:
            mutated_label = random.choice(ner_cats)
            tolerance += 1

        if tolerance == max_tolerance:
            continue
        
        if len(ner_categories[mutated_label]) == 0:
            continue

        tolerance = 0
        mutated_text = random.choice(ner_categories[mutated_label])
        while mutated_text == entity_text and tolerance < max_tolerance:
            mutated_text = random.choice(ner_categories[mutated_label])
        
        if tolerance == max_tolerance:
            continue
        mutated_entities.append((mutated_text, mutated_label, start_char, end_char))
    return mutated_entities
    
def mutate_entity_reference(entities_to_mutate, bounds):
    utt_to_mut = dict()
    for ent in entities_to_mutate:
        print(ent._.coref_cluster)
        referent = ent._.coref_cluster.main
        possible_referees = list(filter(lambda x: x.text != referent.text, ent._.coref_cluster.mentions))
        if len(possible_referees) == 0:
            continue
        referee = random.choice(possible_referees)
        referee_utt = find_utterance(referee, bounds)
        utt_to_mut.setdefault(referee_utt, [])
        utt_to_mut[referee_utt].append((referent.text, ent.label_, referee.start_char, referee.end_char))
    if len(utt_to_mut.keys()) == 0:
        return None, None
    mut_utt = random.choice(list(utt_to_mut.keys()))
    print(utt_to_mut.keys())
    print(utt_to_mut[mut_utt])
    return utt_to_mut[mut_utt], mut_utt

def get_conversation_from_ents(ents, conversation):
    ents.sort(key=lambda x : x[-2])
    to_return = ''
    start = 0
    for ent_text, ent_label, start_char, end_char in ents:
        to_return += conversation[start:start_char] + '{}'
        start = end_char
    to_return += conversation[start:]

    to_return = to_return[:-2] if to_return[-2:] == '{}' else to_return
    to_return = to_return.format(*[x[0].strip() for x in ents])
    return to_return.split('\n')

def get_utterance_boundaries(conversation):
    # print(conversation)
    start = 0
    bounds = []
    done = False
    # print("Getting utterance boundaries")
    while not done:
        try:
            newline_ix = conversation[start:].index('\n')
            bounds.append(start + newline_ix)
            start += newline_ix + 1
            # print(start)
        except Exception as _:
            done = True
            bounds.append(len(conversation))
    # print("These are the bounds:", bounds)
    return bounds

def find_utterance(ent, bounds):
    ix = 0
    while ix < len(bounds) and ent.end_char > bounds[ix]:
        ix += 1
    return ix

def mutate_ner_categories(conversation, attack_type=-1):
    print("%%%%%%%%%%%%%%")
    print(conversation)
    print("%%%%%%%%%%%%%%")
    doc = nlp(conversation)
    detected_entities = []
    # attack_type = random.randint(0, 1) if attack_type == -1 else attack_type
    bounds = get_utterance_boundaries(conversation)
    if attack_type > 2:
        print("Please specify an attack type index '<3'")
        print("0 index: Mutate same entity type.")
        print("1 index: Mutate with different entity type.")
        print("2 index: Resolve entity mentions.")
        exit()
        # return
    print("Getting entities")
    utt_to_ents = dict()
    for ent in doc.ents:
        if ent._.coref_cluster:
            # detected_entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
            detected_entities.append(ent)
            ent_utt = find_utterance(ent, bounds)
            if ent_utt == 0:
                continue
            utt_to_ents.setdefault(ent_utt, [])
            utt_to_ents[ent_utt].append(ent)

    if not len(detected_entities) or not len(utt_to_ents.keys()):
        return [None], [None]
    
    utt_to_mutate = random.choice(list(utt_to_ents.keys()))
    # num_entities = random.randint(1, len(detected_entities))
    
    # entities_to_mutate = random.sample(detected_entities, num_entities)
    entities_to_mutate = utt_to_ents[utt_to_mutate]
    new_ents = []
    if attack_type == 0 or attack_type == -1:
        new_ents.append((mutate_same_entity_type(entities_to_mutate), utt_to_mutate))
    if attack_type == 1 or attack_type == -1:
        new_ents.append((mutate_different_entity_type(entities_to_mutate), utt_to_mutate))
    if attack_type == 2:
        new_ents.append(mutate_entity_reference(entities_to_mutate, bounds))
    
    mttd_convs = []
    orgnl_convs = []
    for nes, utt in new_ents:
        if not nes:
            continue
        mttd_convs.append(get_conversation_from_ents(nes, conversation[:bounds[utt]]))
        orgnl_convs.append(conversation[:bounds[utt]].split('\n'))
    
    for i in range(len(orgnl_convs)):
        if orgnl_convs[i] and mttd_convs[i]:
            print('\n'.join(orgnl_convs[i]))
            print("--------------------")
            print('\n'.join(mttd_convs[i]))
            print("####################")
    # print(orgnl_convs, mttd_convs)
    return orgnl_convs, mttd_convs