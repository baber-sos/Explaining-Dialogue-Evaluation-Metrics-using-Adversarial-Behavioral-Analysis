import random

def get_all_qa_pairs(utterances):
    pairs = []
    for i, utt in enumerate(utterances[:-2]):
        if '?' in utt:
            pairs.append(i)
    return pairs
    

def manipulate_contributions(conversation):
    #Assumption that each alternating utterance is from a different speaker.
    utterances = conversation.split('\n')
    pairs = get_all_qa_pairs(utterances)
    if len(pairs):
        random.shuffle(pairs)
        for qa in pairs:
            if qa + 2 >= len(conversation):
                continue
            return [utterances[:qa + 3]], [utterances[:qa + 2] + [utterances[qa + 1] + ' ' + utterances[qa + 2]]]
    return [None], [None]