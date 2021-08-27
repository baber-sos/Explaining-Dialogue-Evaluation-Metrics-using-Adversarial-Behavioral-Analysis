#Following types of mutations
#1st person pronouns to 2nd/3rd person pronouns and vice versa
#Gender Neutral Pronouns to Gender Specific Pronouns and vice versa
#Collective Pronouns to Singular Pronouns and vice versa

import random

subject_pronouns = set(['i', 'you', 'he', 'she', 'it', 'this', 'they', 'we', 'you', 'they'])
object_pronouns = set(['me', 'you', 'him', 'her', 'it', 'us', 'you', 'your', 'them'])
possessive_pronouns = set(['mine', 'yours', 'his', 'hers', 'ours', 'yours', 'theirs'])
possessive_adjectives = set(['my', 'your', 'his', 'her', 'its', 'her', 'our', 'your', 'their'])
reflexive_pronouns = set(['myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'])

all_pronouns = set(subject_pronouns | object_pronouns | possessive_pronouns | possessive_adjectives | reflexive_pronouns)

first_person_pronouns = set(['i', 'we', 'me', 'us', 'mine', 'ours', 'my', 'our', 'myself', 'ourselves'])
second_person_pronouns = set(['you', 'your', 'yours', 'us', 'yourself', 'yourselves'])
third_person_pronouns = set(['he', 'she', 'it', 'they', 'him', 'her', 'them', 'hers', 'theirs', \
                        'their', 'herself', 'himself', 'themselves'])

gndr_spc_pnouns = set(['he', 'she', 'her', 'him', 'his', 'hers', 'herself', 'himself'])

collective_pnouns = set(['they', 'we', 'us', 'them', 'theirs', 'their', 'our', 'ourselves', 'themselves', 'yourselves'] + \
    ['you', 'your', 'yours'])

def get_utterances_with_pnouns(utterances):
    utterances_with_pnouns = []
    for i, utt in enumerate(utterances):
        if i == 0:
            continue
        for token in utt.split(' '):
            if token in all_pronouns:
                utterances_with_pnouns.append(i)
                break
    return utterances_with_pnouns

def get_pronoun_type(pronoun, reference_types, ptypes):
    rtype = -1
    app_ptypes = []

    for i, rtype_pnouns in enumerate(reference_types):
        if pronoun in rtype_pnouns:
            rtype = i
            break

    for i, ptype in enumerate(ptypes):
        if pronoun in ptype:
            app_ptypes.append(i)

    return rtype, app_ptypes

def mutate_reference_types(utterance):
    tokens = utterance.split(' ')
    reference_types = [first_person_pronouns, second_person_pronouns, third_person_pronouns]
    ptypes = [subject_pronouns, object_pronouns, possessive_pronouns, possessive_adjectives, reflexive_pronouns]
    for i, token in enumerate(tokens):
        token = token.lower()
        if token in all_pronouns:
            rtype, app_ptypes = get_pronoun_type(token, reference_types, ptypes)
            ptype = random.choice(app_ptypes)
            cddt_ref_pnouns = (reference_types[(rtype + 1) % 3] | reference_types[(rtype + 2) % 3])
            tokens[i] = random.choice(list(ptypes[ptype] & cddt_ref_pnouns))
    return ' '.join(tokens)

def mutate_gender_type(utterance):
    ptypes = [subject_pronouns, object_pronouns, possessive_pronouns, possessive_adjectives, reflexive_pronouns]
    gndr_ntrl_pnouns = all_pronouns - gndr_spc_pnouns
    tokens = utterance.split(' ')
    for i, token in enumerate(tokens):
        token = token.lower()
        if token in gndr_spc_pnouns:
            _, app_ptypes = get_pronoun_type(token, set(), ptypes)
            ptype = random.choice(app_ptypes)
            tokens[i] = random.choice(list(ptypes[ptype] & gndr_ntrl_pnouns))
        elif token in gndr_ntrl_pnouns:
            _, app_ptypes = get_pronoun_type(token, set(), ptypes)
            ptype = random.choice(app_ptypes)
            tokens[i] = random.choice(list(ptypes[ptype] & gndr_spc_pnouns))
    return ' '.join(tokens)

def mutate_collectives(utterance):
    snglr_pnouns = (all_pronouns - collective_pnouns) | set(['you', 'your', 'yours'])
    ptypes = [subject_pronouns, object_pronouns, possessive_pronouns, possessive_adjectives, reflexive_pronouns]
    tokens = utterance.split(' ')
    for i, token in enumerate(tokens):
        token = token.lower()
        pnoun_ctgrs = []
        mutation_use = None
        if token in snglr_pnouns:
            pnoun_ctgrs.append(collective_pnouns)
        if token in collective_pnouns:
            pnoun_ctgrs.append(snglr_pnouns)

        if len(pnoun_ctgrs) > 0:
            mutation_use = random.choice(pnoun_ctgrs)
            _, app_ptypes = get_pronoun_type(token, set(), ptypes)
            ptype = random.choice(app_ptypes)
            tokens[i] = random.choice(list((ptypes[ptype] & mutation_use)))
    return ' '.join(tokens)

def mutate_pronouns(args, conversation, attack_type=-1):
    utterances = [utt.strip() for utt in conversation.split('\n')]
    utterances_with_pnouns = get_utterances_with_pnouns(utterances)
    if not len(utterances_with_pnouns):
        return [None], [None]

    # num_utt = random.randint(1, len(utterances_with_pnouns))
    num_utt = 1
    #choose the attack
    # attack_type = random.randint(0, 2)
    # print(utterances)

    print('This is the attack type: {}'.format(attack_type))
    print('Number of utteraces to change: {}'.format(num_utt))
    mttd_convs = []
    num_attacks = 0
    for utt_ix in random.sample(utterances_with_pnouns, num_utt):
        if attack_type == 0 or attack_type == -1:
            new_utt = mutate_reference_types(utterances[utt_ix])
            mttd_convs.append(utterances[:utt_ix] + [new_utt])
            num_attacks += 1
        if attack_type == 1 or attack_type == -1:
            new_utt = mutate_gender_type(utterances[utt_ix])
            mttd_convs.append(utterances[:utt_ix] + [new_utt])
            num_attacks += 1
        if attack_type == 2 or attack_type == -1:
            new_utt = mutate_collectives(utterances[utt_ix])
            mttd_convs.append(utterances[:utt_ix] + [new_utt])
            num_attacks += 1

    # return '\n'.join(utterances)
    # print(utterances)
    # print(mttd_convs)
    return [conversation.split('\n')[:utt_ix + 1]] * num_attacks, mttd_convs

def get_pronoun_stats(conversation, pronoun_counts):
    for pronoun in all_pronouns:
        for utterance in conversation.split('\n'):
            pronoun_counts.setdefault(pronoun, 0)
            pronoun_counts[pronoun] += utterance.strip().lower().split(' ').count(pronoun)
    return pronoun_counts