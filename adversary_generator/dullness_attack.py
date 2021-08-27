#That is very interesting.
#That is amazing.
#I love that.
#Wow.
#That is great!
#I am very happy to hear that!
#That is good!
#That is cool!
#Nice!
#I am happy for you!
#Awesome!

from functools import total_ordering
from speaker_sensitive_attacks import get_all_qa_pairs
import random

general_responses = ["That is very interesting.", "That is amazing.", "I love that.", "Wow.", \
    "That is great.", "I am very happy to hear that.", "That is good.", "That is cool.", "Nice!", "I am happy for you!", "Awesome!", \
        "I like that.", "I see."]
general_answers = ["I don't have a good answer to that.", "That is a good question.", "That question does not make sense to me.", \
    "That is a great question.", "Let me think what I have to say about that!", "I don't know what you mean by that.", "Wow, great question!"]

def add_generic_utterance(conversation):
    utterances = conversation.split('\n')
    pairs = get_all_qa_pairs(utterances)
    if len(pairs) == 0:
        return None, None
    question = random.choice(pairs)
    orgnl_conv = '\n'.join(utterances[:question + 3])
    utterances[question + 2] = random.choice(general_responses)
    # mod_conv = '\n'.join(utterances[:question + 3])
    mod_conv = utterances[:question + 3]
    return orgnl_conv.split('\n'), mod_conv

def repeat_an_utterance(conversation):
    utterances = conversation.split('\n')
    non_question_ixs = [i for i in range(len(utterances) - 2) if '?' not in utterances[i]]
    # i, i + 1, i + 2
    if len(non_question_ixs) == 0:
        return None, None
    
    to_repeat = random.choice(non_question_ixs)
    orgnl_conv = '\n'.join(utterances[:to_repeat + 3])
    utterances[to_repeat + 2] = utterances[to_repeat]
    # mod_conv = '\n'.join(utterances[:to_repeat + 3])
    mod_conv = utterances[:to_repeat + 3]
    return orgnl_conv.split('\n'), mod_conv

def add_generic_answer(conversation):
    utterances = conversation.split('\n')
    pairs = get_all_qa_pairs(utterances)
    if len(pairs) == 0:
        return None, None
    question = random.choice(pairs)
    orgnl_conv = '\n'.join(utterances[:question + 2])
    utterances[question + 1] = random.choice(general_answers)
    # mod_conv = '\n'.join(utterances[:question + 2])
    mod_conv = utterances[:question + 2]
    return orgnl_conv.split('\n'), mod_conv

def introduce_dullness(conversation, attack_type=-1):
    assert -1 <= attack_type <= 2, "Please specify an attack type in the range 0-2."
    if attack_type == -1:
        to_return = [add_generic_utterance(conversation), repeat_an_utterance(conversation), add_generic_answer(conversation)]
    elif attack_type == 0:
        to_return = [add_generic_answer(conversation)]
    elif attack_type == 1:
        to_return = [repeat_an_utterance(conversation)]
    elif attack_type ==2 :
        to_return = [add_generic_utterance(conversation)]
    return list(zip(*to_return))

# from sample import conversation_sampler
# # random.seed(10)
# count = 15
# sampler = conversation_sampler()
# for i in range(count):
#     conversation = sampler.sample().strip()
#     print('case 1')
#     orgnl_conv, mod_conv = add_generic_utterance(conversation)
#     print('The original conversation:', orgnl_conv)
#     print('------------------------')
#     print('The modified conversation:', mod_conv)
#     print('------------------------')
#     print('case 2')
#     orgnl_conv, mod_conv = add_generic_answer(conversation)
#     print('The original conversation:', orgnl_conv)
#     print('------------------------')
#     print('The modified conversation:', mod_conv)
#     print('------------------------')
#     print('case 3')
#     orgnl_conv, mod_conv = repeat_an_utterance(conversation)
#     print('The original conversation:', orgnl_conv)
#     print('------------------------')
#     print('The modified conversation:', mod_conv)
#     print('------------------------')