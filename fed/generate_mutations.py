from sample import conversation_sampler
import sys
import argparse
import random
import os

def randomize_order(coherent_conversation):
    random.shuffle(coherent_conversation)
    return coherent_conversation

def randomize_utterance(to_change, pick_from):
    index_to_change = random.randint(0, len(to_change) - 1)
    to_change[index_to_change] = random.choice(pick_from[2:])
    return to_change

def mutate_coherence(args, conversation, offtopic_conversation):
    utterances = conversation.split('\n')
    conv_len = len(utterances)
    start = 0
    end = random.randint(start + 3, conv_len)
    coherent_snippet = utterances[start : end]

    decider = random.random()
    order_flag = True
    response_flag = True
    order_flag = False if 1/3 <= decider < 2/3 else True
    response_flag = False if 0 <= decider < 1/3 else True
    incoherent_snippet = list(coherent_snippet)
    if order_flag:
        incoherent_snippet = randomize_order(incoherent_snippet)
    if response_flag: 
        incoherent_snippet = randomize_utterance(incoherent_snippet, offtopic_conversation.split('\n'))
    
    return coherent_snippet, incoherent_snippet

def main(args, sampler):
    for i in range(args.num_examples):
        sampled_conversation = sampler.sample()
        if args.property == 'coherence':
            offtopic_conversation = sampled_conversation
            while hash(offtopic_conversation) == hash(sampled_conversation):
                offtopic_conversation = sampler.sample()
                continue
            coherent_snippet, incoherent_snippet = mutate_coherence(args, sampled_conversation, offtopic_conversation)
            output_dir = args.output_directory if args.output_directory else './test_conversations/coherence_mutations'
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_file = open(os.path.join(output_dir, 'case_{}'.format(i)), 'w')
            for line in \
                ['original_conversation'] + coherent_snippet + ['modified_conversation'] + incoherent_snippet:
                output_file.write(f'{line}\n')
            output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('property', help='Specify how you want to mutate the conversation and generate test cases.', 
                        choices=['coherence'])
    parser.add_argument('-n', '--num_examples', help='number of mutated examples to generate', type=int, default=1)
    parser.add_argument('-sp', '--sample_function', \
                        help='path to a python file name "sample" containing a definition sample_conversation to sample human-human conversations')
    parser.add_argument('--output_directory', help='path to an output directory to store mutated conversations')
    # parser.add_argument('-sp', '--sample_function', \
    #                     help='')
    args = parser.parse_args()
    print(args)
    sampler = conversation_sampler()
    main(args, sampler)
