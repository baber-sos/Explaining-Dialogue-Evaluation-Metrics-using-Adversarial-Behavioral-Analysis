from entrainment_attack import entrainment_changes
import sys
import argparse
import random
import os
import time
import math
from shutil import copyfile
from pronoun_mutations import *
from ner_mutations import *
from contradiction_generation import *
from speaker_sensitive_attacks import *
from diversity_attack import change_diversity
from paraphrase_attack import generate_paraphrases
from dullness_attack import introduce_dullness

import importlib.util

def randomize_order(utterances, start, end):
    coherent_part = utterances[start:end - 1]
    remainder = utterances[:start] + utterances[end:]
    if len(remainder) == 0:
        return None
    random_response = random.choice(remainder)
    return coherent_part + [random_response]

def randomize_utterance(to_change, pick_from):
    if len(to_change) == 1:
        return None
    to_change[-1] = random.choice(pick_from)
    return to_change

def mutate_coherence(args, conversation, offtopic_conversation):
    utterances = conversation.split('\n')
    conv_len = len(utterances)
    # start = random.randint(0, conv_len - 2)
    start = 0
    end = random.randint(start + 2, conv_len)
    coherent_snippet = utterances[start : end]
    
    same_conv_mutation = randomize_order(utterances, start, end)
    multi_conv_mutation = randomize_utterance(list(coherent_snippet), offtopic_conversation.split('\n'))
    print(utterances[start:end])
    print("----------------")
    print(same_conv_mutation)
    print("----------------")
    print(multi_conv_mutation)
    print("################")
    return [coherent_snippet] * 2, [same_conv_mutation, multi_conv_mutation]

def output_to_file(dir_path, filename, orgnl_conv, mttd_conv):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    conv_file = open(os.path.join(dir_path, filename), 'w')
    for line in \
        ['original_conversation'] + orgnl_conv + ['modified_conversation'] + mttd_conv:
        conv_file.write('{}\n'.format(line))
    conv_file.close()

def main(args, sampler):
    pstats = dict()
    nerstats = dict()
    num_examples = args.num_examples if not args.all else len(sampler)
    dataset_iterator = sampler.get_next()
    start_time = time.time()
    output_dir = os.path.join(args.output_directory, \
                    args.property if args.sub_attack_type==-1 else f"{args.property}-{args.sub_attack_type}")
    orgnl_convs, mttd_convs = [], []
    for i in range(num_examples):
        if not args.all:
            sampled_conversation = sampler.sample().strip()
        else:
            sampled_conversation = next(dataset_iterator).strip()

        if args.property == 'display':
            print(sampled_conversation)
            recvd_in = input('type "exit" to shut this utility down: ')
            if recvd_in == 'exit':
                return
            else:
                print('----------')
                continue
        elif args.property == 'entailment':
            offtopic_conversation = sampled_conversation.strip()
            while hash(offtopic_conversation) == hash(sampled_conversation):
                offtopic_conversation = sampler.sample().strip()
                continue
            orgnl_convs, mttd_convs = mutate_coherence(args, sampled_conversation, offtopic_conversation)
            # for conv in mttd_convs:
            #     print('\n'.join(conv))
            #     print("-------------")
        elif args.property == 'pronoun_stats':
            pstats = get_pronoun_stats(sampled_conversation, pstats)
            if i + 1 == num_examples:
                stats = [x for x in pstats.items()]
                stats.sort(key=lambda x: -x[1])
                for k, v in stats[:args.top_k]:
                    print('{}: {}'.format(k, v))
        elif args.property == 'pronoun':
            orgnl_convs, mttd_convs = mutate_pronouns(args, sampled_conversation)
            
        elif args.property == 'ner_stats':
            nerstats = get_ner_stats(sampled_conversation, nerstats)
            if i + 1 == num_examples:
                stats = [x for x in nerstats.items()]
                stats.sort(key=lambda x: -x[1])
                for k, v in stats[:args.top_k]:
                    print('{}: {}'.format(k, v))
        elif args.property == 'ner':
            build_dataset_information(sampler, i == 0, args.load_ner_information)
            orgnl_convs, mttd_convs = mutate_ner_categories(sampled_conversation, attack_type=args.sub_attack_type)
            
        elif args.property == 'ner_neutral':
            build_dataset_information(sampler, i == 0, args.load_ner_information)
            orgnl_convs, mttd_convs = mutate_ner_categories(sampled_conversation, attack_type=2)
            
        elif args.property == 'contradiction':
            orgnl_convs, mttd_convs = generate_contradictions(sampler, sampled_conversation)
            
        elif args.property == 'speaker_sensitiveness':
            orgnl_convs, mttd_convs = manipulate_contributions(sampled_conversation)
            
        elif args.property == 'repetitiveness':
            orgnl_convs, mttd_convs = change_diversity(sampled_conversation, attack_type=args.sub_attack_type, \
                max_rept=args.max_repeat, topk=args.top_k)

        elif args.property == 'diversity':
            orgnl_convs, mttd_convs = change_diversity(sampled_conversation, \
                max_rept=args.max_repeat, topk=args.top_k, attack_type=0)

        elif args.property == 'bad_paraphrase':
            orgnl_convs, mttd_convs = generate_paraphrases(sampled_conversation, \
                topk=args.top_k, min_words=args.min_words, randomize_stop=args.randomize_stop)
            
        elif args.property == 'good_paraphrase':
            orgnl_convs, mttd_convs = generate_paraphrases(sampled_conversation, topk=args.top_k, attack_type=1)
            
        elif args.property == 'entrainment':
            orgnl_convs, mttd_convs = entrainment_changes(sampled_conversation)
            
        elif args.property == 'dullness':
            orgnl_convs, mttd_convs = introduce_dullness(sampled_conversation)
        
        for j, mttd_conv in enumerate(mttd_convs):
            if mttd_conv:
                output_to_file(output_dir, 'case_{}'.format((i * len(mttd_convs)) + j), orgnl_convs[j], mttd_conv)

    print('Total Time spent: {}s'.format(math.ceil(time.time() - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('property', help='Specify how you want to mutate the conversation and generate test cases.', 
                        choices=['entailment', 'display', 'pronoun_stats', 'pronoun', 'ner_stats', 'ner', 'ner_neutral', 'contradiction', \
                            'speaker_sensitiveness', 'repetitiveness', 'diversity', 'bad_paraphrase', 'good_paraphrase', 'entrainment', 'dullness'])
    parser.add_argument('sampler_path', \
                        help='path to a python file name containing a class conversation_sampler to sample human-human conversations')
    parser.add_argument('-n', '--num_examples', help='number of mutated examples to generate', type=int, default=1)
    parser.add_argument('--all', help='use all data examples. Overrides n.', action='store_true')
    parser.add_argument('--output_directory', \
                        help='path to an output directory to store mutated conversations', \
                        default='./test_conversations/')
    parser.add_argument('-l', '--load_ner_information', action='store_true', \
                        help='path to a pickle file containing a dictionary of ner category to text mappings.')
    parser.add_argument('-k', '--top_k', help='Top k elements for different statisctics. Default: 10', \
                        type=int, default=10)
    parser.add_argument('-r', "--max_repeat", help="Max repetitions allowed for attacks which rely on a hyperparameter to manipulate vocabulary.", \
                        type=int, default=5)
    parser.add_argument("-a", "--sub_attack_type", help="Give a sub-attack type if any. Default is -1.", default=-1, type=int)
    parser.add_argument("-m", "--min_words", help="Specify the min_number of words to paraphrase for the paraphrasing attack.", \
                        default=5, type=int)
    parser.add_argument("--randomize_stop", help="Paraphrase stop words or not in the paraphrasing attack.", \
                        action='store_true', default=False)
    parser.add_argument("--dataset_type", help="Specify which type of data. This argument is for the conversation_sampler.", \
                        default="all")
    args = parser.parse_args()

    sample_spec = importlib.util.spec_from_file_location("sample", args.sampler_path)
    sample = importlib.util.module_from_spec(sample_spec)
    sample_spec.loader.exec_module(sample)

    print(args)
    # copyfile(args.sampler_path, './sample.py')
    # from sample import conversation_sampler
    
    sampler = sample.conversation_sampler(dataset_type=args.dataset_type)
    main(args, sampler)
