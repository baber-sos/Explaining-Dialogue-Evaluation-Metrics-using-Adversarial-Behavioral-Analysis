import os
import argparse
from scipy import stats
import importlib.util
import numpy as np
import random
from tqdm import tqdm

def get_minimum_difference(dof, mean, std, sig_level=0.05):
    req_t_val = stats.t.ppf(1 - (sig_level/2), dof)
    required_score = -1 * ((req_t_val * std/np.sqrt(dof + 1)) - mean)
    print(f"A score difference of at least {abs(mean - required_score)} will be significant.")
    return abs(mean - required_score)

def check_significance(human_score_distribution, modified_scores, significance_value=0.05):
    # print(f"Shapes of reference and in-test populations: {human_score_distribution.shape}, {modified_scores.shape}")
    min_required_diff = get_minimum_difference(human_score_distribution.shape[0] - 1, \
        human_score_distribution.mean(), human_score_distribution.std(), sig_level=significance_value)
    
    score_changes = np.abs(human_score_distribution - modified_scores)
    sig_change = score_changes > min_required_diff
    # pvals = stats.ttest_1samp(human_score_distribution, modified_scores).pvalue
    # sig_change = pvals < significance_value
    non_sig_changes = np.logical_not(sig_change)
    if np.sum(non_sig_changes) != 0:
        max_insig_pv = score_changes[non_sig_changes].max()
    else:
        max_insig_pv = 0
    # return sig_change, abs(human_score_distribution[max_insig_pv_ix] - modified_scores[max_insig_pv_ix])
    return sig_change, max_insig_pv
    
def get_average(scores):
    avg_score = 0
    for _, score in scores.items():
        avg_score += score
    return avg_score/len(scores)

def compare_conversations(conv1, conv2, scorer):
    #compute score(conv1) < score(conv2)
    conv_to_score = [conv1, conv2]
    # for i, conv in enumerate(conv_to_score):
        # print(f"Conversation Number: {i}")
    # print(conv1)
    # print("########################")
    scores = scorer.get_scores(scorer.format_conversations(conv_to_score))
    # print("scores achieved:", scores)
    return scores

def check_length(conv_file, min_length):
    with open(conv_file) as cfile:
        cdata = cfile.read()
        mod_conv = cdata.split("modified_conversation")[1].strip().split('\n')
        orgnl_conv = cdata.split("modified_conversation")[0].strip().split('\n')[1:]
        return len(orgnl_conv) != len(mod_conv) or len(mod_conv) < min_length

def main(args, scorer, sampler=None):
    correct_predictions = 0
    total_conv = 0
    human_scores = []
    modified_scores = []
    if args.specific_test_case:
        file_list = ['case_{}'.format(args.specific_test_case)]
    else:
        file_list = list(filter(lambda x: x not in ['.', ".."] and not x.startswith("prob_cases"), os.listdir(args.conversation_dir)))
    
    file_list = list(filter(lambda x: not check_length(os.path.join(args.conversation_dir, x), args.min_conv_length), file_list))
    chosen_files = random.sample(file_list, k=min(args.num_examples, len(file_list)))
    print(f"Number of chosen files: {len(chosen_files)}")
    file_progress_bar = tqdm(chosen_files)
    for filename in file_progress_bar:
        file_progress_bar.set_description(f"{filename}")
        # print('Case under consideration: {0}'.format(filename))
        conversations = {}
        conversation_file = open(os.path.join(args.conversation_dir, filename))
        cur_key = conversation_file.readline().strip()
        conversations[cur_key] = []
        for line in conversation_file:
            line = line.strip()
            if line in ['original_conversation', 'modified_conversation']:
                cur_key = line
                continue
                
            conversations.setdefault(cur_key, [])
            conversations[cur_key].append(line)
        orgnl_score, modfy_score = compare_conversations('\n'.join(conversations['original_conversation']).strip(), \
            '\n'.join(conversations['modified_conversation']).strip(), scorer)
        # if "entailment" in args.conversation_dir:
        #     print('\n'.join(conversations["original_conversation"]))
        #     print("------------------------------------")
        #     print('\n'.join(conversations["modified_conversation"]))
        #     print("####################################")
        human_scores.append(orgnl_score)
        modified_scores.append(modfy_score)
        total_conv += 1

    human_scores = np.array(human_scores)
    if sampler:
        reference_scores = []
        for _ in range(args.sampler_examples):
            conv = sampler.sample().strip()
            reference_scores.append(scorer.get_scores(scorer.format_conversations([conv]))[0])
        reference_scores = np.array(reference_scores)
    else:
        reference_scores = human_scores
    modified_scores = np.array(modified_scores)
    sig_change, max_insig_change = check_significance(reference_scores, modified_scores, significance_value=args.significance_level)
    order = np.zeros(sig_change.shape)
    if  "g" in args.comparison_type:
        order[np.logical_and(human_scores > modified_scores, sig_change)] = 1
    if "l" in args.comparison_type:
        order[np.logical_and(human_scores < modified_scores, sig_change)] = 1
    if "e" in args.comparison_type:
        order[np.logical_not(sig_change)] = 1
    
    # if args.comparison_type == "equal":
    correct_predictions = order.sum()
    # else:
    #     correct_predictions = order[sig_change].sum()
    print("Mean of the reference population: {}".format(reference_scores.mean()))
    print("Standard Deviation of reference population: {}".format(reference_scores.std()))
    print("Maximum insignificant change observed: {}".format(max_insig_change))
    print("Proportion of significant score changes: {}".format(sig_change.sum()/total_conv))
    print('The accuracy of this prediction is: {}'.format(correct_predictions/total_conv))
    print("The error rate is: {}".format(1 - (correct_predictions/total_conv)))
    print("Saving the problematic cases at:", os.path.join(args.conversation_dir, f"prob_cases_{args.scorer_name}"))

    prob_cf = open(os.path.join(args.conversation_dir, f"prob_cases_{args.scorer_name}"), 'w')
    for i, fname in enumerate(chosen_files):
        if order[i] == 0:
            # if human_scores[i] > modified_scores[i]:
            #     print(sig_change[i], (human_scores > modified_scores)[i], human_scores[i], modified_scores[i])
            prob_cf.write(f"{fname} {human_scores[i]} {modified_scores[i]}\n")
    prob_cf.close()
    
    if args.specific_test_case:
        print ('{:<25} {:<15} {:<10}'.format('','Original','Modified'))
        print ('{:<25} {:<15} {:<10}'.format('',human_scores[0],modified_scores[0]))
        print ('{:<25} {:<15} {:<10}'.format("p-value < 0.05", sig_change[0],''))
        # for attribute_name in scores_by_attributes[0].keys():
        #     print('{:<25} {:<15} {:<10}'.format(attribute_name, \
        #             round(scores_by_attributes[0][attribute_name], 2), round(scores_by_attributes[1][attribute_name], 2)))
        # print('{:<25} {:<15} {:<10}'.format('Average', round(get_average(scores_by_attributes[0]), 3), \
        #                                         round(get_average(scores_by_attributes[1]), 3)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conversation_dir', help='Specify the directory containing the conversation files')
    parser.add_argument('scorer_path', help="path to the module containing the 'conversation_scorer' class.")
    # parser.add_argument('sampler_path', help="path to the file containing the class 'conversation_sampler'.")
    parser.add_argument('--comparison_type', \
        help="specify how the scores for human conversations need to compared with the adversarial ones' scores. Default is g for greater.", \
        default="g", choices=["g", "l", "e", "ge", "le"])
    parser.add_argument('-s', '--specific_test_case', \
                        help='Specify the specific test case to run the model for. Otherwise the model will be run for all the conversations in the specified directory.')
    parser.add_argument("--sampler", help="'conversation_sampler' class definition to measure the score distribution for score significance.")
    parser.add_argument("-e", "--sampler_examples", help="Number of conversation to computer standard dev for human conversations.", \
                        type=int, default=100)
    parser.add_argument("-n", "--num_examples", help="Maximum number of adversarial examples.", type=int, default=100)
    parser.add_argument("--significance_level", help="Significance level for t-test. Default is 0.05.", type=float, default=0.05)
    parser.add_argument("--score_subdirs", help="This flag determines if the passed conversation directories is a parent directory for the adversarial attacks.", \
                        action="store_true", default=False)
    parser.add_argument("--metric_name", help="Specify the name/type for a sub-metric if any.", default=None)
    parser.add_argument("--scorer_name", help="The scorer name to distinguish problematic cases between metrics.")
    parser.add_argument("--specify_cases", nargs="+", help="specify multiple cases to run the metric on.")
    parser.add_argument("--min_conv_length", help="Filter any conversations having length less than this.", type=int, default=2)
    args = parser.parse_args()

    def get_module(name, path):
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    score = get_module("score", args.scorer_path)
    sampler = None
    if args.sampler:
        sampler = get_module("sample", args.sampler).conversation_sampler()
    if args.metric_name == None:
        scorer = score.conversation_scorer(contains_response=True, use_response_lengths=False)
    else:
        scorer = score.conversation_scorer(contains_response=True, metric_name=args.metric_name, use_response_lengths=False)
    sub_dirs = list(filter(lambda x: x not in ['.', ".."], os.listdir(args.conversation_dir)))
    if args.specify_cases:
        sub_dirs = args.specify_cases
    # exit()
    args.scorer_name = args.scorer_path.split('/')[-2] if len(args.scorer_path.split('/')) > 1 and args.scorer_name == None else args.scorer_name
    if args.score_subdirs or args.specify_cases:
        root_dir = args.conversation_dir
        for subdir in sub_dirs:
            if subdir in ["ner_neutral", "good_paraphrase"]:
                args.comparison_type = 'e'
            elif subdir in ["diversity"]:
                args.comparison_type = "ge"
            else:
                args.comparison_type = 'g'
            args.conversation_dir = os.path.join(root_dir, subdir)
            print(f"Solving the case: {subdir}")
            print(f"The conversation directory is: {args.conversation_dir} with comparison type: {args.comparison_type}")
            print(f"The problematic cases file name would be: prob_cases_{args.scorer_name}")
            main(args, scorer, sampler)
            print("---------------------------")
    else:
        main(args, scorer, sampler)