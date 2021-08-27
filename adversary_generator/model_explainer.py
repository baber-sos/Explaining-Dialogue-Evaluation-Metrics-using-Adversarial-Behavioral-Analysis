import argparse
import sys
import shap

def main(args, sampler, scorer):
    explainer = shap.Explainer(scorer.get_scores, scorer.get_masker())
    if args.randomize_data:
        sampler.randomize()
    conv_to_explain = []
    for i, conversation in enumerate(sampler.get_next()):
        conv_to_explain.append(conversation)
        if not args.all and (i + 1) == args.num_examples:
            break
    print(f'Number of conversation: {len(conv_to_explain)}')
    return explainer(conv_to_explain)

if __name__ == '__main__':
    #this file would take a scoring function as input and generate an explainability plot using
    #the SHAP library. For now, the explainability will focus on the importance of tokens when
    #correlating to model outputs. Have to think of ways this can be extended to detect phrases
    #The scoring function would also provide a tokenizer and a masking token which would serve 
    #as input to the explainer.
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler_path', default='/home/ubuntu/fed/', \
                        help='path to a directory containing a python module by the name of "sample.py". It should have a class definition "conversation_sampler".')
    parser.add_argument('--scorer_path', default='/home/ubuntu/fed/', \
                        help='path to a directory containing a python module by the name of "score.py". It should have a class definition "conversation_scorer".')
    parser.add_argument('-n', '--num_examples', help='number of mutated examples to generate', type=int, default=10)
    parser.add_argument('--all', help='use all data examples. Overrides n.', action='store_true')
    parser.add_argument('-r', '--randomize_data', action='store_true')
    parser.add_argument('--e', '-explainer_level', help='specify at which level do you want this module to explain the decisions', \
                        choices=['vocab'], default='vocab')
    parser.add_argument('-k', '--top_k', help='Top k elements for different statisctics. Default: 10', \
                        type=int, default=10)
    
    args = parser.parse_args()  
    print(args)
    sys.path.insert(0, args.scorer_path)
    if args.sampler_path != args.scorer_path:
        sys.path.insert(0, args.sampler_path)

    from sample import conversation_sampler
    from score import conversation_scorer

    sampler = conversation_sampler()
    scorer = conversation_scorer()
    shap_values = main(args, sampler, scorer)

