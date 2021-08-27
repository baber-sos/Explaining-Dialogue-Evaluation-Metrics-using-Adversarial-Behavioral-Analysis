import sys
import os
import fed
import argparse

def get_average(scores):
    avg_score = 0
    for _, score in scores.items():
        avg_score += score
    return avg_score/len(scores)

def compare_conversations(conv1, conv2, model, tokenizer):
    #compute score(conv1) < score(conv2)
    separator_token = '<|endoftext|>'
    average_scores = []
    scores_by_attributes = []
    for conversation in [conv1, conv2]:
        current_conversation = f'{separator_token} '
        for i, line in enumerate(conversation):
            if i == len(conversation) - 1:
                current_conversation += line.strip().split(':')[1].strip()
            else:
                current_conversation += line.strip().split(':')[1].strip() +  f' {separator_token} '
        print(current_conversation)
        scores = fed.evaluate(current_conversation.strip(), 
                      model, 
                      tokenizer)
        scores_by_attributes.append(scores)
        average_scores.append(get_average(scores))
        print(round(average_scores[-1], 3))
    print('----')
    return average_scores[0] > average_scores[1], scores_by_attributes

def main(args, model, tokenizer):
    correct_predictions = 0
    count = 0
    if args.specific_test_case:
        file_list = ['case_{}'.format(args.specific_test_case)]
    else:
        file_list = os.listdir(args.conversation_dir)
    for filename in file_list:
        print('Case under consideration: {0}'.format(filename))
        conversations = {}
        conversation_file = open(os.path.join(args.conversation_dir, filename))
        cur_key = conversation_file.readline().strip()
        conversations[cur_key] = []
        for line in conversation_file:
            line = line.strip()
            if line in ['original_conversation', 'modified_conversation']:
                cur_key = line
                conversations[cur_key] = []
                continue
            conversations[cur_key].append(line)
        result, scores_by_attributes = compare_conversations(conversations['original_conversation'], \
                                        conversations['modified_conversation'], model, \
                                        tokenizer)
        count += 1
        correct_predictions += result
    print('The accuracy of this prediction is: {}'.format(correct_predictions/count))
    if args.specific_test_case:
        print ('{:<25} {:<15} {:<10}'.format('','Original','Modified'))
        for attribute_name in scores_by_attributes[0].keys():
            print('{:<25} {:<15} {:<10}'.format(attribute_name, \
                    round(scores_by_attributes[0][attribute_name], 2), round(scores_by_attributes[1][attribute_name], 2)))
        print('{:<25} {:<15} {:<10}'.format('Average', round(get_average(scores_by_attributes[0]), 3), \
                                                round(get_average(scores_by_attributes[1]), 3)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conversation_dir', help='Specify the directory containing the conversation files')
    parser.add_argument('-s', '--specific_test_case', \
                        help='Specify the specific test case to run the model for. Otherwise the model will be run for all the conversations in the specified directory.')
    args = parser.parse_args()

    model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
    main(args, model, tokenizer)