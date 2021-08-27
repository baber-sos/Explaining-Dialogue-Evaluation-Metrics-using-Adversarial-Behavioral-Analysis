import json
import random
import numpy as np

def init_conv_score(categories):
    conv_score = {category: 0 for category in categories}
    return conv_score

class conversation_sampler:
    def __init__(self):
        conversation_data = json.load(open('/home/ubuntu/fed/fed_data.json'))
        self.candidate_conversations = []
        self.categories = ['interesting', 'engaging', 'specific', 'relevant', 'correct', \
                'semantically appropriate', 'understandable', 'fluent', 'coherent', 'error recovery', \
                'consistent', 'diverse', 'depth', 'likeable', 'understandable', 'flexible', 'informative', 'inquisitive']
        self.candidate_scores = []

        conv_score = init_conv_score(self.categories)
        conv_len = 0
        for conversation in conversation_data:
            if 'response' not in conversation and conversation['system'] == 'Human':
                self.candidate_conversations.append((conversation['context']))
                for key, values in conversation['annotations'].items():
                    key = key.lower()
                    values = list(filter(lambda x: type(x) == int, values))
                    if len(values) == 0:
                        values.append(0)
                    if key in self.categories:
                        conv_score[key] = np.average(values)
                self.candidate_scores.append(conv_score)
                # print(conv_score)
                # print('------------')
                conv_score = init_conv_score(self.categories)
                conv_len = 0
            else:
                for key, values in conversation['annotations'].items():
                    key = key.lower()
                    values = list(filter(lambda x: type(x) == int, values))
                    if len(values) == 0:
                        values.append(0)
                    if key in self.categories:
                        conv_score[key] = ((conv_score[key] * conv_len) + np.average(values))/(conv_len + 1)
                conv_len += 1
                # print(conv_score, key, values)
        
        print('Number of candidate conversations: {}'.format(len(self.candidate_conversations)))

    def __format_conversation(self, conversation):
        separator_token = '<|endoftext|>'
        utterances = [separator_token,]
        for utterance in conversation.split('\n'):
            utterances.append(f'{utterance.strip()} {separator_token}')
        return ' '.join(utterances)
    
    def sample(self):
        return random.choice(self.candidate_conversations)
    
    def get_next(self):
        for conversation in self.candidate_conversations:
            yield conversation
    
    def randomize(self):
        random.shuffle(self.candidate_conversations)

    def __len__(self):
        return len(self.candidate_conversations)

    def get_next_with_scores(self):
        for i, conversation in enumerate(self.candidate_conversations):
            yield (conversation, self.candidate_scores[i])