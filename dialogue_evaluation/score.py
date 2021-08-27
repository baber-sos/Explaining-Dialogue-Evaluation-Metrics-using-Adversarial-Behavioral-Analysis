import numpy as np
import sys
from transformers import AutoTokenizer
import shap
sys.path.insert(0, "/home/ubuntu/dialogue_evaluation")
from metrics_evaluation import *

class conversation_scorer:
    def __init__(self, **kwargs):
        print(kwargs)
        self.pretrained_model_path = 'gpt2'
        self.metric_name = 'context' if not kwargs.get('metric_name', None) else kwargs['metric_name']
        self.possible_metrics = ['context', 'fluency', 'diversity', 'logic']
        print(f"The metric name is: {self.metric_name}")
        if self.metric_name not in self.possible_metrics:
            print('Please enter one of the following metrics: {}, {}, {} or {}'. format(*self.possible_metrics))
            raise("Wrong metric name specified.")
        self.investigative_scores = []
        # self._tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_path)
        # self._tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.masker = shap.maskers.Text(self._tokenizer, mask_token=kwargs.get('mask_token', "<|endoftext|>"))
        self.aggregate = False
        self.split_token = kwargs.get('split_token', '\n')
        self.ngrams = kwargs.get('ngrams', 2)
        self.contexts = []
        self.contains_response = kwargs.get("contains_response", False)
        self.use_response_lengths = kwargs.get("use_response_lengths", True)
    
    def format_conversations(self, conversations):
        return conversations
        # contexts, follow_ups = self.split_last_utterances(conversations)
        # return contexts, follow_ups

    def get_masker(self):
        return self.masker
    
    def get_tokenizer(self):
        return self._tokenizer

    def set_aggregate(self, aggregate):
        self.aggregate = False

    def split_last_utterances(self, conversations):
        contexts = []
        follow_ups = []
        if not self.contains_response:
            for i, conv in enumerate(conversations):
                contexts.append(self.contexts[i % len(self.contexts)])
                follow_ups.append(conv.strip())
            return contexts, follow_ups

        if not self.use_response_lengths:
            for conv in conversations:
                utterances = conv.strip().split(self.split_token)
                # print(utterances)
                contexts.append(self.split_token.join(utterances[:-1]))
                follow_ups.append(utterances[-1])
        else:
            tokenized_convs = [self._tokenizer.encode(conv) for conv in conversations]
            for i, tokens in enumerate(tokenized_convs):
                resp_len = self.response_lengths[i % len(self.response_lengths)]
                contexts.append(self._tokenizer.decode(tokens[:-resp_len]))
                follow_ups.append(self._tokenizer.decode(tokens[-resp_len:]))
        return contexts, follow_ups

    def set_response_lengths(self, response_lengths):
        self.response_lengths = response_lengths

    def set_contexts(self, ctxts):
        self.contexts = ctxts

    def get_scores(self, conversations, **kwargs):
        # print('Number of conversations:', len(conversations))
        # print('The input conversations:', conversations[0])
        # if len(conversations) > 1:
        #     print('The input conversations:', conversations[1])
        context, follow_up = self.split_last_utterances(conversations)
        # print(follow_up[0])
        if self.metric_name == 'context':
            # context, follow_up = self.split_last_utterances(conversations)
            # for i, ctxt in enumerate(context):
            #     print("Context:", ctxt)
            #     print("Follow Up:", follow_up[i])
            #     print('------------------------')
            scores = context_score(context, follow_up, self.pretrained_model_path)
            # print('These are the given scores:', scores[:10])
            return np.array(scores)
        elif self.metric_name == 'fluency':
            return fluency_score(follow_up, self.pretrained_model_path)
        elif self.metric_name == 'diversity':
            return diversity_score(follow_up, self.ngrams)
        elif self.metric_name == 'logic':
            # context, follow_up = self.split_last_utterances(conversations)
            return logic_consistency(context, follow_up)
