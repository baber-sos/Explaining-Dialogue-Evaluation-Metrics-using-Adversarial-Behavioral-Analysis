import fed
import shap
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

class DialoGPTMetric(torch.nn.Module):
    def __init__(self):
        super(DialoGPTMetric, self).__init__()
        self.dgpt_model = AutoModelWithLMHead.from_pretrained('microsoft/DialoGPT-large')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')


    def forward(self, input_strings):
        # tokenized_inputs = [self.tokenizer.tokenize(text) for text in input_strings]
        # tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_input) for tokenized_input in tokenized_inputs]).cuda()
        # outputs = model(tensor_input, labels=tensor_input)
        # loss, logits = outputs[:2]
        # return calculcate_score(loss.item())
        categorical_scores = [fed.evaluate(conv, self.dgpt_model, self.tokenizer) for conv in input_strings]
        to_return = torch.tensor([cscore['coherent'] for cscore in categorical_scores])
        return to_return

class conversation_scorer:
    def __init__(self):
        self._model, self._tokenizer = fed.load_models('microsoft/DialoGPT-large')
        self._tokenizer.max_model_input_size = 1500
        self.masker = shap.maskers.Text(self._tokenizer, mask_token='<|endoftext|>')
        self.categories = ['interesting', 'engaging', 'specific', 'relevant', 'correct', \
                'semantically appropriate', 'understandable', 'fluent', 'coherent', 'error recovery', \
                'consistent', 'diverse', 'depth', 'likeable', 'understand', 'flexible', 'informative', 'inquisitive']
        self.investigative_scores = []
        self.aggregate = True
    
    def _get_average(self, scores):
        avg_score = 0.0
        count = 0
        for _, score in scores.items():
            avg_score += score
            count += 1
        return avg_score/count
    
    def get_categories(self):
        if self.aggregate:
            return None
        return self.categories

    def get_masker(self):
        return self.masker

    def set_aggregate(self, aggregate):
        self.aggregate = aggregate
    
    def format_conversations(self, conversations):
        return [fed.format_conversation(conv).strip() for conv in conversations]

    def score_by_attributes(self, scores):
        # return np.array([scores[key] for key in self.categories])
        # return np.array([scores[key] for key in self.categories['understand']])
        # return np.array([scores['coherent']])
        return scores['coherent']

    def get_scores(self, conversations):
        # print(conversations.shape)
        # print('Number of tokens in the masked conversation:', len(self._tokenizer.tokenize(conversations[0])))
        # print(conversations[0])
        # conversations = [self._tokenizer.convert_ids_to_tokens(conv) for conv in conversations]
        # conversations = [' '.join(conv) for conv in conversations]
        # print('Input by the explainer:', conversations)
        print('Number of conversations:', conversations.shape[0])
        print('The input conversations:', conversations[0])
        if len(conversations) > 1:
            print('The input conversations:', conversations[1])
        categorical_scores = [fed.evaluate(form_conv[0], self._model, self._tokenizer) for form_conv in conversations]
        if not self.aggregate:
            to_return = np.array([self.score_by_attributes(cscore) for cscore in categorical_scores])
            # return np.array([self.score_by_attributes(cscore) for cscore in categorical_scores])
        else:
            to_return = np.array(list(map(self._get_average, categorical_scores)))
        # self.investigative_scores.append(zip(conversations, to_return))
        return to_return
