from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import shap

class conversation_scorer:
    def __init__(self, **kwargs):
        model_prefix = "microsoft/DialogRPT-"
        model_cards = ["updown", "width", "depth", "human-vs-rand"]
        self.possible_metrics = ["overall", "width", "depth", "updown", "human-vs-rand"]
        self.metric = kwargs.get("metric_name", "overall")
        if self.metric not in self.possible_metrics:
            print(f"Please enter one of the following metrics: {self.possible_metrics}")
        self.gpu_count = torch.cuda.device_count()
        self.models = {card: AutoModelForSequenceClassification.from_pretrained(model_prefix + card) for card in model_cards}
        self.model_to_device = dict()
        for i, card in enumerate(model_cards):
            print(f"Loading Model: {card}")
            self.model_to_device[card] = f"cuda:{i % self.gpu_count}"
            self.models[card].to(f"cuda:{i % self.gpu_count}")

        self.tokenizers = {card: AutoTokenizer.from_pretrained(model_prefix + card) for card in model_cards}
        self.contexts = []
        self.weights = {"width" : -0.5, "depth": 0.48, "updown": 1.0}
        self.contains_response = kwargs.get('contains_response', False)
        self.mask_token = kwargs.get("mask_token", "<|endoftext|>")
    
    def get_masker(self):
        return shap.maskers.Text(self.tokenizers["depth"], mask_token=self.mask_token)

    def get_tokenizer(self):
        return self.tokenizers["depth"]
    
    def get_full_conversation(self, ix, response):
        if self.contains_response:
            utterances = response.split('\n')
            return '\n'.join(utterances[:-1]) + "<|endoftext|>" + utterances[-1]
        return self.contexts[ix % len(self.contexts)] + "<|endoftext|>" + response
    
    def set_contexts(self, ctxts):
        self.contexts = ctxts

    def format_conversations(self, conversations):
        return conversations
    
    def score(self, conversation, card):
        model_input = self.tokenizers['depth'].encode(conversation, return_tensors="pt").to("cuda")
        result = self.models[card](model_input.to(self.model_to_device[card]), return_dict=True)
        return torch.sigmoid(result.logits)

    def get_scores(self, to_score):
        # print(f"The number of inputs: {len(to_score)}")
        # print(f"The first input: {to_score[0]}")
        to_return = []
        for i in range(len(to_score)):
            conversation = self.get_full_conversation(i, to_score[i])
            # print("This is the input conversation:", conversation)
            # print("------------")
            if self.metric == "overall":
                final_score = 0
                model_cards = ["width", "depth", "updown", "human-vs-rand"]
                scores = {}
                for card in model_cards:
                    scores[card] = self.score(conversation, card).squeeze()
                for m, w in self.weights.items():
                    final_score += w * float(scores[m].detach().to("cpu"))
                    # print(final_score, scores[m])
                final_score *= float(scores["human-vs-rand"].detach().to("cpu"))
                to_return.append(final_score)
            else:
                to_return.append(self.score(conversation, self.metric))
        return np.array(to_return)