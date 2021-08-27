from transformers import AutoTokenizer
import torch
import numpy as np
import shap
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/ubuntu/user-satisfaction-simulation")
from baselines.models import GRUAttention, BERTBackbone
from baselines.models import HierarchicalAttention, ClassModel

class conversation_scorer:
    def __init__(self, **kwargs):
        self.possible_metrics = ["hi-gru", "bert", "bertw", "hi-gruw"]
        self.metric = kwargs.get("metric_name", "bert")
        if self.metric not in self.possible_metrics:
            print(f"Please enter one of the following metrics: {self.possible_metrics}")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if "bert" in self.metric:
            self.model = ClassModel(backbone=BERTBackbone(layers_used=2, name='bert-base-uncased'), class_num=[5])    
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
            self.model.load_state_dict(torch.load("/home/ubuntu/user-satisfaction-simulation/baselines/outputs/MWOZ_BERT_0_33.pt"))
        else:
            self.model = HierarchicalAttention(backbone=GRUAttention(vocab_size=self.tokenizer.vocab_size), class_num=[5])
            self.model.cuda()
            self.model.load_state_dict(torch.load("/home/ubuntu/user-satisfaction-simulation/baselines/outputs/MWOZ_HiGRU+ATTN_0_99.pt"))
        self.model.eval()

        self.contexts = []
        self.contains_response = kwargs.get("contains_response", False)
        self.mask_token = kwargs.get("mask_token", "[MASK]")
        self.wavg = True if self.metric[-1] == 'w' else False
        print(f"The metric is: {self.metric}. Score will be calculated as a weighted average: {self.wavg}")
        self.dialogues_used = 10
        self.bc_size = 16
    
    def get_masker(self):
        return shap.maskers.Text(self.tokenizer, mask_token=self.mask_token)

    def get_tokenizer(self):
        return self.tokenizer
    
    def encode_conversation(self, conversation):
        utterances = conversation.split('\n')
        if "bert" in self.metric:
            encoded_rep = sum([self.tokenizer.encode(utt)[:64] for utt in utterances], [])
            return torch.tensor([101] + encoded_rep[-500:])
        else:
            encoded_utt = [torch.tensor([101] + self.tokenizer.encode(utt)[:64]) for utt in utterances]
            utt_padding = [torch.tensor([101])] * (self.dialogues_used - len(utterances))
            unpadded_encoding = utt_padding + encoded_utt[-self.dialogues_used:]
            return unpadded_encoding
            # return pad_sequence(unpadded_encoding, batch_first=False, padding_value=0)
            # return torch.tensor(pad_sequence(unpadded_encoding, batch_first=False, padding_value=0)).view(1, len(unpadded_encoding), -1)

    def encode_multiple(self, conversations):
        enc_convs = [self.encode_conversation(conv) for conv in conversations]
        bc_size = len(enc_convs)
        if "bert" in self.metric:
            return pad_sequence(enc_convs, batch_first=True, padding_value=0).view(bc_size, -1).cuda()
        else:
            enc_utts = [utt for dialog in enc_convs for utt in dialog]
            return pad_sequence(enc_utts, batch_first=True, padding_value=0).view(bc_size, self.dialogues_used, -1).cuda()

    def get_full_conversation(self, ix, response):
        if self.contains_response:
            return response
        return self.contexts[ix % len(self.contexts)] + "\n" + response
    
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
        # for conv in to_score:
        #     print(conv)
        #     print("-------------------------------------")
        # print(f"The first input: {to_score[0]}")
        conversations = [self.get_full_conversation(i, to_score[i]) for i in range(len(to_score))]
        encoded_convs = self.encode_multiple(conversations)
        to_return = []
        with torch.no_grad():
            for i in range(0, len(to_score), self.bc_size):
                # print(f"Inputs: {encoded_convs.shape}")
                scores, *o = self.model(input_ids=encoded_convs[i : i+self.bc_size])
                scores = scores.view(-1, 5)
                # print(f"The assigned logits: {scores}")s
                scores = F.softmax(scores, dim=1)
                # print(f"The assigned probabilities: {scores}")         
                # print("##################################")
                if self.wavg:
                    for score_vec in scores:
                        # print(score_vec)
                        to_return.append(np.sum([float(px * (pix + 1)) for (pix, px) in enumerate(score_vec)]))
                else:
                    indices = scores.argmax(dim=1)
                    to_return.extend([float(x) for x in indices])
        return np.array(to_return)