import json
import random
import os


class conversation_sampler:
    def __init__(self, dataset_type="all"):
        self.data_folder = "/home/ubuntu/multiwoz/data/MultiWOZ_2.2/test"
        dfnames = ["dialogues_001.json", "dialogues_002.json"]
        
        self.candidate_covnersations = []
        for dfname in dfnames:
            with open(os.path.join(self.data_folder, dfname)) as dfile:
                conv_data = json.load(dfile)
                for dinst in conv_data:
                    conv = []
                    for turn in dinst["turns"]:
                        conv.append(turn["utterance"])
                    self.candidate_covnersations.append('\n'.join(conv))
    
    def __len__(self):
        return len(self.candidate_covnersations)

    def randomize(self):
        random.shuffle(self.candidate_covnersations)
    
    def get_next(self):
        for conv in self.candidate_covnersations:
            yield conv
    
    def sample(self):
        return random.choice(self.candidate_covnersations)
