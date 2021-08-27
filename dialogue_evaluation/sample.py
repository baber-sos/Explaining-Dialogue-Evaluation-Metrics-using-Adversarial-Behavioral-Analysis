#dialy dialogue sampler
import random

class conversation_sampler:
    def __init__(self, dataset_type="all"):
        self.dialogues = []
        if dataset_type == "all":
            with open('/home/ubuntu/dialogue_evaluation/ijcnlp_dailydialog/dialogues_text.txt') as dfile:
                for dialogue in dfile:
                    self.dialogues.append('\n'.join([x.strip() for x in dialogue.split('__eou__')]))
        elif dataset_type == "test":
            print("Building sampler for test data.")
            with open("/home/ubuntu/dialogue_evaluation/ijcnlp_dailydialog/test/dialogues_test.txt") as dfile:
                for dialogue in dfile:
                    self.dialogues.append('\n'.join([x.strip() for x in dialogue.split('__eou__')]))

    def sample(self):
        return random.choice(self.dialogues)
    
    def get_next(self):
        for conversation in self.dialogues:
            yield conversation
    
    def get_next_random(self):
        to_use = list(self.dialogues)
        random.shuffle(to_use)
        for conversation in to_use:
            yield conversation
    
    def randomize(self):
        random.shuffle(self.dialogues)
    
    def __len__(self):
        return len(self.dialogues)