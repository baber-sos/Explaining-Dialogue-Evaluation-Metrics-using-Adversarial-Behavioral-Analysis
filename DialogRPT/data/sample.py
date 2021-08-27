import os
import random

class conversation_sampler:
    def __init__(self, dataset_type="all"):
        if dataset_type in ["all", "test"]:
            files_path = "/home/ubuntu/DialogRPT/data/test/human_feedback"
            datafiles = ["depth.tsv", "updown.tsv", "width.tsv"]
        elif dataset_type == "twitter":
            files_path = "/home/ubuntu/DialogRPT/data/test/human_vs_rand"
            datafiles = ["ref.tsv"]
        
        self.candidate_conversations = []
        for dfile in datafiles:
            with open(os.path.join(files_path, dfile)) as conv_file:
                for line in conv_file:
                    columns = line.split('\t')
                    context = [utt.strip() for utt in columns[0].split("_EOS_")]
                    resp = columns[1]
                    self.candidate_conversations.append('\n'.join(context + [resp]))
    
    def sample(self):
        return random.choice(self.candidate_conversations)
    
    def randomize(self):
        random.shuffle(self.candidate_conversations)

    def __len__(self):
        return len(self.candidate_conversations)

    def get_next(self):
        for conv in self.candidate_conversations:
            yield conv