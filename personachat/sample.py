from parlai.tasks.personachat.agents import *
import random

class conversation_sampler:
    def __init__(self, dataset_type="all"):
        self.agent = BothRevisedTeacher({"datapath": "/home/ubuntu/personachat/personachat", "datatype" : "test:persona"})
        self.candidate_covnersations = []

        for _ in range(self.agent.num_episodes()):
            conv = self.get_conversation_from_agent()
            self.candidate_covnersations.append(conv)

    def get_conversation_from_agent(self):
        cur_example = self.agent.next_example()
        next_pair = [cur_example[0]["text"].split('\n')[-1], cur_example[0]["labels"][0]]
        utterances = []
        while not cur_example[0]["episode_done"]:
            utterances.extend(next_pair)
            cur_example = self.agent.next_example()
            next_pair = [cur_example[0]["text"].split('\n')[-1], cur_example[0]["labels"][0]]
        utterances.extend(next_pair)
        return '\n'.join(utterances)
    
    def __len__(self):
        return len(self.candidate_covnersations)

    def randomize(self):
        random.shuffle(self.candidate_covnersations)
    
    def get_next(self):
        for conv in self.candidate_covnersations:
            yield conv
    
    def sample(self):
        return random.choice(self.candidate_covnersations)
