import os
import random

class adversary_sampler:
    def __init__(self, path, relation="l"):
        self.cur_ix = 0
        self.candidate_conversations = []
        self.relation = relation
        self.og_scores = []
        self.conversation_files, self.score_diffs = self.get_file_list(path)
        
        for cfile in self.conversation_files:
            adv_conv = open(os.path.join(path, cfile))
            self.candidate_conversations.extend(list(self.get_conversations(adv_conv)))
            adv_conv.close()

    def get_file_list(self, path):
        score_dict = dict()
        og_scores = dict()
        with open(os.path.join(path, "prob_cases")) as pcase_file:
            for line in pcase_file:
                name, og_score, mod_score = line.strip().split()
                og_score = float(og_score)
                mod_score = float(mod_score)
                score_dict[name.strip()] = (mod_score - og_score)
                if self.relation == "g":
                    score_dict[name.strip()] = -1 * (og_score - mod_score)
                elif self.relation == "e":
                    score_dict[name.strip()] = abs(og_score - mod_score)
                og_scores[name] = (og_score, mod_score)

        files = list(score_dict.keys())
        files.sort(key=lambda x: score_dict[x], reverse=True)
        self.og_scores = [og_scores[x] for x in files]
        return files, [score_dict[x] for x in files]

    def get_next_random(self):
        random_list = list(self.candidate_conversations)
        random.shuffle(random_list)
        for i, conv in enumerate(random_list):
            if i % 2 == 1:
                continue
            yield conv

    def get_conversations(self, cfd):
        orgnl_conv, mod_conv = cfd.read().strip().split("modified_conversation")
        return orgnl_conv.split("original_conversation")[1].strip(), mod_conv.strip()

    def sample(self):
        self.cur_ix = (self.cur_ix + 1) % self.candidate_conversations
        return self.candidate_conversations[self.cur_ix - 1 if self.cur_ix != 0 else len(self.candidate_conversations) - 1]

    def get_next(self):
        for conv in self.candidate_conversations:
            yield conv

    def randomize(self):
        pass

    def __len__(self):
        return len(self.candidate_conversations)