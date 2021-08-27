import random
import importlib.util

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import OrderedDict

def replicate_ordered_dict(ord_dict):
    to_return = OrderedDict()
    for key in ord_dict.keys():
        new_key = key[7:] if key.startswith("module.") else key
        to_return[new_key] = ord_dict[key]
    return to_return

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model.to("cuda:1")
# model = torch.nn.DataParallel(model, device_ids=[1])
# print(model.state_dict().keys())
saved_state = replicate_ordered_dict(torch.load("t5_contradiction_gen.pt"))
# print(saved_state.keys())
model.load_state_dict(saved_state)

# exit()
sample_spec = importlib.util.spec_from_file_location("sample", "/home/ubuntu/dialogue_evaluation/sample.py")
sample = importlib.util.module_from_spec(sample_spec)
sample_spec.loader.exec_module(sample)

sampler = sample.conversation_sampler()

count = 20
with torch.no_grad():
    for i in range(count):
        conv = sampler.sample().strip()
        utt = random.choice(conv.split('\n'))
        print("Original Utterance: {}".format(utt))
        text =  utt + " </s>"
        
        encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to("cuda:1"), encoding["attention_mask"].to("cuda:1")

        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=1024,
            do_sample=True,
            top_k=256,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
        )
        print("Candidate Contradictions:")
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print("-------------------------")
