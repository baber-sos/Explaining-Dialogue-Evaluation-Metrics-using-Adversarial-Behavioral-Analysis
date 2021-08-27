#%%
import sys
import copy
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import shap
import scipy as sp
from datasets import load_dataset
import torch
import fed

model, tokenizer = fed.load_models("microsoft/DialoGPT-large")

def get_average(scores):
    avg_score = 0
    for _, score in scores.items():
        avg_score += score
    return avg_score/len(scores)

def f(inputs):
    # print(inputs.shape)
    scores = [fed.evaluate(x, model, tokenizer) for x in inputs]
    to_return = np.array(list(map(get_average, scores)))
    # print(to_return.shape)
    return to_return

explainer = shap.Explainer(f, shap.maskers.Text(tokenizer, mask_token='<|endoftext|>'))
# print(explainer)
conversation_to_test = "User: Hi!\nSystem: Hi, how are you?\nUser: I'm doing great. what are you up to?\nUser: It's good! Just went to the park to enjoy the sun. You?\nUser: oh, I'm enjoying a sip of coffee, looking over the ocean. what book are you reading?\nSystem: I'm reading Why We Sleep It's a book by a sleep scientist about the benefits of sleeping/cons of not sleeping enough I'm trying to scare myself into sleeping more haha"
formatted_conversation = fed.format_conversation(conversation_to_test)
print(formatted_conversation)
shap_values = explainer([formatted_conversation])

# %%
print(shap_values.shape)
# print(shap_values[0, :, 0].shape)
# print(shap_values.mean(0)[:,0].shape)
# print(shap_values.data.shape)
shap.plots.bar(shap_values[0], max_display=5)
shap_values.mean(0)
# shap_values
shap_values[0][0].values
# %%
tokens = tokenizer.tokenize(formatted_conversation)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
unique_tokens = set(token_ids)

id_to_shapley = dict()
for i, tid in enumerate(token_ids):
    id_to_shapley.setdefault(tid, (0, 0))
    prev_shv, prev_count = id_to_shapley[tid]
    id_to_shapley[tid] = (shap_values[0][i].values + prev_shv, prev_count + 1)
average_shapley = dict()
for k, v in id_to_shapley.items():
    average_shapley[k] = v[0]/v[1]

ids = list(average_shapley.keys())
values = [average_shapley[k] for k in ids]
tokens = tokenizer.convert_ids_to_tokens(ids)
shap.plots.bar(shap.Explanation(values, data=tokens), max_display=6)

# %%
