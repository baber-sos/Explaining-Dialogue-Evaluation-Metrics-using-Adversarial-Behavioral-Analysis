from transformers import AdamW, BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import random
import json
import copy
import torch
import warnings
import numpy as np
import os
import pickle
from sklearn.metrics import cohen_kappa_score
from spearman import spearman
from tqdm import tqdm

warnings.filterwarnings("ignore")


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_main_score(scores):
    number = [0, 0, 0, 0, 0]
    for item in scores:
        number[item] += 1
    score = np.argmax(number)
    return score


def load_ccpe(dirname, tokenizer):
    print(dirname)
    name = 'hierarchical_data'

    if os.path.exists(f'{dirname}/tokenized/{name}.pkl'):
        return read_pkl(f'{dirname}/tokenized/{name}.pkl')
    print('tokenized data')
    raw = [line[:-1] for line in open(f'{dirname}/data.txt', encoding='utf-8')]
    data = []
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    x = []
    emo = []
    act1 = []
    act2 = []
    action_list1 = {}
    action_list2 = {}
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, action, score = turn.split('\t')
            score = score.split(',')
            action = action.split(',')
            action = action[0]
            if action == '':
                action1, action2 = 'other', 'other'
            else:
                action1, action2 = action.split('+')
            if role == 'USER':
                x.append(copy.deepcopy(his_input_ids))
                emo.append(get_main_score([int(item) - 1 for item in score]))
                action1 = action1.strip()
                if action1 not in action_list1:
                    action_list1[action1] = len(action_list1)
                act1.append(action_list1[action1])

                action2 = action2.strip()
                if action2 not in action_list2:
                    action_list2[action2] = len(action_list2)
                act2.append(action_list2[action2])
            ids = tokenizer.encode(text.strip())[1:]
            his_input_ids.append(ids)
    action_num1 = len(action_list1)
    action_num2 = len(action_list2)
    data = [x, emo, act1, act2, action_num1, action_num2]
    write_pkl(data, f'{dirname}/tokenized/{name}.pkl')
    return data


def load_dstc(dirname, tokenizer):
    print(dirname)
    name = 'hierarchical_data'

    if os.path.exists(f'{dirname}/tokenized/{name}.pkl'):
        return read_pkl(f'{dirname}/tokenized/{name}.pkl')

    print('tokenized data')
    raw = [line[:-1] for line in open(f'{dirname}/data.txt', encoding='utf-8')]
    data = []
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    x = []
    emo = []
    act = []
    action_list = {}
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, action, score = turn.split('\t')
            score = score.split(',')
            action = action.split(',')
            action = action[0]
            if role == 'USER':
                x.append(copy.deepcopy(his_input_ids))
                emo.append(get_main_score([int(item) - 1 for item in score]))
                action = action.strip()
                if action not in action_list:
                    action_list[action] = len(action_list)
                act.append(action_list[action])

            ids = tokenizer.encode(text.strip())[1:]
            his_input_ids.append(ids)

    action_num = len(action_list)
    data = [x, emo, act, action_num]
    write_pkl(data, f'{dirname}/tokenized/{name}.pkl')
    return data


class HierarchicalData(Dataset):
    def __init__(self, x, act, dialog_used=5, up_sampling=False):
        self.x = x
        self.act = act
        self.dialog_used = dialog_used

        if up_sampling:
            enhance_idx = [idx for idx, a in enumerate(act) if a != 2]
            enhance_idx = enhance_idx * 10
            enhance_x = [x[idx] for idx in enhance_idx]
            enhance_act = [act[idx] for idx in enhance_idx]
            self.x = x + enhance_x
            self.act = act + enhance_act

    def __getitem__(self, index):
        x = [torch.tensor([101])] * (self.dialog_used - len(self.x[index])) + \
            [torch.tensor([101] + item[:64]) for item in self.x[index][-self.dialog_used:]]
        act = self.act[index]
        return x, act

    def __len__(self):
        return len(self.x)


class FlatData(Dataset):
    def __init__(self, x, act, dialog_used=5, up_sampling=False):
        self.x = x
        self.act = act
        self.dialog_used = dialog_used

        if up_sampling:
            enhance_idx = [idx for idx, a in enumerate(act) if a != 2]
            enhance_idx = enhance_idx * 10
            enhance_x = [x[idx] for idx in enhance_idx]
            enhance_act = [act[idx] for idx in enhance_idx]
            self.x = x + enhance_x
            self.act = act + enhance_act

    def __getitem__(self, index):
        seq = sum([item[:64] for item in self.x[index]], [])
        x = torch.tensor([101] + seq[-500:])
        act = self.act[index]
        return x, act

    def __len__(self):
        return len(self.x)


def collate_fn(data):
    x, act = zip(*data)
    bc_size = len(x)
    dialog_his = len(x[0])
    x = [item for dialog in x for item in dialog]
    x = pad_sequence(x, batch_first=True, padding_value=0)
    x = x.view(bc_size, dialog_his, -1)

    return {'input_ids': x,
            'act': torch.tensor(act).long()
            }


def flat_collate_fn(data):
    x, act = zip(*data)
    x = pad_sequence(x, batch_first=True, padding_value=0)

    return {'input_ids': x,
            'act': torch.tensor(act).long()
            }


def train(fold=0, data_name='dstc8', model_name='HiGRU+ATTN', dialog_used=10):
    print('[TRAIN]')

    data_name = data_name.replace('\r', '')
    model_name = model_name.replace('\r', '')

    print('dialog used', dialog_used)

    name = f'{data_name}_{model_name}_{fold}'
    print('TRAIN ::', name)

    save_path = f'outputs/{data_name}_emo/{model_name}_{fold}'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # x, emo, act1, act2, action_num1, action_num2 = load_ccpe(f'dataset/{data_name}', tokenizer)
    x, emo, act, action_num = load_dstc(f'dataset/{data_name}', tokenizer)

    # print(action_num)
    from models import GRU, GRUAttention, BERTBackbone
    from models import HierarchicalAttention, Hierarchical, ClassModel
    if model_name == 'HiGRU+ATTN':
        model = HierarchicalAttention(backbone=GRUAttention(vocab_size=tokenizer.vocab_size), class_num=[5])
        model = model.cuda()
        optimizer = AdamW(model.parameters(), 1e-4)
        batch_size = 16
        DataFunc = HierarchicalData
        cf = collate_fn
    elif model_name == 'HiGRU':
        model = Hierarchical(backbone=GRU(vocab_size=tokenizer.vocab_size), class_num=[5])
        model = model.cuda()
        optimizer = AdamW(model.parameters(), 1e-4)
        batch_size = 16
        DataFunc = HierarchicalData
        cf = collate_fn
    elif model_name == 'GRU':
        model = ClassModel(backbone=GRU(vocab_size=tokenizer.vocab_size), class_num=[5])
        model = model.cuda()
        optimizer = AdamW(model.parameters(), 1e-4)
        batch_size = 16
        DataFunc = FlatData
        cf = flat_collate_fn
    elif model_name == 'BERT':
        model = ClassModel(backbone=BERTBackbone(layers_used=2, name='bert-base-uncased'), class_num=[5])
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        optimizer = AdamW(model.parameters(), 2e-5)
        batch_size = 16
        DataFunc = FlatData
        cf = flat_collate_fn
    else:
        print('[unknown model name]')
        return

    ll = int(len(x) / 10)
    train_x = x[:ll * fold] + x[ll * (fold + 1):]
    train_act = emo[:ll * fold] + emo[ll * (fold + 1):]

    test_x = x[ll * fold:ll * (fold + 1)]
    test_act = emo[ll * fold:ll * (fold + 1)]

    print(len(train_x), len(test_x))
    print()
    best_result = [0. for _ in range(4)]
    for i in range(100):
        print('train epoch', i, name)
        train_loader = DataLoader(DataFunc(train_x, train_act, dialog_used=dialog_used, up_sampling=True),
                                  batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=cf)
        # tk0 = tqdm(train_loader, total=len(train_loader))
        tk0 = train_loader
        act_acc = []
        model.train()
        epoch_pbar = tqdm(enumerate(tk0))
        avg_loss_pe = 0.0
        for j, batch in epoch_pbar:
            act_pred, *o = model(input_ids=batch['input_ids'].cuda())
            act = batch['act'].cuda()
            act_loss = F.cross_entropy(act_pred, act)
            loss = act_loss
            avg_loss_pe += float(loss.item())
            epoch_pbar.set_description(str(avg_loss_pe/(j + 1)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            act_acc.append((act_pred.argmax(dim=-1) == act).sum().item() / act.size(0))

            # tk0.set_postfix(act_acc=round(sum(act_acc) / max(1, len(act_acc)), 4))
        torch.save(model.state_dict(), f'outputs/{name}_{i}.pt')
        # print('test epoch', i)
        test_result = test(model, DataFunc(test_x, test_act, dialog_used=dialog_used), f'{save_path}_{i}.txt', cf)
        best_result = [max(i1, i2) for i1, i2 in zip(test_result, best_result)]
        print(f'text_result={test_result}')
        print(f'best_result={best_result}')
        print()


def test(model, test_data, save_path, cf):
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False, num_workers=0, collate_fn=cf)
    # tk0 = tqdm(test_loader, total=len(test_loader))
    tk0 = test_loader
    prediction = []
    label = []

    model.eval()
    for j, batch in enumerate(tk0):
        act = batch['act'].cuda()
        with torch.no_grad():
            act_pred, *o = model(input_ids=batch['input_ids'].cuda())
        prediction.extend(act_pred.argmax(dim=-1).cpu().tolist())
        label.extend(act.cpu().tolist())

    recall = [[0, 0] for _ in range(5)]
    for p, l in zip(prediction, label):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]
    print('Recall value:', recall_value)
    print('Recall:', recall)
    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(prediction, label)
    rho = spearman(prediction, label)

    bi_pred = [int(item < 2) for item in prediction]
    bi_label = [int(item < 2) for item in label]
    bi_recall = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if l == 1]) / max(bi_label.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if p == 1]) / max(bi_pred.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

    with open(save_path, 'w', encoding='utf-8') as f:
        for p, l in zip(prediction, label):
            f.write(f'{p}, {l}\n')

    return UAR, kappa, rho, bi_f1
