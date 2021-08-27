from train_dstc import train
import os
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('-fold', type=int)
parser.add_argument('--data', type=str, default='dstc8')
parser.add_argument('--model', type=str, default='HiGRU+ATTN')
args = parser.parse_args()

print('CUDA_VISIBLE_DEVICES', torch.cuda.device_count())
print('train data', args.data)
print('train model', args.model)
print('train fold', args.fold)

train(fold=args.fold, data_name=args.data, model_name=args.model)



