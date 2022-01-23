import csv
import numpy as np
import pickle
import pandas as pd
from options import args
import json
import torch
import math
import torch.nn.functional as F

from transformers import BertTokenizer
def prepare_instance_bert(filename, args):
    instances = []
    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    mention_gold = pickle.load(open(filename, 'rb'))

    for mention, gold in mention_gold.items():

        mention_tokens = wp_tokenizer.tokenize(mention)
        tokens_max_len = int(args.mentions_max_length - 2)
        if len(mention_tokens) > tokens_max_len:
            mention_tokens = mention_tokens[:tokens_max_len]
        mention_tokens.insert(0, '[CLS]')
        mention_tokens.append('[SEP]')
        mention_tokens_id = wp_tokenizer.convert_tokens_to_ids(mention_tokens)

        mention_masks = [1] * len(mention_tokens_id)
        mention_padding = [0] * (args.mentions_max_length - len(mention_tokens_id))
        mention_tokens_id += mention_padding
        mention_masks += mention_padding

        dict_instance = {'mention_tokens_id': mention_tokens_id,
                         'mention_masks': mention_masks,
                         'gold': gold}

        instances.append(dict_instance)

    return instances


from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=np.int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x


def my_collate_bert(x):
    mention_inputs_id = [x_['mention_tokens_id'] for x_ in x]
    mention_masks = [x_['mention_masks'] for x_ in x]
    gold = [x_['gold'] for x_ in x]

    return mention_inputs_id, mention_masks, gold


def get_positive(targets):
    positive = []
    for target in targets:
        positive += [target] * args.hard
    return positive

def get_negative_hard(targets, model, tokens, masks):
    negative = []
    hard_number = args.hard
    with torch.no_grad():
        tokens, masks = torch.LongTensor(tokens).cuda(args.gpu), \
                        torch.LongTensor(masks).cuda(args.gpu)
        descriptions = model.get_descriptions(tokens, masks)
        single = []
        for target in targets:
            distance = F.pairwise_distance(descriptions[target], descriptions)
            sorted, indices = torch.sort(distance, descending=False)
            indices = indices.cpu().numpy().tolist()
            for indice in indices:
                if indice not in targets:
                    single.append(indice)
            single = single[:hard_number]
            negative.extend(single)
    return negative

def get_description():
    descriptions = list()
    with open("./data/yidu-n7k/code.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            entity = line.split('\t')[1].strip()
            descriptions.append(entity)

    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    all_tokens = []
    all_masks = []
    for description in descriptions:
        tokens = wp_tokenizer.tokenize(description)

        tokens_max_len = args.candidates_max_length - 2
        if len(tokens) > tokens_max_len:
            tokens = tokens[:tokens_max_len]

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')

        tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(tokens)
        candidate_padding = [0] * (args.candidates_max_length - len(tokens))
        tokens_id += candidate_padding
        masks += candidate_padding
        all_tokens.append(tokens_id)
        all_masks.append(masks)

    return all_tokens, all_masks