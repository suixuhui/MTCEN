import csv
import numpy as np
import pickle
import pandas as pd
from options import args
import json
import torch
import math

from transformers import BertTokenizer
def prepare_instance_bert(filename, candidatename, args):
    instances = []
    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    mention_gold = pickle.load(open(filename, 'rb'))
    candidates = pickle.load(open(candidatename, 'rb'))

    for mention, gold in mention_gold.items():
        candidate = candidates[mention]
        if 'train' in filename:
            for go in gold:
                if go not in candidate:
                    candidate.insert(0, go)
                    for j in range(len(candidate) - 1, -1, -1):
                        if candidate[j] not in gold:
                            del(candidate[j])
                            break

        mention_tokens = wp_tokenizer.tokenize(mention)
        tokens_max_len = int(args.mentions_max_length - 2)
        if len(mention_tokens) > tokens_max_len:
            mention_tokens = mention_tokens[:tokens_max_len]
        mention_tokens.insert(0, '[CLS]')
        mention_tokens.append('[SEP]')
        mention_tokens_id = wp_tokenizer.convert_tokens_to_ids(mention_tokens)

        candidate_max_len = int(args.candidates_max_length - 1)
        for ca in candidate[:args.top_k]:
            if ca not in gold:
                candidate_tokens = wp_tokenizer.tokenize(ca)
                if len(candidate_tokens) > candidate_max_len:
                    candidate_tokens = candidate_tokens[:candidate_max_len]
                candidate_tokens.append('[SEP]')
                candidate_tokens_id = wp_tokenizer.convert_tokens_to_ids(candidate_tokens)
                tokens_id = mention_tokens_id + candidate_tokens_id
                masks = [1] * len(tokens_id)
                segment_ids = [0] * len(mention_tokens_id) + [1] * len(candidate_tokens_id)
                padding = [0] * (args.mentions_max_length + args.candidates_max_length - len(tokens_id))
                tokens_id += padding
                masks += padding
                segment_ids += padding

                label1 = len(gold)
                label2 = 0

                dict_instance = {'label1': label1,
                                 'label2': label2,
                                 'tokens_id': tokens_id,
                                 'masks': masks,
                                 'segment_ids': segment_ids,
                                 }

                instances.append(dict_instance)
            else:
                for i in range(args.repeat_number):
                    candidate_tokens = wp_tokenizer.tokenize(ca)
                    if len(candidate_tokens) > candidate_max_len:
                        candidate_tokens = candidate_tokens[:candidate_max_len]
                    candidate_tokens.append('[SEP]')
                    candidate_tokens_id = wp_tokenizer.convert_tokens_to_ids(candidate_tokens)
                    tokens_id = mention_tokens_id + candidate_tokens_id
                    masks = [1] * len(tokens_id)
                    segment_ids = [0] * len(mention_tokens_id) + [1] * len(candidate_tokens_id)
                    padding = [0] * (args.mentions_max_length + args.candidates_max_length - len(tokens_id))
                    tokens_id += padding
                    masks += padding
                    segment_ids += padding

                    label1 = len(gold)
                    label2 = 1

                    dict_instance = {'label1': label1,
                                     'label2': label2,
                                     'tokens_id': tokens_id,
                                     'masks': masks,
                                     'segment_ids': segment_ids,
                                     }

                    instances.append(dict_instance)

    return instances

def prepare_instance_bert_test(filename, candidatename, args):
    instances = []
    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    mention_gold = pickle.load(open(filename, 'rb'))
    candidates = pickle.load(open(candidatename, 'rb'))
    generate_scores = pickle.load(open('./data/generate_scores', 'rb'))

    for mention, gold in mention_gold.items():
        candidate = candidates[mention]
        if 'train' in filename:
            for go in gold:
                if go not in candidate:
                    candidate.insert(0, go)

        mention_tokens = wp_tokenizer.tokenize(mention)
        tokens_max_len = int(args.mentions_max_length - 2)
        if len(mention_tokens) > tokens_max_len:
            mention_tokens = mention_tokens[:tokens_max_len]
        mention_tokens.insert(0, '[CLS]')
        mention_tokens.append('[SEP]')
        mention_tokens_id = wp_tokenizer.convert_tokens_to_ids(mention_tokens)

        candidate_max_len = int(args.candidates_max_length - 1)

        all_tokens, all_masks, all_segment_ids = [], [], []
        for ca in candidate[:args.test_top_k]:
            candidate_tokens = wp_tokenizer.tokenize(ca)
            if len(candidate_tokens) > candidate_max_len:
                candidate_tokens = candidate_tokens[:candidate_max_len]
            candidate_tokens.append('[SEP]')
            candidate_tokens_id = wp_tokenizer.convert_tokens_to_ids(candidate_tokens)
            tokens_id = mention_tokens_id + candidate_tokens_id
            masks = [1] * len(tokens_id)
            segment_ids = [0] * len(mention_tokens_id) + [1] * len(candidate_tokens_id)
            padding = [0] * (args.mentions_max_length + args.candidates_max_length - len(tokens_id))
            tokens_id += padding
            masks += padding
            segment_ids += padding
            all_tokens.append(tokens_id)
            all_masks.append(masks)
            all_segment_ids.append(segment_ids)

        label1 = len(gold)
        label2 = [0] * args.test_top_k
        for go in gold:
            if go not in candidate:
                label2 = [0] * args.test_top_k
                break
            for i in range(args.test_top_k):
                if go == candidate[i]:
                    label2[i] = 1
                    break
        g_score = generate_scores[mention]

        dict_instance = {'label1': label1,
                         'label2': label2,
                         'tokens_id': all_tokens,
                         'masks': all_masks,
                         'segment_ids': all_segment_ids,
                         'generate_score': g_score}

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
    inputs_id = [x_['tokens_id'] for x_ in x]
    segment_ids = [x_['segment_ids'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    labels1 = [x_['label1'] for x_ in x]
    labels2 = [x_['label2'] for x_ in x]

    return inputs_id, segment_ids, masks, labels1, labels2


def my_collate_bert_test(x):
    inputs_id = [x_['tokens_id'] for x_ in x]
    segment_ids = [x_['segment_ids'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    labels1 = [x_['label1'] for x_ in x]
    labels2 = [x_['label2'] for x_ in x]
    generate_score = [x_['generate_score'] for x_ in x]

    return inputs_id, segment_ids, masks, labels1, labels2, generate_score


def get_positive(target, nums):
    positive = [target] * nums
    return positive
