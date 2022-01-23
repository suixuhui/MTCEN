import torch.nn as nn
import os
import numpy as np
from transformers import BertModel
import torch.nn.functional as F
import torch
from torch.nn.init import xavier_uniform_ as xavier_uniform
from utils import get_positive, get_negative_hard
from options import args
import time
import copy
from torch.nn.init import xavier_uniform_ as xavier_uniform

class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.loss = F.triplet_margin_loss

    def forward(self, input_ids, attention_mask, gold, all_tokens, all_masks, fold, test_descriptions):
        x, mention_pooled_out = self.bert(input_ids, attention_mask=attention_mask)
        if fold == "train":
            label_negative = [get_negative_hard(label_target, self, all_tokens, all_masks) for label_target in gold]
            label_positive = [get_positive(label_target) for label_target in gold]
            x_avgs = []
            for i in range(len(gold)):
                x_avgs.extend([mention_pooled_out[i]] * (args.hard * len(gold[i])))
            x_avgs = torch.cat(x_avgs, dim=0)

            negative_tokens = [[all_tokens[label_negative[i][j]] for j in range(len(label_negative[i]))] for i in
                               range(len(label_negative))]
            negative_masks = [[all_masks[label_negative[i][j]] for j in range(len(label_negative[i]))] for i in
                              range(len(label_negative))]
            positive_tokens = [[all_tokens[label_positive[i][j]] for j in range(len(label_positive[i]))] for i in
                               range(len(label_positive))]
            positive_masks = [[all_masks[label_positive[i][j]] for j in range(len(label_positive[i]))] for i in
                              range(len(label_positive))]

            all_negetive_tokens, all_negative_masks, all_positive_tokens, all_positive_masks = [], [], [], []
            for i in range(len(negative_tokens)):
                for token in negative_tokens[i]:
                    all_negetive_tokens.append(token)
                for token in positive_tokens[i]:
                    all_positive_tokens.append(token)
                for mask in negative_masks[i]:
                    all_negative_masks.append(mask)
                for mask in positive_masks[i]:
                    all_positive_masks.append(mask)

            train_negative_tokens, train_negative_masks = torch.LongTensor(all_negetive_tokens).cuda(self.gpu), \
                                                          torch.LongTensor(all_negative_masks).cuda(self.gpu)

            train_positive_tokens, train_positive_masks = torch.LongTensor(all_positive_tokens).cuda(self.gpu), \
                                                          torch.LongTensor(all_positive_masks).cuda(self.gpu)
            x, negative_x = self.bert(train_negative_tokens, attention_mask=train_negative_masks)
            x, positive_x = self.bert(train_positive_tokens, attention_mask=train_positive_masks)
            x_avgs = x_avgs.view(positive_x.size(0), -1)
            final_loss = self.loss(anchor=x_avgs, positive=positive_x, negative=negative_x, margin=1.0)

            return final_loss
        else:
            y_sorted = []
            y_pred = []
            for mention in mention_pooled_out:
                distance = F.pairwise_distance(mention, test_descriptions)
                sorted, indices = torch.sort(distance, descending=False)
                y_sorted.append(sorted[:args.train_top_k*2])
                y_pred.append(indices[:args.train_top_k*2])
            return y_pred, y_sorted

    def get_descriptions(self, tokens, masks):
        x, pooled_out = self.bert(tokens, attention_mask=masks)
        return pooled_out


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model