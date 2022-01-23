import torch.nn as nn
import os
import numpy as np
from transformers import BertModel
import torch.nn.functional as F
import torch
from torch.nn.init import xavier_uniform_ as xavier_uniform
from utils import get_positive
from options import args
import time
import copy
from torch.nn.init import xavier_uniform_ as xavier_uniform

class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
        self.loss = nn.CrossEntropyLoss()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.classifier = nn.Linear(args.embed_size, 2)
        self.num_classifier = nn.Linear(args.embed_size, 8)
        self.num_multihead_attn = nn.MultiheadAttention(args.embed_size, 8)
        self.nen_multihead_attn = nn.MultiheadAttention(args.embed_size, 8)

    def forward(self, input_ids, segment_ids, attention_mask, labels1, labels2, fold):

        if fold != "train":
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
            flat_segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            x, nen_pooled_out = self.bert(flat_input_ids, token_type_ids=flat_segment_ids, attention_mask=flat_attention_mask)

            mention_num = []
            for i in range(len(flat_segment_ids)):
                n = 0
                for segment_id in flat_segment_ids[i]:
                    if segment_id == 1:
                        n += 1
                    else:
                        break
                mention_num.append(n)
            mention = []
            for i in range(len(mention_num)):
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]
                men = torch.mean(men, dim=0)
                mention.append(men)
            mention = torch.cat(mention, dim=0)
            mention = mention.view(len(mention_num), -1)

            num_key = num_value = torch.unsqueeze(mention, dim=0)
            num_query = torch.unsqueeze(nen_pooled_out, dim=0)
            num_outputs, _ = self.num_multihead_attn(num_query, num_key, num_value)
            num_outputs = torch.squeeze(num_outputs)

            nen_key = nen_value = torch.unsqueeze(nen_pooled_out, dim=0)
            nen_query = torch.unsqueeze(mention, dim=0)
            nen_outputs, _ = self.nen_multihead_attn(nen_query, nen_key, nen_value)
            nen_outputs = torch.squeeze(nen_outputs)

            num_logits = self.num_classifier(num_outputs)
            num_logits = num_logits.view(input_ids.size(0), args.test_top_k, -1)
            num_logits = torch.mean(num_logits, dim=1)
            num_loss = self.loss(num_logits, labels1)
            nen_logits = self.classifier(nen_outputs)
            nen_logits = nen_logits.view(input_ids.size(0) * args.test_top_k, -1)
            labels2 = labels2.view(input_ids.size(0) * args.test_top_k)
            nen_loss = self.loss(nen_logits, labels2)
            nen_logits = nen_logits.view(input_ids.size(0), args.test_top_k, -1)
            labels2 = labels2.view(input_ids.size(0), args.test_top_k)
            nen_logits_pred = torch.softmax(nen_logits, dim=-1)
            nen_logits_pred = torch.index_select(nen_logits_pred, dim=-1, index=torch.LongTensor([1]).cuda(self.gpu))
            nen_logits_pred = nen_logits_pred.view(input_ids.size(0), args.test_top_k)
            # loss = args.alpha * num_loss + (1 - args.alpha) * nen_loss
            loss = num_loss + nen_loss
            return num_logits, nen_logits_pred, loss

        else:
            x, nen_pooled_out = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
            mention_num = []
            for i in range(len(segment_ids)):
                n = 0
                for segment_id in segment_ids[i]:
                    if segment_id == 1:
                        n += 1
                    else:
                        break
                mention_num.append(n)
            mention = []
            for i in range(len(mention_num)):
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]
                men = torch.mean(men, dim=0)
                mention.append(men)
            mention = torch.cat(mention, dim=0)
            mention = mention.view(len(mention_num), -1)

            num_key = num_value = torch.unsqueeze(mention, dim=0)
            num_query = torch.unsqueeze(nen_pooled_out, dim=0)
            num_outputs, _ = self.num_multihead_attn(num_query, num_key, num_value)
            num_outputs = torch.squeeze(num_outputs)

            nen_key = nen_value = torch.unsqueeze(nen_pooled_out, dim=0)
            nen_query = torch.unsqueeze(mention, dim=0)
            nen_outputs, _ = self.nen_multihead_attn(nen_query, nen_key, nen_value)
            nen_outputs = torch.squeeze(nen_outputs)

            num_logits = self.num_classifier(num_outputs)
            num_loss = self.loss(num_logits, labels1)
            nen_logits = self.classifier(nen_outputs)
            nen_loss = self.loss(nen_logits, labels2)
            # loss = args.alpha * num_loss + (1 - args.alpha) * nen_loss
            loss = num_loss + nen_loss
            return num_logits, nen_logits, loss


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model