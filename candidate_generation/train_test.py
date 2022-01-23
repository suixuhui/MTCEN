import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
import random
import pandas as pd

def train(args, model, optimizer, epoch, gpu, data_loader, scheduler, all_tokens, all_masks):

    print("EPOCH %d" % epoch)

    losses = []

    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        inputs_id, masks, gold = next(data_iter)

        inputs_id, masks = torch.LongTensor(inputs_id), torch.LongTensor(masks)

        inputs_id, masks = inputs_id.cuda(gpu), masks.cuda(gpu)

        loss = model(inputs_id, masks, gold, all_tokens, all_masks, "train", test_descriptions=None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    return losses

def test(args, model, fold, gpu, data_loader, all_tokens, all_masks):

    y, yhat, ysort = [], [], []

    model.eval()
    with torch.no_grad():
        test_tokens, test_masks = torch.LongTensor(all_tokens).cuda(args.gpu), \
                                  torch.LongTensor(all_masks).cuda(args.gpu)
        descriptions = model.get_descriptions(test_tokens, test_masks)

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():
            inputs_id, masks, gold = next(data_iter)

            inputs_id, masks = torch.LongTensor(inputs_id), torch.LongTensor(masks)

            inputs_id, masks = inputs_id.cuda(gpu), masks.cuda(gpu)

            y_pred, y_sorted = model(inputs_id, masks, gold, all_tokens, all_masks, "test", descriptions)

            for sor in y_sorted:
                ysort.append(sor.detach().cpu().numpy())
            for pred in y_pred:
                y.append(pred.detach().cpu().numpy())
            for go in gold:
                yhat.append(go)

    if args.test_model:
        dictionary = []
        with open("./data/yidu-n7k/code.txt", "r", encoding='utf-8') as f:
            for line in f.readlines():
                entity = line.split('\t')[1].strip()
                dictionary.append(entity)
        if fold == "train":
            train_df = pd.read_excel('./data/yidu-n7k/train.xlsx')
            train_data = train_df.values
            train_datas = {}
            for i in range(len(train_data)):
                target = []
                for gold in y[i]:
                    if dictionary[gold] not in target:
                        target.append(dictionary[gold])
                train_datas[train_data[i][0]] = target[:args.train_top_k]
            pickle.dump(train_datas, open('./data/train_candidates_20', 'wb'), -1)
        else:
            val_df = pd.read_excel('./data/yidu-n7k/val.xlsx')
            val_data = val_df.values
            answer_df = pd.read_excel('./data/yidu-n7k/answer.xlsx')
            answer_data = answer_df.values
            test_datas = {}
            generate_scores = {}
            for i in range(len(val_data)):
                scores = []
                target = []
                for j in range(len(y[i])):
                    if dictionary[y[i][j]] not in target:
                        target.append(dictionary[y[i][j]])
                        scores.append(ysort[i][j])
                test_datas[val_data[i][0]] = target[:args.top_k]
                generate_scores[val_data[i][0]] = scores[:args.top_k]
            for i in range(len(answer_data)):
                scores = []
                target = []
                for j in range(len(y[i + len(val_data)])):
                    if dictionary[y[i + len(val_data)][j]] not in target:
                        target.append(dictionary[y[i + len(val_data)][j]])
                        scores.append(ysort[i + len(val_data)][j])
                test_datas[answer_data[i][0]] = target[:args.top_k]
                generate_scores[answer_data[i][0]] = scores[:args.top_k]
            pickle.dump(test_datas, open('./data/test_candidates', 'wb'), -1)
            pickle.dump(generate_scores, open('./data/generate_scores', 'wb'), -1)

            datas = test_datas
            test_wrong = 0
            for i in range(len(datas)):
                for gold_entity in yhat[i]:
                    if dictionary[gold_entity] not in list(datas.values())[i]:
                        test_wrong += 1
                        break
            print("recall:", 1 - test_wrong / 3000)
            print()