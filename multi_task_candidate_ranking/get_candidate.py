import pandas as pd
from gensim.summarization import bm25
import pickle
from random import shuffle

entities = []
entities_cut = []
train_df = pd.read_excel('./data/yidu-n7k/train.xlsx')
train_data = train_df.values
train_datas = {}
train_mentions = []
train_mentions_cut = []
for i in range(len(train_data)):
    train_datas[train_data[i][0]] = train_data[i][1].split("##")
    for gold in train_data[i][1].split("##"):
        if gold not in entities:
            entities.append(gold)
            entities_cut.append(list(gold))
    train_mentions.append(train_data[i][0])
    train_mentions_cut.append(list(train_data[i][0]))
print(train_datas)
val_df = pd.read_excel('./data/yidu-n7k/val.xlsx')
val_data = val_df.values
answer_df = pd.read_excel('./data/yidu-n7k/answer.xlsx')
answer_data = answer_df.values
test_datas = {}
test_mentions = []
test_mentions_cut = []
for i in range(len(val_data)):
    test_datas[val_data[i][0]] = val_data[i][1].split("##")
    for gold in val_data[i][1].split("##"):
        if gold not in entities:
            entities.append(gold)
            entities_cut.append(list(gold))
    test_mentions.append(val_data[i][0])
    test_mentions_cut.append(list(val_data[i][0]))
for i in range(len(answer_data)):
    test_datas[answer_data[i][0]] = answer_data[i][1].split("##")
    for gold in answer_data[i][1].split("##"):
        if gold not in entities:
            entities.append(gold)
            entities_cut.append(list(gold))
    test_mentions.append(answer_data[i][0])
    test_mentions_cut.append(list(answer_data[i][0]))
print(len(test_datas))



pickle.dump(train_datas, open('./data/train_mentions_gold', 'wb'), -1)
pickle.dump(test_datas, open('./data/test_mentions_gold', 'wb'), -1)
