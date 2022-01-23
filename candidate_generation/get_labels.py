import pandas as pd
import pickle

dictionary = []
with open("./data/yidu-n7k/code.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        entity = line.split('\t')[1].strip()
        dictionary.append(entity)
train_df = pd.read_excel('./data/yidu-n7k/train.xlsx')
train_data = train_df.values
train_datas = {}
for i in range(len(train_data)):
    target = []
    golds = train_data[i][1].split("##")
    for gold in golds:
        if gold not in dictionary:
            print(gold)
        target.append(dictionary.index(gold))
        train_datas[train_data[i][0]] = target
print(train_datas)
print(len(train_datas))

val_df = pd.read_excel('./data/yidu-n7k/val.xlsx')
val_data = val_df.values
answer_df = pd.read_excel('./data/yidu-n7k/answer.xlsx')
answer_data = answer_df.values
test_datas = {}
for i in range(len(val_data)):
    target = []
    golds = val_data[i][1].split("##")
    for gold in golds:
        if gold not in dictionary:
            print(gold)
        target.append(dictionary.index(gold))
        test_datas[val_data[i][0]] = target
for i in range(len(answer_data)):
    target = []
    golds = answer_data[i][1].split("##")
    for gold in golds:
        if gold not in dictionary:
            print(gold)
        target.append(dictionary.index(gold))
        test_datas[answer_data[i][0]] = target
print(test_datas)
print(len(test_datas))

pickle.dump(train_datas, open('./data/train_mentions_gold', 'wb'), -1)
pickle.dump(test_datas, open('./data/test_mentions_gold', 'wb'), -1)