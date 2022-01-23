import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
import random

def train(args, model, optimizer, epoch, gpu, data_loader, scheduler):

    print("EPOCH %d" % epoch)

    losses = []

    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):

        inputs_id, segment_ids, masks, labels1, labels2 = next(data_iter)

        inputs_id, segment_ids, masks, labels1, labels2 = torch.LongTensor(inputs_id), torch.LongTensor(segment_ids),\
                                                          torch.LongTensor(masks), torch.tensor(labels1), torch.tensor(labels2)

        inputs_id, segment_ids, masks, labels1, labels2 = inputs_id.cuda(gpu), segment_ids.cuda(gpu), masks.cuda(gpu),\
                                                          labels1.cuda(gpu), labels2.cuda(gpu)

        num_pred, nen_pred, loss = model(inputs_id, segment_ids, masks, labels1, labels2, "train")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
    return losses

def test(args, model, fold, gpu, data_loader):

    y, yhat, num_y, num_yhat = [], [], [], []

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():
            inputs_id, segment_ids, masks, labels1, labels2, generate_score = next(data_iter)

            inputs_id, segment_ids, masks, labels1, labels2, generate_score = torch.LongTensor(inputs_id), torch.LongTensor(segment_ids),\
                                                              torch.LongTensor(masks), torch.tensor(labels1), torch.LongTensor(labels2), torch.tensor(generate_score)

            inputs_id, segment_ids, masks, labels1, labels2, generate_score = inputs_id.cuda(gpu), segment_ids.cuda(gpu), masks.cuda(gpu),\
                                                              labels1.cuda(gpu), labels2.cuda(gpu), generate_score.cuda(gpu)

            num_pred, nen_pred, loss = model(inputs_id, segment_ids, masks, labels1, labels2, fold)

            num_pred = num_pred.detach().cpu().numpy()
            num_pred = np.argmax(num_pred, axis=1)
            num_y.append(num_pred)
            num_yhat.append(labels1.cpu().numpy())

            nen_pred = torch.softmax(nen_pred, dim=-1)
            generate_score = torch.softmax(generate_score, dim=-1)
            generate_score = 1 - generate_score
            nen_pred = nen_pred + generate_score
            nen_pred = nen_pred.detach().cpu().numpy()
            nen_pred = np.argsort(-nen_pred, axis=-1)
            nen_pred_new = []
            for t in range(len(num_pred)):
                nen_pred_new.append(nen_pred[t][:num_pred[t]].tolist())
            y.append(nen_pred_new)
            yhat.append(labels2.cpu().numpy())

    pred, valid, nums = [], [], []
    for i in range(len(y)):
        for j in range(len(y[i])):
            valid.append(1)
            if (1 not in yhat[i][j]):
                pred.append(0)
            else:
                y_hat_results = np.where(yhat[i][j] == 1)[0].tolist()
                y_results = y[i][j]
                y_hat_results.sort()
                y_results.sort()
                if y_hat_results == y_results:
                    pred.append(1)
                else:
                    pred.append(0)
            nums.append(num_yhat[i][j])
    uni_pred, uni_valid = [], []
    mul_pred, mul_valid = [], []
    for i in range(len(nums)):
        if nums[i] == 1:
            uni_pred.append(pred[i])
            uni_valid.append(valid[i])
        else:
            mul_pred.append(pred[i])
            mul_valid.append(valid[i])

    uni_score = metrics.accuracy_score(uni_valid, uni_pred)
    mul_score = metrics.accuracy_score(mul_valid, mul_pred)
    score = metrics.accuracy_score(valid, pred)

    print("uni_score, mul_score, score")
    print("%.4f, %.4f, %.4f" % (uni_score, mul_score, score))
    print()

    number_pred, num_valid = [], []
    for i in range(len(num_y)):
        for j in range(len(num_y[i])):
            number_pred.append(num_y[i][j])
            num_valid.append(num_yhat[i][j])
    num_uni_pred, num_uni_valid = [], []
    num_mul_pred, num_mul_valid = [], []
    for i in range(len(num_valid)):
        if num_valid[i] == 1:
            num_uni_pred.append(number_pred[i])
            num_uni_valid.append(num_valid[i])
        else:
            num_mul_pred.append(number_pred[i])
            num_mul_valid.append(num_valid[i])

    num_uni_score = metrics.accuracy_score(num_uni_valid, num_uni_pred)
    num_mul_score = metrics.accuracy_score(num_mul_valid, num_mul_pred)
    num_score = metrics.accuracy_score(num_valid, number_pred)

    print("num_uni_score, num_mul_score, num_score")
    print("%.4f, %.4f, %.4f" % (num_uni_score, num_mul_score, num_score))
    print()
