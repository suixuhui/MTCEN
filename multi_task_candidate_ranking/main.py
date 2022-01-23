from options import args
import random
import numpy as np
import torch
import csv
import sys
from utils import prepare_instance_bert, prepare_instance_bert_test, MyDataset, my_collate_bert, my_collate_bert_test
from models import pick_model
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test
from transformers import AdamW, get_linear_schedule_with_warmup

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)



    # print(args)

    csv.field_size_limit(sys.maxsize)

    model = pick_model(args)

    if args.load_model:
        pretrained_model_path = args.load_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)


    train_instances = prepare_instance_bert(args.DATA_DIR + '/' + 'train_mentions_gold',
                                            args.DATA_DIR + '/' + 'train_candidates_20',
                                            args)
    # print("train_instances {}".format(len(train_instances)))
    test_instances = prepare_instance_bert_test(args.DATA_DIR + '/' + 'test_mentions_gold',
                                                args.DATA_DIR + '/' + 'test_candidates',
                                                args)
    # print("test_instances {}".format(len(test_instances)))

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=my_collate_bert)
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate_bert_test)


    optimizer = AdamW(
        model.parameters(),
        # optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    total_steps = len(train_loader) * args.n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    test_only = args.test_model is not None

    if not test_only:
        for epoch in range(args.n_epochs):
            if epoch == 0 and not args.test_model:
                dir_name = args.model + args.dir_name
                model_dir = os.path.join(args.MODEL_DIR, dir_name)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
            epoch_start = time.time()
            losses = train(args, model, optimizer, epoch, args.gpu, train_loader, scheduler)
            loss = np.mean(losses)
            epoch_finish = time.time()
            print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
            fold = 'dev'
            # test on dev
            evaluation_start = time.time()
            test(args, model, fold, args.gpu, test_loader)
            evaluation_finish = time.time()
            print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
            if epoch == args.n_epochs - 1:
                print("last epoch: testing on dev and test sets")
                test(args, model, fold, args.gpu, test_loader)
                torch.save(model.state_dict(), model_dir + '/' + model.name)
    else:
        model_dir = args.test_model
        pretrained_model_path = args.test_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        test(args, model, "test", args.gpu, test_loader)

    sys.stdout.flush()
