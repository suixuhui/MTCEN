import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='./data')

# model
parser.add_argument("-model", type=str, choices=["bert", "bilstm"], default='bert')
parser.add_argument("-mentions_max_length", type=int, default=40)
parser.add_argument("-candidates_max_length", type=int, default=25)
parser.add_argument("-repeat_number", type=int, default=20)
parser.add_argument("-test_model", type=str, default=None)
parser.add_argument("-load_model", type=str, default=None)
parser.add_argument("-embed_size", type=int, default=768)
parser.add_argument("-alpha", type=float, default=0.3)

# training
parser.add_argument("-n_epochs", type=int, default=10)
parser.add_argument("-dropout", type=float, default=0.5)
parser.add_argument("-dir_name", type=str, default='try')
parser.add_argument("-top_k", type=int, default=20)
parser.add_argument("-test_top_k", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=2e-5)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-gpu", type=int, default=0, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-tune_wordemb", action="store_const", const=True, default=True)
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')

parser.add_argument('-bert_dir', type=str, default="bert-base-chinese")

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
