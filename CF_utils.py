import os
import random
import argparse
from CAFE.utils import TMP_DIR
import torch


import argparse
import os

# Assuming TMP_DIR is defined globally or within the function
TMP_DIR = {
    'beauty': os.path.join('.', 'CAFE', 'tmp', 'Beauty'),
    'cell': './tmp/Cellphones_Accessories',
    'clothing': './tmp/Clothing',
    'cd': './tmp/CDs_Vinyl',
}

CAFE_DIR = "./CAFE"  # This should be defined to reflect the actual directory structure

def str_to_bool(x):
    return str(x).lower() in ('true', 'yes', '1')

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run the recommendation system with counterfactual analysis.")
    parser.add_argument('--dataset', type=str, default='beauty', help="Name of the dataset to use.")
    parser.add_argument('--name', type=str, default='neural_symbolic_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help="Seed for random number generators.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID for CUDA execution.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate.")
    parser.add_argument('--embed_size', type=int, default=100, help='KG embedding size.')
    parser.add_argument('--deep_module', type=str_to_bool, default=True, help='Use deep module or not.')
    parser.add_argument('--use_dropout', type=str_to_bool, default=True, help='Use dropout or not.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=100, help='Number of steps per checkpoint.')
    parser.add_argument('--sample_size', type=int, default=15, help='Sample size for model.')
    parser.add_argument('--do_infer', type=str_to_bool, default=False, help='Whether to infer paths after training.')
    parser.add_argument('--do_execute', type=str_to_bool, default=False, help='Whether to execute neural programs.')
    parser.add_argument('--preprocess_dir', type=str, default=f'{CAFE_DIR}/preprocess.py', help="Path to preprocess.py.") 
    parser.add_argument('--model_path', type=str, help="Path to the trained model.")
    parser.add_argument('--train_neural_symbol_dir', type=str, default=f'{CAFE_DIR}/train_neural_symbol.py', help="Path to train_neural_symbol.py.")
    parser.add_argument('--user_id', type=int, default=0, help="User ID for generating recommendations and counterfactual analysis.")

    args = parser.parse_args()

    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    args.symbolic_model = os.path.join(args.log_dir, f'symbolic_model_epoch{args.epochs}.ckpt')
    args.infer_path_data = os.path.join(args.log_dir, 'infer_path_data.pkl')

    # Set GPU device.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args

args = parse_args()