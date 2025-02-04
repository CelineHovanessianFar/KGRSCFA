import pickle
import sys
sys.path.append('CAFE')

from CAFE.utils import *
import os
import torch
from CAFE import preprocess
import subprocess

import CAFE.my_knowledge_graph_shrinkable as my_knowledge_graph_shrinkable
from CAFE.execute_neural_symbol import *
from data_registry import DataRegistry
from CF_utils import TMP_DIR


def preprocess_completed(args):
    count_file = os.path.join(TMP_DIR[args.dataset], "path_count.pkl")
    return os.path.exists(count_file)

def is_model_trained(model_path):
    return os.path.exists(model_path)


def get_recommendation(args):

    if not preprocess_completed(args):
        print("Running preprocess.py...")
        preprocess.main(args)
    print('Preprocess Step Already Completed!')
        
    if not is_model_trained(args.symbolic_model):
        print("Running train_neural_symbol.py...")
        subprocess.run(["python", args.train_neural_symbol_dir], check=True)
    print("Recommender System already trained.")
    
    if not os.path.exists(args.infer_path_data):
        infer_paths(args)
    kg_mask = KGMask(DataRegistry.kg)

    train_labels = utils.load_labels(args.dataset, 'train')
    path_counts = utils.load_path_count(args.dataset)  # Training path freq
    with open(args.infer_path_data, 'rb') as f:
        raw_paths = pickle.load(f)  # Test path with scores

    symbolic_model = create_symbolic_model(args, DataRegistry.kg, train=False)
    program_exe = MetaProgramExecutor(symbolic_model, kg_mask, args)

    uid = args.user_id
    program = create_heuristic_program(DataRegistry.kg.metapaths, raw_paths[uid], path_counts[uid], args.sample_size)
    program_exe.execute(program, uid, train_labels[uid])
    paths = program_exe.collect_results(program)

    # Ensure the 'tmp' directory exists
    tmp_directory = 'tmp'
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    filename = os.path.join('tmp', f'paths_user_{args.user_id}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(paths, file)

    return paths


