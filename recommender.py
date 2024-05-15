import pickle
import sys
sys.path.append('CAFE')

from CAFE.utils import *
import os
import torch

import CAFE.my_knowledge_graph_shrinkable as my_knowledge_graph_shrinkable
from CAFE.execute_neural_symbol import *

TMP_DIR = './CAFE/tmp/Beauty/neural_symbolic_model'

def get_recommendation(args):
    
    file_path = f'{TMP_DIR}/infer_path_data.pkl'
    args.symbolic_model = f'{TMP_DIR}/symbolic_model_epoch20.ckpt'
    args.infer_path_data = f'{TMP_DIR}/infer_path_data.pkl'
    if not os.path.exists(file_path):
        infer_paths(args)
    kg = utils.load_kg(args.dataset)
    kg_mask = KGMask(kg)

    train_labels = utils.load_labels(args.dataset, 'train')
    path_counts = utils.load_path_count(args.dataset)  # Training path freq
    with open(args.infer_path_data, 'rb') as f:
        raw_paths = pickle.load(f)  # Test path with scores

    symbolic_model = create_symbolic_model(args, kg, train=False)
    program_exe = MetaProgramExecutor(symbolic_model, kg_mask, args)

    uid = args.user_id
    program = create_heuristic_program(kg.metapaths, raw_paths[uid], path_counts[uid], args.sample_size)
    program_exe.execute(program, uid, train_labels[uid])
    paths = program_exe.collect_results(program)

    # Ensure the 'tmp' directory exists
    tmp_directory = 'tmp'
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    filename = f'tmp/paths_user_{args.user_id}.pkl'
    # Open a file in write-binary mode
    with open(filename, 'wb') as file:
        # Serialize the dictionary using pickle.dump
        pickle.dump(paths, file)

    return paths


