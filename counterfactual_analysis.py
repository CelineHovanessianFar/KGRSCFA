from tqdm import tqdm
import sys
import torch
sys.path.append('CAFE')
import pickle
import os
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np

from CAFE.execute_neural_symbol import create_symbolic_model, MetaProgramExecutor,create_heuristic_program, infer_paths
from kg_info import KGENtitiesRelationsInfo
from CAFE import utils
from counterfactual_generator import EntityFilter


class ClusterCounterfactuals:
    def __init__(self, recommendation_path) -> None:
        self.kg = utils.load_kg('beauty')
        self.kg_info = KGENtitiesRelationsInfo()
        self.recommendation_path = recommendation_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.counter_scores = {}


    def find_counter_mp(self, entity):
        if entity == 'brand':
            mp_template = [[
                    (None, 'user'), 
                    ('purchase', 'product'), 
                    ('produced_by', 'brand'),
                    ('rev_produced_by', 'product')
                ]]
        elif entity == 'category':
            mp_template = [[
                    (None, 'user'), 
                    ('purchase', 'product'), 
                    ('belongs_to', 'category'),
                    ('rev_belongs_to', 'product')
                ]]
        elif entity == 'word':
            mp_template = [[
                    (None, 'user'), 
                    ('purchase', 'product'), 
                    ('described_by', 'word'),
                    ('rev_described_by', 'product')
                ]]
        elif entity == 'related_product':
            mp_template = [
                [(None, 'user'),          ('purchase', 'product'),      ('also_bought',     'related_product'),     ('rev_also_bought',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('also_bought',     'related_product'),     ('rev_also_viewed',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('also_bought',     'related_product'),     ('rev_bought_together', 'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('also_viewed',     'related_product'),     ('rev_also_bought',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('also_viewed',     'related_product'),     ('rev_also_viewed',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('also_viewed',     'related_product'),     ('rev_bought_together', 'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('bought_together', 'related_product'),     ('rev_also_bought',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('bought_together', 'related_product'),     ('rev_also_viewed',     'product')],
                [(None, 'user'),          ('purchase', 'product'),      ('bought_together', 'related_product'),     ('rev_bought_together', 'product')]
                ]
        else:
            raise ValueError(f"Unknown entity type: {entity}")
        return mp_template

    def infer_counter_path_score(self, model, metapath, path_ent_ids, excluded_pids=None):
        modules = model._get_modules(metapath)
        assert len(modules) == len(path_ent_ids) - 1
        uid = path_ent_ids[0]
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        outputs = model._forward(modules, uid_tensor)  # list of tensor of [1, d]

        layer_logprobs = []
        for i, module in enumerate(modules):
            et_vecs = model.embedding(module.et_name)
            scores = torch.matmul(outputs[i], et_vecs.t())  # [1, vocab_size]
            logprobs = F.log_softmax(scores[0], dim=0)  # [vocab_size, ]
            valid_et_ids = torch.LongTensor([path_ent_ids[i+1]]).to(self.device)
            et_logprobs = logprobs.index_select(0, valid_et_ids)
            layer_logprobs.append(et_logprobs)
        return layer_logprobs

    def save_intermediate_results(self, filename='temp/cluster_counter_path_score.pkl'):
        temp_filename = filename + '.tmp'
        with open(temp_filename, 'wb') as file:
            pickle.dump(self.counter_scores, file)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_filename, filename)

    def related_ent_counterfactuals_scores(self, args, save_interval=100, save_path='cluster_counter_path_score.pkl'):
        model = create_symbolic_model(args, self.kg, train=False)
        recommended_product_id = self.recommendation_path[0][-1]
        product_related_ents = self.kg_info.ent2related_ent.get('product', set()) - {'user'}
        user_id = self.recommendation_path[0][0]  # Assuming user_id is the first element
        user_purchases = self.kg.G['user'].get(user_id, {}).get('purchase', [])
        filename = f'tmp/neighbor_entities_{recommended_product_id}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Deserialize the dictionary using pickle.load
                neighboring_entities = pickle.load(file)
        #TODO: fill the flow
        else:
            filter = EntityFilter()
            neighboring_entities = filter(recommended_product_id, force_community_filter=True, print_report=True)
            with open(filename, 'wb') as file:
                pickle.dump(neighboring_entities, file)


        for purchased_product in tqdm(user_purchases, desc="Processing Purchased Products: "):
            counter_scores = {}
            for entity in product_related_ents:
                print(entity)
                related_ent_ids = neighboring_entities[entity]
                counter_metapaths = self.find_counter_mp(entity)

                mp_id = 0
                for counter_mp in tqdm(counter_metapaths, desc=f"Processing Metapaths for Entity: {entity}", leave=False):
                    mp_temp_dict = {'mp': counter_mp}
                    for rel_ent_id in tqdm(related_ent_ids, desc="Processing IDs in entity"):
                        path_ent_ids = [
                            self.recommendation_path[0][0],  # user_id
                            purchased_product,
                            rel_ent_id,
                            self.recommendation_path[0][-1]  # Assuming last element is product_id
                        ]
                        layer_logprobs = self.infer_counter_path_score(model, counter_mp, path_ent_ids, excluded_pids=None)

                        mp_temp_dict[rel_ent_id] = {'log_probs': [item.detach().cpu().numpy()[0] for item in layer_logprobs]}

                    if entity not in counter_scores:
                        counter_scores[entity] = {}
                    if mp_id not in counter_scores[entity]:
                        counter_scores[entity][mp_id] = []
                    counter_scores[entity][mp_id].append(mp_temp_dict)
                    mp_id += 1


        return counter_scores

class CounterfactualFramework:
    def __init__(self, args, recommendations, path_to_analyize) -> None:
        cluster_counter = ClusterCounterfactuals(path_to_analyize)
        counter_scores = cluster_counter.related_ent_counterfactuals_scores(args)
        self.ser_top_10_recoms = recommendations
        self.cluster_path_scores = counter_scores
        self.user_top_10_recoms = recommendations[:10]
        self.positive_ids = {}
        self.cluster_counterfactual_analysis()
        
        
    def cluster_counterfactual_analysis(self):
        # List of paths: ([0, 261, 849], [-4.5635014, -3.4569123], ['mentions', 'rev_described_by'])
        recommended_paths_scores = [item[1] for item in self.user_top_10_recoms]
        min_score = sorted([np.mean(item) for item in recommended_paths_scores], reverse=True)[:10][-1]
        
        
        for entity in self.cluster_path_scores:
            for mpid, scores in self.cluster_path_scores[entity].items():
                metapath_scores = scores.copy()
                for mp_scores in metapath_scores:
                    mp_scores.pop('mp', None)
                    for key in mp_scores:
                        mp_scores[key]['mean'] = np.mean(mp_scores[key]['log_probs'])
                    sorted_entity_ids_scores = dict(sorted(mp_scores.items(), key=lambda item: item[1]['mean'], reverse=True))
                    # Maybe you'd want to make it a dicitonary
                    pos_entity_ids_scores = [(ent_id, scores) for ent_id, scores in sorted_entity_ids_scores.items() if scores['mean'] > min_score]
                    self.positive_ids.setdefault(entity, []).extend(pos_entity_ids_scores)

    
    def counterfactual_explanation(self):
        print('The recommended product would still be selected if :')
        print('The Recommended product had any of the following features:')
        for entity in ['category', 'brand', 'word']:
            entity_top_10 = self.positive_ids[entity][:10]
            entity_top_10_ids = [details[0] for details in entity_top_10]
            print(entity)
            print(entity_top_10_ids)
        print('And was also_viewed, bought_together, also_bought with: ')
        entity_top_10 = self.positive_ids['related_product'][:10]
        entity_top_10_ids = [details[0] for details in entity_top_10]
        print(entity_top_10_ids)



