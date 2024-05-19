import os
import pickle
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append('CAFE')

from CAFE import utils
from CAFE.execute_neural_symbol import create_symbolic_model, MetaProgramExecutor, create_heuristic_program, infer_paths
from counterfactual_generator import EntityFilter
from kg_info import KGENtitiesRelationsInfo



class ClusterCounterfactuals:
    def __init__(self, recommendation_path):
        self.kg = utils.load_kg('beauty')
        self.kg_info = KGENtitiesRelationsInfo()
        self.recommendation_path = recommendation_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.counter_scores = {}

    def find_counter_mp(self, entity):
        # Define metapath templates for different entities
        entity_to_mp = {
            'brand': [[(None, 'user'), ('purchase', 'product'), ('produced_by', 'brand'), ('rev_produced_by', 'product')]],
            'category': [[(None, 'user'), ('purchase', 'product'), ('belongs_to', 'category'), ('rev_belongs_to', 'product')]],
            'word': [[(None, 'user'), ('purchase', 'product'), ('described_by', 'word'), ('rev_described_by', 'product')]],
            'related_product': [
                # Multiple metapaths for related_product
                [(None, 'user'), ('purchase', 'product'), (relation, 'related_product'), (reverse_relation, 'product')]
                for relation, reverse_relation in [
                    ('also_bought', 'rev_also_bought'),
                    ('also_bought', 'rev_also_viewed'),
                    ('also_bought', 'rev_bought_together'),
                    ('also_viewed', 'rev_also_bought'),
                    ('also_viewed', 'rev_also_viewed'),
                    ('also_viewed', 'rev_bought_together'),
                    ('bought_together', 'rev_also_bought'),
                    ('bought_together', 'rev_also_viewed'),
                    ('bought_together', 'rev_bought_together')
                ]
            ]
        }
        return entity_to_mp.get(entity, ValueError(f"Unknown entity type: {entity}"))

    def infer_counter_path_score(self, model, metapath, path_ent_ids, excluded_pids=None):
        # Ensure the number of modules matches the number of path entity IDs minus one.
        modules = model._get_modules(metapath)
        assert len(modules) == len(path_ent_ids) - 1, "Mismatch in the number of modules and path entity IDs."

        # Prepare the input tensor and perform a forward pass through the model.
        uid = path_ent_ids[0]
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        outputs = model._forward(modules, uid_tensor)  # list of tensor of shape [1, d]

        # Compute log probabilities for each module output.
        layer_logprobs = []
        for i, module in enumerate(modules):
            et_vecs = model.embedding(module.et_name)
            scores = torch.matmul(outputs[i], et_vecs.t())  # Shape: [1, vocab_size]
            logprobs = F.log_softmax(scores[0], dim=0)  # Shape: [vocab_size]
            valid_et_ids = torch.LongTensor([path_ent_ids[i+1]]).to(self.device)
            et_logprobs = logprobs.index_select(0, valid_et_ids)
            layer_logprobs.append(et_logprobs)

        return layer_logprobs


    def related_ent_counterfactuals_scores(self, args):
        recommended_product_id = self.recommendation_path[0][-1]
        user_id = self.recommendation_path[0][0]

        scores_path = self.get_score_filename(user_id, recommended_product_id)

        if os.path.exists(scores_path):
            return self.load_scores(scores_path)
        
        else:
            model = self.initialize_model(args)
            user_purchases = self.get_user_purchases(user_id)
            neighboring_entities = self.load_or_create_neighboring_entities(recommended_product_id)
            counter_scores = self.process_purchases(user_purchases, neighboring_entities, model)
            # Save scores for future use
            self.save_scores(user_id, recommended_product_id, counter_scores)

        return counter_scores

    def save_scores(self, user_id, recommended_product_id, scores):
        filename = self.get_score_filename(user_id, recommended_product_id)
        with open(filename, 'wb') as file:
            pickle.dump(scores, file)

    def load_scores(self, scores_path):
        with open(scores_path, 'rb') as file:
            return pickle.load(file)

    def get_score_filename(self, user_id, recommended_product_id):
        return f"tmp/raw_counter_scores_{user_id}_{recommended_product_id}.pkl"

    def initialize_model(self, args):
        return create_symbolic_model(args, self.kg, train=False)

    def get_user_purchases(self, user_id):
        return self.kg.G['user'].get(user_id, {}).get('purchase', [])

    def load_or_create_neighboring_entities(self, recommended_product_id):
        filename = f'tmp/neighbor_entities_{recommended_product_id}.pkl'
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            filter = EntityFilter()
            neighboring_entities = filter(recommended_product_id, force_community_filter=True, print_report=True)
            with open(filename, 'wb') as file:
                pickle.dump(neighboring_entities, file)
            return neighboring_entities

    def process_purchases(self, user_purchases, neighboring_entities, model):
        counter_scores = {}
        for purchased_product in tqdm(user_purchases, desc="Processing Purchased Products"):
            self.process_each_product(purchased_product, neighboring_entities, model, counter_scores)
        return counter_scores

    def process_each_product(self, purchased_product, neighboring_entities, model, counter_scores):
        product_related_ents = self.kg_info.ent2related_ent.get('product', set()) - {'user'}
        for entity in product_related_ents:
            related_ent_ids = neighboring_entities[entity]
            counter_metapaths = self.find_counter_mp(entity)
            self.evaluate_metapaths(purchased_product, entity, counter_metapaths, related_ent_ids, model, counter_scores)

    def evaluate_metapaths(self, purchased_product, entity, counter_metapaths, related_ent_ids, model, counter_scores):
        for mp_id, counter_mp in enumerate(tqdm(counter_metapaths, desc=f"Processing Metapaths for Entity: {entity}", leave=False)):
            mp_temp_dict = {'mp': counter_mp}
            self.score_metapaths(purchased_product, related_ent_ids, counter_mp, model, mp_temp_dict)
            if entity not in counter_scores:
                counter_scores[entity] = {}
            if mp_id not in counter_scores[entity]:
                counter_scores[entity][mp_id] = []
            counter_scores[entity][mp_id].append(mp_temp_dict)

    def score_metapaths(self, purchased_product, related_ent_ids, counter_mp, model, mp_temp_dict):
        for rel_ent_id in tqdm(related_ent_ids, desc="Processing IDs in entity"):
            path_ent_ids = [
                self.recommendation_path[0][0],  # user_id
                purchased_product,
                rel_ent_id,
                self.recommendation_path[0][-1]  # product_id
            ]
            layer_logprobs = self.infer_counter_path_score(model, counter_mp, path_ent_ids)
            mp_temp_dict[rel_ent_id] = {'log_probs': [item.detach().cpu().numpy()[0] for item in layer_logprobs]}

class CounterfactualFramework:
    def __init__(self, args, recommendations, path_to_analyze):
        cluster_counter = ClusterCounterfactuals(path_to_analyze)
        self.recommended_path = path_to_analyze
        counter_scores = cluster_counter.related_ent_counterfactuals_scores(args)
        self.ser_top_10_recoms = recommendations
        self.cluster_path_scores = counter_scores
        self.user_top_10_recoms = recommendations[:10]
        self.id2entity = self.load_ids_dictionary()
        self.kg = utils.load_kg('beauty')
        self.positive_ids = {}
        self.kg_info = KGENtitiesRelationsInfo()
        self.analyze_counterfactual_scores()

    def analyze_counterfactual_scores(self):
        min_score = self.calculate_min_score()
        self.aggregate_positive_ids(min_score)

    def calculate_min_score(self):
        recommended_paths_scores = [item[1] for item in self.user_top_10_recoms]
        return sorted([np.mean(item) for item in recommended_paths_scores], reverse=True)[:10][-1]

    def aggregate_positive_ids(self, min_score):
        for entity, scores in self.cluster_path_scores.items():
            for mpid, mp_scores_list in scores.items():
                for mp_scores in mp_scores_list:
                    self.process_metapath_scores(mp_scores, entity, min_score)

    def process_metapath_scores(self, mp_scores, entity, min_score):
        mp_scores.pop('mp', None)
        for key in mp_scores:
            mp_scores[key]['mean'] = np.mean(mp_scores[key]['log_probs'])
        sorted_entity_ids_scores = dict(sorted(mp_scores.items(), key=lambda item: item[1]['mean'], reverse=True))
        pos_entity_ids_scores = [(ent_id, scores) for ent_id, scores in sorted_entity_ids_scores.items() if scores['mean'] > min_score]
        self.positive_ids.setdefault(entity, []).extend(pos_entity_ids_scores)

    def load_ids_dictionary(self):
        filepath = r'CAFE\data\Beauty\id2entity.pickle'
        with open(filepath, 'rb') as file:
            id2entity = pickle.load(file)
        
        id2entity['word'] = id2entity['vocab']
        _ = id2entity.pop('vocab')
        return id2entity


    def counterfactual_explanation(self):
        print('The recommended product would still be selected if:')
        self.print_features_by_entity(['category', 'brand', 'word'])
        self.print_features_by_entity(['related_product'], prefix="And was also_viewed, bought_together, also_bought with: ")

    def print_features_by_entity(self, entities, prefix="The Recommended product had any of the following features:"):
        print(prefix)
        for entity in entities:
            entity_top_10_ids = [details[0] for details in self.positive_ids.get(entity, [])[:10]]
            print(f"{entity.capitalize()}:")
            print(entity_top_10_ids)

    def counterfactual_explanation_textual(self):
        # Extract information from the recommended path
        recommended_product_id = self.recommended_path[0][-1]
        purchased_product_id = self.recommended_path[0][1]
        caused_relation = self.recommended_path[-1][-2]
        caused_entity_type = self.kg_info.relation_info[caused_relation][1]  # Assuming relation_info maps relations to [relation_type, entity_type]
        caused_entity_id = self.recommended_path[0][-2]
        caused_entity_old_id = self.kg_info.new2old_ids[caused_entity_type][caused_entity_id]
        entity_info = self.id2entity[caused_entity_type][caused_entity_old_id]

        # Attributes extraction
        purchased_product_attributes = {
            'word': self.kg.G['product'][purchased_product_id].get('described_by', []),
            'brand': self.kg.G['product'][purchased_product_id].get('produced_by', None),
            'category': self.kg.G['product'][purchased_product_id].get('belongs_to', None)
        }

        recommended_product_attributes = {
            'word': self.kg.G['product'][recommended_product_id].get('described_by', []),
            'brand': self.kg.G['product'][recommended_product_id].get('produced_by', None),
            'category': self.kg.G['product'][recommended_product_id].get('belongs_to', None)
        }


        # Display attributes
        print("Attributes of the purchased product:")
        for entity in ['word', 'brand', 'category']:
            ids = purchased_product_attributes[entity]
            if ids:
                old_ids = [self.kg_info.new2old_ids[entity][id] for id in ids]
                entities = [self.id2entity[entity][old_id] for old_id in old_ids]
                print(f"{entity.capitalize()} Entities:", entities)

        print("\nAttributes of the recommended product:")
        for entity in ['word', 'brand', 'category']:
            ids = recommended_product_attributes[entity]
            if ids: 
                old_ids = [self.kg_info.new2old_ids[entity][id] for id in ids]
                entities = [self.id2entity[entity][old_id] for old_id in old_ids]
                print(f"{entity.capitalize()} Entities:", entities)

        # Plausible Counterfactuals
        print("\nPlausible Counterfactuals:")
        for entity in ['word', 'brand', 'category']:
            attributes_with_scores = self.positive_ids[entity]
            ids = [ent_id[0] for ent_id in attributes_with_scores]
            if ids: 
                old_ids = [self.kg_info.new2old_ids[entity][id] for id in ids]
                entities = [self.id2entity[entity][old_id] for old_id in old_ids]
                print(f"{entity.capitalize()} Entities:", entities)

